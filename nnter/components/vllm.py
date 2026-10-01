"""The base envoys of a vLLM family: the same values as `nnter.components`, read where vLLM keeps them.

vLLM runs a family through its own implementation, which differs from
transformers' in three ways a value has to absorb:

* **No batch axis.** A request's activations are ``[tokens, hidden]``: every
  prompt token on the prefill, one row on each decode step. The values keep
  nnter's layouts, so they are served as ``[1, tokens, hidden]`` and code
  written against ``[:, -1]`` runs on both engines.
* **Live buffers.** A served tensor is the model's own, and the next fused
  kernel rewrites it after the block has read it, so a saved reference comes
  back holding later data. Every value here is a private copy, handed back
  to the model when the block moves on; in-place edits to the copy and
  assignment both reach the model.
* **The residual stream in two halves.** Most blocks fuse the residual add
  into the next norm and are called with, and return, ``(hidden_states,
  residual)``: the stream is their sum. `FusedLayer` is that block, `Layer`
  one that is called with the stream and returns it.

The attention module hands its queries, keys and values to vLLM's attention
layer (its ``attn`` child) and gets the per-head outputs back, each
``[tokens, heads * head_dim]``; they are that child's inputs and output,
served in nnter's layouts with the heads split out. What the layer computes
between them (the scores, the pattern) is inside its kernel, not Python on
this engine, and is marked unavailable; `StandardizedVLLM.status` says so
like it does for any other value.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import Mediator

from . import attention, layer, mlp
from .attention import HeadOutputs, Keys, Queries, Values
from .eproperty import EProperty, unavailable
from .layer import Residual

#: Why nothing between the queries and the head outputs is readable: the softmax runs inside the engine's kernel.
KERNEL = "computed inside vLLM's attention kernel, which serves nothing between its inputs and its output"


def batched(rows: torch.Tensor, head_dim: int | None = None, heads_first: bool = False) -> torch.Tensor:
    """``[tokens, ...]`` as a private ``[1, tokens, ...]`` copy; with ``head_dim``, the last axis split into heads.

    ``[tokens, heads * head_dim]`` becomes ``[1, tokens, heads, head_dim]``,
    or ``[1, heads, tokens, head_dim]`` with ``heads_first``.
    """
    view = rows.clone().unsqueeze(0)
    if head_dim is None:
        return view
    view = view.unflatten(-1, (-1, head_dim))
    return view.transpose(1, 2) if heads_first else view


def unbatched(value: torch.Tensor, rows: torch.Tensor, name: str, head_dim: int | None = None, heads_first: bool = False) -> torch.Tensor:
    """What `batched` served, back as the ``[tokens, ...]`` the model holds, refusing another shape.

    A written value is spliced into the step the engine is running, among
    other requests' rows. One of the wrong height would reach the next kernel
    as it is, where the mismatch can end the engine and every request in it,
    so it is refused here, in the block that wrote it.
    """
    expected = (1, *rows.shape)
    if head_dim is not None:
        tokens, heads = rows.shape[0], rows.shape[-1] // head_dim
        expected = (1, heads, tokens, head_dim) if heads_first else (1, tokens, heads, head_dim)
    if value.shape != expected:
        raise ValueError(
            f"{name} is {expected} on this request and cannot be replaced by a value of shape "
            f"{tuple(value.shape)}; write into the rows you mean instead (``value[:, positions] = ...``)"
        )
    if head_dim is not None and heads_first:
        value = value.transpose(1, 2)
    return value.reshape(rows.shape).to(rows.dtype)


class Flat(EProperty):
    """An `EProperty` over a ``[tokens, ...]`` tensor, served as a private ``[1, tokens, ...]`` copy.

    Takes what `EProperty` takes (a path for a key, ``select`` for one
    argument of a call). The decorated function receives the batched copy and
    returns the value; its annotation is the layout. In-place edits are handed
    back to the model when the block moves on, and an assignment is checked
    against the rows this request has before it is swapped in.

    Args:
        heads: For a ``[tokens, heads * head_dim]`` tensor, where the head
            axis goes: ``"first"`` serves ``[1, heads, tokens, head_dim]``
            (the queries, keys and values), ``"last"`` ``[1, tokens, heads,
            head_dim]`` (the head outputs). The width of a head is the host
            module's ``head_dim``.
    """

    def __init__(self, *args: Any, heads: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.heads = heads

    def _layout(self, obj: Envoy) -> dict:
        """How this value's rows are laid out for the reader: the keyword arguments of `batched` and `unbatched`."""
        if self.heads is None:
            return {}
        return {"head_dim": obj._module.head_dim, "heads_first": self.heads == "first"}

    def __call__(self, preprocess: Callable) -> "Flat":
        @functools.wraps(preprocess)
        def read(envoy: Envoy, value: torch.Tensor) -> torch.Tensor:
            return preprocess(envoy, batched(value, **self._layout(envoy)))

        super().__call__(read)
        self._transform = self._hand_back
        return self

    def _rows(self, obj: Envoy, value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return unbatched(value, rows, f"{obj.path}.{self.name}", **self._layout(obj))

    def _hand_back(self, obj: Envoy, view: torch.Tensor, raw: Any) -> Any:
        attribute, select = self.path(obj).rsplit(".", 1)[-1], self._selection(obj)
        return self._put(attribute, raw, self._rows(obj, view, self._pick(attribute, raw, select)), select)

    def __set__(self, obj: Envoy, value: torch.Tensor) -> None:
        self._check(obj)
        key = self.path(obj)
        location, attribute, select = self._resolve(obj, key), key.rsplit(".", 1)[-1], self._selection(obj)
        current = Mediator.value(location)
        Mediator.swap(location, self._put(attribute, current, self._rows(obj, value, self._pick(attribute, current, select)), select))


def argument(inputs: tuple, index: int, name: str) -> Any:
    """One argument of a call served as ``(args, kwargs)``, however the caller passed it; ``None`` when absent."""
    args, kwargs = inputs
    return kwargs[name] if name in kwargs else args[index] if index < len(args) else None


def with_argument(inputs: tuple, index: int, name: str, value: Any) -> tuple:
    """``inputs`` with that argument replaced, where the caller passed it."""
    args, kwargs = inputs
    if name in kwargs:
        return args, {**kwargs, name: value}
    return (*args[:index], value, *args[index + 1:]), kwargs


class Layer(layer.Layer):
    """A vLLM block that is called with the residual stream and returns it (GPT-2).

    A family whose block takes the positions first says where the stream is:
    ``layer_input = Flat("inputs", select=1, ...)``.
    """

    def skip_with(self, hidden: torch.Tensor) -> None:
        """Skip this block, handing ``hidden`` (``[1, tokens, hidden]``) on as its residual stream.

        Safe on a request that runs alone. The engine batches whatever is in
        flight into one step, and a skipped block has to answer for all of
        it; see nnsight's vLLM guide before skipping on a shared engine.
        """
        self.skip(hidden.squeeze(0))

    @Flat("input", description="The residual stream entering the block, [1, tokens, hidden]")
    def layer_input(self, value: torch.Tensor) -> Residual:
        return value

    @Flat("output", description="The residual stream leaving the block, [1, tokens, hidden]")
    def layer_output(self, value: torch.Tensor) -> Residual:
        return value


class FusedLayer(layer.Layer):
    """A vLLM block with the residual add fused into the next norm (Llama, Qwen, Gemma, most families).

    It is called ``forward(positions, hidden_states, residual)`` and returns
    ``(hidden_states, residual)``: ``hidden_states`` is what the last sublayer
    produced and ``residual`` the stream before it, so the stream is their
    sum, entering and leaving. The first block is called with ``residual``
    ``None`` and the embeddings as ``hidden_states``.

    The sum is computed for the read, so it is nobody's buffer. An edit goes
    back as a change to ``hidden_states`` (the next norm adds the two), and
    a read that edits nothing changes nothing: the difference it hands back is
    exactly zero.
    """

    #: Where the two halves sit in the block's call: ``(index, name)``.
    HIDDEN = (1, "hidden_states")
    RESIDUAL = (2, "residual")

    def skip_with(self, hidden: torch.Tensor) -> None:
        """Skip this block, handing ``hidden`` (``[1, tokens, hidden]``) on as its residual stream.

        Safe on a request that runs alone. The engine batches whatever is in
        flight into one step, and a skipped block has to answer for all of
        it; see nnsight's vLLM guide before skipping on a shared engine.
        """
        hidden = hidden.squeeze(0)
        self.skip((hidden, torch.zeros_like(hidden)))

    @staticmethod
    def _stream(hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        return (hidden.clone() if residual is None else hidden + residual).unsqueeze(0)

    @staticmethod
    def _edited(hidden: torch.Tensor, residual: torch.Tensor | None, view: torch.Tensor, name: str) -> torch.Tensor:
        """``hidden_states`` carrying the difference between ``view`` and the stream as the model has it."""
        stream = unbatched(view, hidden, name)
        if residual is None:
            return stream
        return hidden + (stream - (hidden + residual))

    # -- the stream entering the block ---------------------------------------------

    @EProperty("inputs", description="The residual stream entering the block: hidden_states + residual, [1, tokens, hidden]")
    def layer_input(self, value: tuple) -> Residual:
        return self._stream(argument(value, *self.HIDDEN), argument(value, *self.RESIDUAL))

    def _with_input(self, inputs: tuple, view: torch.Tensor) -> tuple:
        hidden, residual = argument(inputs, *self.HIDDEN), argument(inputs, *self.RESIDUAL)
        return with_argument(inputs, *self.HIDDEN, self._edited(hidden, residual, view, f"{self.path}.layer_input"))

    @layer_input.postprocess
    def layer_input(self, value: torch.Tensor) -> tuple:
        return self._with_input(self.inputs, value)

    @layer_input.transform
    def layer_input(self, view: torch.Tensor, raw: tuple) -> tuple:
        return self._with_input(raw, view)

    # -- the stream leaving it -------------------------------------------------------

    @EProperty("output", description="The residual stream leaving the block: hidden_states + residual, [1, tokens, hidden]")
    def layer_output(self, value: tuple) -> Residual:
        return self._stream(*value)

    def _with_output(self, output: tuple, view: torch.Tensor) -> tuple:
        hidden, residual = output
        return self._edited(hidden, residual, view, f"{self.path}.layer_output"), residual

    @layer_output.postprocess
    def layer_output(self, value: torch.Tensor) -> tuple:
        return self._with_output(self.output, value)

    @layer_output.transform
    def layer_output(self, view: torch.Tensor, raw: tuple) -> tuple:
        return self._with_output(raw, view)


class Attention(attention.Attention):
    """A vLLM attention module: its contribution, and what goes into and comes out of the engine's attention layer.

    The module projects (and, on a rotary family, rotates) its queries, keys
    and values and calls vLLM's attention layer on them, ``self.attn(q, k,
    v)``, which returns the per-head outputs the output projection then
    mixes. Those four are the ``attn`` child's inputs and output, each
    ``[tokens, heads * head_dim]``, served with the heads split out by the
    module's own ``head_dim``. A family whose attention module names that
    child, or its head width, another way points the values at it.

    They are this step's rows: on a decode step the keys and values are the
    new token's alone, and the earlier ones are in the engine's cache, which
    the kernel reads for itself. The scores and the pattern are inside the
    kernel.
    """

    attention_scores = unavailable(KERNEL)
    attention_probabilities = unavailable(KERNEL)

    @Flat("attn.inputs", select=0, heads="first", description="The queries entering vLLM's attention layer, [1, heads, tokens, head_dim]")
    def attention_queries(self, value: torch.Tensor) -> Queries:
        """The queries the attention layer receives, after the projection, any query norm and the rotary embedding."""
        return value

    @Flat("attn.inputs", select=1, heads="first", description="The keys entering vLLM's attention layer, [1, kv_heads, tokens, head_dim]")
    def attention_keys(self, value: torch.Tensor) -> Keys:
        """This step's keys as the attention layer receives them; ``num_kv_heads`` wide under grouped-query attention."""
        return value

    @Flat("attn.inputs", select=2, heads="first", description="The values entering vLLM's attention layer, [1, kv_heads, tokens, head_dim]")
    def attention_values(self, value: torch.Tensor) -> Values:
        """This step's values as the attention layer receives them; ``num_kv_heads`` wide under grouped-query attention."""
        return value

    @Flat("attn.output", heads="last", description="The per-head outputs before the output projection, [1, tokens, heads, head_dim]")
    def attention_head_outputs(self, value: torch.Tensor) -> HeadOutputs:
        """Each head's output as the attention layer returns it, before the output projection mixes them."""
        return value

    @Flat("output", description="What the attention adds to the residual stream, [1, tokens, hidden]")
    def attention_output(self, value: torch.Tensor) -> Residual:
        return value


class Mlp(mlp.Mlp):
    """A vLLM feed-forward module: its contribution is its output."""

    @Flat("output", description="What the MLP adds to the residual stream, [1, tokens, hidden]")
    def mlp_output(self, value: torch.Tensor) -> Residual:
        return value
