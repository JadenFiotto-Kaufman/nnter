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

Everything inside the attention kernel (the scores, the pattern) is not
Python on this engine and is marked unavailable; `StandardizedVLLM.status`
says so like it does for any other value.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import Mediator

from . import attention, layer, mlp
from .eproperty import EProperty, unavailable
from .layer import Residual

#: Why nothing between the queries and the head outputs is readable: the softmax runs inside the engine's kernel.
KERNEL = "computed inside vLLM's attention kernel, which serves nothing between its inputs and its output"
#: A value the kernel's inputs or output would give, not mapped onto vLLM's forwards yet.
NOT_MAPPED = "not mapped onto vLLM's attention forward yet; read it on the transformers engine"


def batched(rows: torch.Tensor) -> torch.Tensor:
    """``[tokens, ...]`` as a private ``[1, tokens, ...]`` copy."""
    return rows.clone().unsqueeze(0)


def unbatched(value: torch.Tensor, rows: torch.Tensor, name: str) -> torch.Tensor:
    """``[1, tokens, ...]`` back as the ``[tokens, ...]`` the model holds, refusing another shape.

    A written value is spliced into the step the engine is running, among
    other requests' rows. One of the wrong height would reach the next kernel
    as it is, where the mismatch can end the engine and every request in it,
    so it is refused here, in the block that wrote it.
    """
    if value.shape != (1, *rows.shape):
        raise ValueError(
            f"{name} is {(1, *rows.shape)} on this request and cannot be replaced by a value of shape "
            f"{tuple(value.shape)}; write into the rows you mean instead (``value[:, positions] = ...``)"
        )
    return value.squeeze(0).to(rows.dtype)


class Flat(EProperty):
    """An `EProperty` over a ``[tokens, ...]`` tensor, served as a private ``[1, tokens, ...]`` copy.

    Takes what `EProperty` takes (a path for a key, ``select`` for one
    argument of a call). The decorated function receives the batched copy and
    returns the value; its annotation is the layout. In-place edits are handed
    back to the model when the block moves on, and an assignment is checked
    against the rows this request has before it is swapped in.
    """

    def __call__(self, preprocess: Callable) -> "Flat":
        @functools.wraps(preprocess)
        def read(envoy: Envoy, value: torch.Tensor) -> torch.Tensor:
            return preprocess(envoy, batched(value))

        super().__call__(read)
        self._transform = self._hand_back
        return self

    def _hand_back(self, obj: Envoy, view: torch.Tensor, raw: Any) -> Any:
        attribute, select = self.path(obj).rsplit(".", 1)[-1], self._selection(obj)
        rows = self._pick(attribute, raw, select)
        return self._put(attribute, raw, unbatched(view, rows, f"{obj.path}.{self.name}"), select)

    def __set__(self, obj: Envoy, value: torch.Tensor) -> None:
        self._check(obj)
        key = self.path(obj)
        location, attribute, select = self._resolve(obj, key), key.rsplit(".", 1)[-1], self._selection(obj)
        current = Mediator.value(location)
        rows = self._pick(attribute, current, select)
        Mediator.swap(location, self._put(attribute, current, unbatched(value, rows, f"{obj.path}.{self.name}"), select))


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
    """A vLLM attention module: its contribution is its output; the kernel's interior is not served."""

    attention_queries = unavailable(NOT_MAPPED)
    attention_keys = unavailable(NOT_MAPPED)
    attention_values = unavailable(NOT_MAPPED)
    attention_scores = unavailable(KERNEL)
    attention_probabilities = unavailable(KERNEL)
    attention_head_outputs = unavailable(NOT_MAPPED)

    @Flat("output", description="What the attention adds to the residual stream, [1, tokens, hidden]")
    def attention_output(self, value: torch.Tensor) -> Residual:
        return value


class Mlp(mlp.Mlp):
    """A vLLM feed-forward module: its contribution is its output."""

    @Flat("output", description="What the MLP adds to the residual stream, [1, tokens, hidden]")
    def mlp_output(self, value: torch.Tensor) -> Residual:
        return value
