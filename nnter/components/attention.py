"""`Attention`: a softmax-attention module, its contribution, its pattern and its interior."""

from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor
from nnsight.intervention.envoy import Envoy

from .eproperty import EProperty, SourceEProperty
from .standard import Standard, first_tensor, rewrap


def needs_eager(envoy: Envoy) -> str | None:
    """Why a value read inside the eager attention forward is unavailable, or ``None``."""
    implementation = envoy._module.config._attn_implementation
    if implementation != "eager":
        return f"read inside the eager attention forward, but this model runs {implementation!r}; load with attn_implementation='eager'"
    return None


#: The call every family on transformers' shared attention path makes:
#: ``attention_interface(module, query, key, value, attention_mask, ...)``.
INTERFACE = "attention_interface_1"

#: The reason a family gives for an interface value it has not mapped onto its own arithmetic.
NOT_ON_INTERFACE = (
    "The attention does its own arithmetic rather than transformers' shared "
    "attention interface; not mapped for this family yet"
)


def seq_first(value: torch.Tensor) -> torch.Tensor:
    """``[batch, heads, seq, head_dim]`` <-> ``[batch, seq, heads, head_dim]``, as a view.

    The standard layout of ``attention_head_outputs`` is what the shared
    interface returns, sequence before heads. A family whose own arithmetic
    keeps heads first serves a transposed view on read, so in-place edits
    still land, and transposes back on write; the transpose is its own inverse.
    """
    return value.transpose(1, 2)


def interface_reason(envoy: Envoy) -> str | None:
    return envoy.off_interface()


class Attention(Standard):
    """A softmax-attention module: its contribution, its pattern, and its interior.

    Everything but ``attention_output`` is read inside transformers' shared
    ``eager_attention_forward``, reached through the module's
    ``attention_interface`` call (`INTERFACE`): the queries, keys and values
    it receives, the scores entering its softmax, the pattern leaving it, and
    the per-head outputs it returns. They are unavailable unless the model was
    loaded with ``attn_implementation="eager"``; `off_interface` is the one
    place that decides, so a family with another reason (GPT-2's
    ``reorder_and_upcast_attn``) overrides that method. A family whose
    attention does its own arithmetic redefines the pattern on its own op and
    marks the rest ``unavailable(NOT_ON_INTERFACE)``.

    The pattern is the dropout *after* the softmax, not the softmax itself:
    that is the tensor the values are mixed with on every family, after the
    cast back to the model dtype and, on a model with an attention sink
    (GPT-OSS), after the sink column is dropped.
    """

    def off_interface(self) -> str | None:
        """Why the shared attention interface does not run on this module, or ``None``."""
        return needs_eager(self)

    @SourceEProperty(INTERFACE, attribute="inputs", select=1, description="The queries entering attention, [batch, heads, seq, head_dim]", unavailable=interface_reason)
    def attention_queries(self, value: torch.Tensor) -> Float[Tensor, "batch heads seq qk_head_dim"]:
        """The queries the attention interface receives, ``[batch, heads, seq, head_dim]``.

        After the query projection and, on a family with rotary embeddings,
        after they are applied. Assign to replace them. In-place edits reach
        the model where torch allows them: GPT-2's queries, keys and values are
        split views of one ``c_attn`` tensor and torch refuses to edit those in
        place, so assign there.
        """
        return value

    @SourceEProperty(INTERFACE, attribute="inputs", select=2, description="The keys entering attention, [batch, kv_heads, seq, head_dim]", unavailable=interface_reason)
    def attention_keys(self, value: torch.Tensor) -> Float[Tensor, "batch kv_heads seq head_dim"]:
        """The keys the attention interface receives, ``[batch, kv_heads, seq, head_dim]``.

        Before ``repeat_kv``, so under grouped-query attention the head axis
        is ``num_kv_heads`` wide. Assign to replace them; in-place edits reach
        the model except on GPT-2 (see `attention_queries`).
        """
        return value

    @SourceEProperty(INTERFACE, attribute="inputs", select=3, description="The values entering attention, [batch, kv_heads, seq, head_dim]", unavailable=interface_reason)
    def attention_values(self, value: torch.Tensor) -> Float[Tensor, "batch kv_heads seq head_dim"]:
        """The values the attention interface receives, ``[batch, kv_heads, seq, head_dim]``.

        Before ``repeat_kv``, like the keys. Assign to replace them; in-place
        edits reach the model except on GPT-2 (see `attention_queries`).
        """
        return value

    @SourceEProperty(f"{INTERFACE}.source.nn_functional_softmax_0", attribute="input", description="The attention scores entering the softmax, masked, [batch, heads, query, key]", unavailable=interface_reason)
    def attention_scores(self, value: torch.Tensor) -> Float[Tensor, "batch heads query key"]:
        """The scaled, masked scores entering the softmax, ``[batch, heads, query, key]``.

        ``softmax(attention_scores)`` is ``attention_probabilities`` up to the
        dtype cast. Assign to replace them; in-place edits reach the model.
        """
        return value

    @EProperty(key="output", description="What the attention adds to the residual stream")
    def attention_output(self, value: Any) -> Float[Tensor, "batch seq hidden"]:
        """The attention sublayer's contribution to the residual stream.

        The tensor the block adds to its input, as a tensor even when the
        module returns ``(attn_output, attn_weights)``. On a family whose
        attention adds the residual inside the module, the family's subclass
        reads the pre-residual value instead, so this always means the same
        thing. In-place edits and assignment reach the model.
        """
        return first_tensor(value)

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)

    @SourceEProperty(
        f"{INTERFACE}.source.nn_functional_dropout_0",
        description="The attention pattern the values are mixed with, [batch, heads, query, key]",
        unavailable=interface_reason,
    )
    def attention_probabilities(self, value: torch.Tensor) -> Float[Tensor, "batch heads query key"]:
        """The attention pattern, ``[batch, heads, query, key]``.

        The post-softmax probabilities as the values are mixed with them: in
        the model's dtype, and with an attention sink's column already dropped.
        Rows sum to one (to less than one on a sink model). Read it, edit it in
        place, or assign a tensor of the same shape::

            with model.trace(prompt):
                pattern = model.layers[3].self_attn.attention_probabilities.save()
                model.layers[3].self_attn.attention_probabilities[:, 5] = 0
        """
        return value

    @SourceEProperty(INTERFACE, attribute="output", select=0, description="The per-head outputs before the output projection, [batch, seq, heads, head_dim]", unavailable=interface_reason)
    def attention_head_outputs(self, value: torch.Tensor) -> Float[Tensor, "batch seq heads head_dim"]:
        """Each head's output before they are concatenated and projected, ``[batch, seq, heads, head_dim]``.

        What the attention interface returns; the module reshapes it to
        ``[batch, seq, hidden]`` and applies the output projection to get
        ``attention_output``. Assign to replace it, or edit it in place: the
        interface returns a contiguous tensor on most families and a plain
        transposed view on GPT-2, and torch accepts in-place edits on both.
        """
        return value
