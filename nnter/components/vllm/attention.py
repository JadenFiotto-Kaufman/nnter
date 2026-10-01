"""`Attention`: a vLLM attention module, its contribution and what crosses the engine's attention layer."""

from __future__ import annotations

import torch

from .. import attention as standard
from ..attention import HeadOutputs, Keys, Queries, Values
from ..eproperty import unavailable
from ..layer import Residual
from .flat import Flat

#: Why nothing between the queries and the head outputs is readable: the softmax runs inside the engine's kernel.
KERNEL = "computed inside vLLM's attention kernel, which serves nothing between its inputs and its output"


class Attention(standard.Attention):
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
