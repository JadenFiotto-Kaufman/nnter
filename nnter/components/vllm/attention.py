"""`Attention`: a vLLM attention module, its contribution, what crosses the engine's attention layer, and the pattern."""

from __future__ import annotations

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import Mediator

from .. import attention as standard
from ..attention import HeadOutputs, Keys, Pattern, Queries, Values
from ..eproperty import DerivedEProperty, Unavailable
from ..layer import Residual
from .flat import Flat

#: Why the scores and the pattern are not there when part of the sequence is in the cache.
IN_CACHE = (
    "recomputed from this step's queries and keys, and this request's first {cached} tokens are in vLLM's KV cache "
    "(a decode step, or a prompt whose prefix an earlier request left there), which only its kernel reads; read it on "
    "a step that computes the whole prompt, and load with enable_prefix_caching=False to read it on prompts that "
    "share a prefix"
)


def cached(envoy: Envoy) -> int | None:
    """How many of this request's tokens are in vLLM's KV cache before this step's; ``None`` where the backend does not say.

    ``0`` on a prefill of the whole prompt. Read off the attention metadata
    vLLM builds for the step that is running (``seq_lens``, each request's
    length with this step's tokens counted) at the row nnsight gives this
    request, so it is per request under continuous batching. A decode step,
    a prompt whose prefix vLLM had cached, and every chunk but the first of
    a chunked prefill start past position 0.
    """
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):  # micro-batched: one per micro-batch
        return None
    lengths = getattr(metadata.get(envoy.attn._module.layer_name), "seq_lens", None)
    if lengths is None:
        return None
    mediator = Mediator.current(envoy.path)
    return int(lengths[mediator.row]) - mediator.batch_group[1]


def recompute(self: "Attention", name: str) -> torch.Tensor:
    """The scores, recomputed from this step's queries and keys; `Unavailable` (naming ``name``) unless the step holds every key.

    The check is made once the queries arrive, inside the step that serves
    them: a block asks for a step's value while the step before it is still
    running, so nothing known at the ask says which step the read lands in.
    """
    queries, keys = self.attention_queries, self.attention_keys
    count = cached(self)
    if count is None:
        raise Unavailable(
            f"{self.path}.{name} is not available: it needs each request's length, and this attention backend's "
            "metadata does not give it"
        )
    if count:
        raise Unavailable(f"{self.path}.{name} is not available: {IN_CACHE.format(cached=count)}")
    keys = keys.repeat_interleave(queries.shape[1] // keys.shape[1], dim=1)  # grouped-query: one key head per group
    layer = self.attn._module
    values = queries @ keys.transpose(-1, -2) * layer.impl.scale
    if layer.impl.logits_soft_cap:
        values = layer.impl.logits_soft_cap * torch.tanh(values / layer.impl.logits_soft_cap)
    position = torch.arange(values.shape[-1], device=values.device)
    behind = position[:, None] - position[None, :]  # how far behind the query each key is
    slopes = getattr(layer.impl, "alibi_slopes", None)
    if slopes is not None:
        values = values - slopes.to(values)[:, None, None] * behind
    hidden = behind < 0
    if layer.sliding_window is not None:
        hidden = hidden | (behind >= layer.sliding_window)
    return values.masked_fill(hidden, float("-inf"))


def scores(self: "Attention") -> Pattern:
    """The scaled, masked scores the kernel's softmax runs on, ``[1, heads, query, key]``, recomputed.

    ``queries @ keys.T * scale``, softcapped where the layer softcaps, plus
    the ALiBi bias where the layer has slopes, with the keys a query may not
    see (later tokens, and tokens past the layer's sliding window) at
    ``-inf``. The scale, the softcap, the slopes and the window are the ones
    vLLM's attention layer was built with. The ALiBi bias is each head's
    slope times how far behind the query the key is; transformers' families
    write the same bias from another origin, which moves a query's scores by
    one constant and leaves the pattern alone.
    """
    return recompute(self, "attention_scores")


def probabilities(self: "Attention") -> Pattern:
    """The attention pattern, ``[1, heads, query, key]``: the softmax of `attention_scores`, in float32 then the model's dtype."""
    values = recompute(self, "attention_probabilities")
    return values.float().softmax(-1).to(values.dtype)


class Attention(standard.Attention):
    """A vLLM attention module: its contribution, what goes into and comes out of the engine's attention layer, and the pattern.

    The module projects (and, on a rotary family, rotates) its queries, keys
    and values and calls vLLM's attention layer on them, ``self.attn(q, k,
    v)``, which returns the per-head outputs the output projection then
    mixes. Those four are the ``attn`` child's inputs and output, each
    ``[tokens, heads * head_dim]``, served with the heads split out by the
    module's own ``head_dim``. A family whose attention module names that
    child, or its head width, another way points the values at it.

    They are this step's rows: on a decode step the keys and values are the
    new token's alone, and the earlier ones are in the engine's cache, which
    the kernel reads for itself.

    The scores and the pattern are computed inside the kernel, which serves
    nothing between its inputs and its output, so here they are *recomputed*
    from the queries and keys, with the layer's own scale, softcap and
    window. That makes them read-only (the kernel never takes a pattern: edit
    the queries or keys to change what a head attends to, or its head outputs
    to change what it wrote) and there only on a step that computes the
    whole sequence so far, a prefill from position 0, where the step holds
    every key. A step that starts past it (a decode step, a prompt whose
    prefix vLLM had cached) raises `Unavailable` (`cached`).
    """

    attention_scores = DerivedEProperty(
        scores,
        description="The scores entering the softmax, [1, heads, query, key], recomputed from the queries and keys; on an uncached prefill only, read-only",
    )
    attention_probabilities = DerivedEProperty(
        probabilities,
        description="The attention pattern, [1, heads, query, key], recomputed from the queries and keys; on an uncached prefill only, read-only",
    )

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
