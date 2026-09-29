"""BLOOM (``BloomForCausalLM``).

``transformer.{word_embeddings, word_embeddings_layernorm, h[i].{input_layernorm,
self_attention, post_attention_layernorm, mlp}, ln_f}`` and ``lm_head``. Both
sublayers take the residual as an argument and add it *inside* the module
(``dropout_add``), so the module outputs are residual-stream states; the
contributions are the first argument of each ``dropout_add`` call. The
attention does its own arithmetic whatever ``attn_implementation`` says, so
the pattern is its dropout's output and needs no eager load.
The embedding norm has no standard name.
"""

import torch
from jaxtyping import Float
from torch import Tensor
from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomMLP

from ..components import Attention, Layer, Mlp, SourceEProperty

MODEL_TYPES = ("bloom",)

RENAME = {
    "transformer.word_embeddings": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "self_attention": "self_attn",
}


class Layer(Layer):
    """BLOOM's block; returns a tuple, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """BLOOM's attention adds the residual inside: the contribution is what enters ``dropout_add``."""

    # ``_reshape`` splits the fused projection into ``(query, key, value)``,
    # heads first (the keys are transposed for the score matmul only later);
    # the scores are the softmax's input after the mask; the head outputs are
    # the ``bmm`` result, ``[batch * heads, seq, head_dim]``.

    @SourceEProperty("self__reshape_0", attribute="output", select=0, description=Attention.attention_queries.description)
    def attention_queries(self, value) -> Float[Tensor, "batch heads seq qk_head_dim"]:
        return value

    @SourceEProperty("self__reshape_0", attribute="output", select=1, description=Attention.attention_keys.description)
    def attention_keys(self, value) -> Float[Tensor, "batch kv_heads seq head_dim"]:
        return value

    @SourceEProperty("self__reshape_0", attribute="output", select=2, description=Attention.attention_values.description)
    def attention_values(self, value) -> Float[Tensor, "batch kv_heads seq head_dim"]:
        return value

    @SourceEProperty("F_softmax_0", attribute="input", description=Attention.attention_scores.description)
    def attention_scores(self, value) -> Float[Tensor, "batch heads query key"]:
        return value

    @SourceEProperty("torch_bmm_0", attribute="output", description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> Float[Tensor, "batch seq heads head_dim"]:
        batch_heads, seq, head_dim = value.shape
        heads = self._module.num_heads
        return value.view(batch_heads // heads, heads, seq, head_dim).transpose(1, 2)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        batch, seq, heads, head_dim = value.shape
        return value.transpose(1, 2).reshape(batch * heads, seq, head_dim)

    @SourceEProperty(
        "dropout_add_0",
        attribute="input",
        description="What the attention adds to the residual stream: the tensor entering dropout_add",
    )
    def attention_output(self, value) -> Float[Tensor, "batch seq hidden"]:
        return value

    @SourceEProperty(
        "self_attention_dropout_0",
        description="The attention pattern the values are mixed with, [batch, heads, query, key]",
    )
    def attention_probabilities(self, value) -> Float[Tensor, "batch heads query key"]:
        return value


class Mlp(Mlp):
    """BLOOM's MLP adds the residual inside: the contribution is what enters ``dropout_add``."""

    @SourceEProperty(
        "dropout_add_0",
        attribute="input",
        description="What the MLP adds to the residual stream: the tensor entering dropout_add",
    )
    def mlp_output(self, value) -> Float[Tensor, "batch seq hidden"]:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {BloomBlock: Layer, BloomAttention: Attention, BloomMLP: Mlp}
