"""Falcon (``FalconForCausalLM``), the 7B layout: parallel attention, no alibi.

``transformer.{word_embeddings, h[i].{input_layernorm, self_attention, mlp},
ln_f}`` and ``lm_head``. One norm feeds both sublayers and the block sums
``x + attn + mlp``, but it does so by adding the attention output *into the
MLP's output tensor in place*, so ``mlp_output`` reads a copy taken as the
MLP returns, and a transform carries edits to it back into the model. The attention does its own arithmetic; with
no alibi the pattern is its first softmax, which has no dropout after it.
The alibi layout (``F_softmax_1`` then ``self_attention_dropout_0``) and the
40B layout (``ln_attn`` / ``ln_mlp``) are not covered yet.
"""

import torch
from jaxtyping import Float
from torch import Tensor
from transformers.models.falcon.modeling_falcon import FalconAttention, FalconDecoderLayer, FalconMLP

from ..components import Attention, EProperty, Layer, Mlp, SourceEProperty, first_tensor, needs_eager, rewrap, seq_first

MODEL_TYPES = ("falcon",)

RENAME = {
    "transformer.word_embeddings": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "self_attention": "self_attn",
}


class Layer(Layer):
    """Falcon's parallel block; returns a tuple, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """Falcon's attention: the residual is added in the block; the pattern is its softmax (no dropout follows)."""

    # Without alibi, queries and keys are ``apply_rotary_pos_emb``'s two
    # returns and the values the binding just before it (so read the values
    # before the queries or keys in one trace: they bind first); keys and
    # values are ``num_kv_heads`` wide (1 under multi-query). The scores are
    # the softmax's input after the mask; the head outputs are the
    # ``attention_scores @ value_layer`` binding, heads first.

    @SourceEProperty("apply_rotary_pos_emb_0", attribute="output", select=0, description=Attention.attention_queries.description, unavailable=needs_eager)
    def attention_queries(self, value) -> Float[Tensor, "batch heads seq qk_head_dim"]:
        return value

    @SourceEProperty("apply_rotary_pos_emb_0", attribute="output", select=1, description=Attention.attention_keys.description, unavailable=needs_eager)
    def attention_keys(self, value) -> Float[Tensor, "batch kv_heads seq head_dim"]:
        return value

    @SourceEProperty("value_layer_0", description=Attention.attention_values.description, unavailable=needs_eager)
    def attention_values(self, value) -> Float[Tensor, "batch kv_heads seq head_dim"]:
        return value

    @SourceEProperty("F_softmax_0", attribute="input", description=Attention.attention_scores.description, unavailable=needs_eager)
    def attention_scores(self, value) -> Float[Tensor, "batch heads query key"]:
        return value

    @SourceEProperty("attn_output_1", description=Attention.attention_head_outputs.description, unavailable=needs_eager)
    def attention_head_outputs(self, value) -> Float[Tensor, "batch seq heads head_dim"]:
        return seq_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        return seq_first(value)

    @SourceEProperty(
        "F_softmax_0",
        description="The attention pattern the values are mixed with, [batch, heads, query, key]",
        unavailable=lambda self: needs_eager(self) or (
            "this checkpoint uses alibi, whose attention path is not covered yet"
            if self._module.config.alibi else None
        ),
    )
    def attention_probabilities(self, value) -> Float[Tensor, "batch heads query key"]:
        return value


class Mlp(Mlp):
    """Falcon's MLP: the block later adds the attention into this tensor in place, so read a copy.

    The copy keeps a saved read honest. So that in-place edits to it still
    reach the model, a transform hands a copy of the edited copy back to be
    swapped in once the block is done with the read; the second copy is what
    keeps the user's tensor clean when the block then adds into it.
    """

    @EProperty(key="output", description="What the MLP adds to the residual stream (a copy, since the block adds the attention into the live tensor in place)")
    def mlp_output(self, value) -> Float[Tensor, "batch seq hidden"]:
        return first_tensor(value).clone()

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value)

    @mlp_output.transform
    def mlp_output(self, value, raw):
        # Fires on the model side, after the read. The module returns a bare
        # tensor, so ``raw`` needs no rebuilding around the edited copy.
        return value.clone()


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FalconDecoderLayer: Layer, FalconAttention: Attention, FalconMLP: Mlp}
