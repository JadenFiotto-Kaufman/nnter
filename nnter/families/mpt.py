"""MPT (``MptForCausalLM``).

``transformer.{wte, blocks[i].{norm_1, attn, norm_2, ffn, resid_attn_dropout},
norm_f}`` and ``lm_head``. The block adds the attention's output itself, but
the MLP takes the residual and adds it inside, so its contribution is the
tensor before that add (its dropout's output). The attention does its own
arithmetic whatever ``attn_implementation`` says, so the pattern is its
dropout's output and needs no eager load.
"""

from typing import TYPE_CHECKING

from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptMLP

from ..components import (
    Attention, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Residual, SourceEProperty, Values, seq_first,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("mpt",)

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.blocks": "layers",
    "transformer.norm_f": "norm",
    "norm_1": "input_layernorm",
    "norm_2": "post_attention_layernorm",
    "attn": "self_attn",
    "ffn": "mlp",
}


class Layer(Layer):
    """MPT's block; returns a tuple, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """MPT's attention: the residual is added in the block; the pattern is its own dropout."""

    # The three ``*_states`` bindings are the projections reshaped heads first
    # (no grouped-query attention); the scores are the softmax's input after
    # the mask, upcast to float32; the head outputs are the second matmul,
    # heads first.

    @SourceEProperty("query_states_0", description=Attention.attention_queries.description)
    def attention_queries(self, value) -> Queries:
        return value

    @SourceEProperty("key_states_0", description=Attention.attention_keys.description)
    def attention_keys(self, value) -> Keys:
        return value

    @SourceEProperty("value_states_0", description=Attention.attention_values.description)
    def attention_values(self, value) -> Values:
        return value

    @SourceEProperty("nn_functional_softmax_0", attribute="input", description=Attention.attention_scores.description)
    def attention_scores(self, value) -> Pattern:
        return value

    @SourceEProperty("torch_matmul_1", description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> HeadOutputs:
        return seq_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        return seq_first(value)

    @SourceEProperty(
        "nn_functional_dropout_0",
        description="The attention pattern the values are mixed with, [batch, heads, query, key]",
    )
    def attention_probabilities(self, value) -> Pattern:
        return value


class Mlp(Mlp):
    """MPT's MLP adds the residual inside: the contribution is its dropout's output, before the add."""

    @SourceEProperty(
        "F_dropout_0",
        description="What the MLP adds to the residual stream: the tensor before the residual is added inside",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MptBlock: Layer, MptAttention: Attention, MptMLP: Mlp}


# -- sizes: what MPT's config calls them --------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``expansion_ratio`` times the hidden size."""
    return int(model.config.expansion_ratio * model.hidden_size)
