"""DeepSeek-V2 (``DeepseekV2ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. Multi-head latent attention: queries and keys are ``qk_head_dim`` wide (``qk_nope_head_dim + qk_rope_head_dim``), values ``v_head_dim``, and the interface sees ``num_heads`` key/value heads whatever ``num_key_value_heads`` says. The first ``first_k_dense_replace`` blocks have a dense MLP, the rest a mixture of experts.
"""

from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention, DeepseekV2DecoderLayer, DeepseekV2MLP, DeepseekV2Moe

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("deepseek_v2",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """DeepSeek-V2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """DeepSeek-V2's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """DeepSeek's dense MLP or mixture of experts; both return the hidden states, and the residual is added in the block."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {DeepseekV2DecoderLayer: Layer, DeepseekV2Attention: Attention, DeepseekV2MLP: Mlp, DeepseekV2Moe: Mlp}
