"""Qwen2-MoE (``Qwen2MoeForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. The MLP is a sparse mixture of experts with a shared expert.
"""

from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeAttention, Qwen2MoeDecoderLayer, Qwen2MoeSparseMoeBlock

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("qwen2_moe",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Qwen2-MoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen2-MoE's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """A mixture of experts: the module returns the routed hidden states (a bare tensor on this transformers), so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen2MoeDecoderLayer: Layer, Qwen2MoeAttention: Attention, Qwen2MoeSparseMoeBlock: Mlp}
