"""JetMoE (``JetMoeForCausalLM``).

Llama's tree with the attention under its own name: ``model.{embed_tokens,
layers[i].{input_layernorm, self_attention, post_attention_layernorm, mlp}, norm}``
and ``lm_head``, the residual added in the block, which returns a bare tensor.
``self_attention`` is aliased ``self_attn``.

The attention is a mixture of attention heads: a router picks
``num_experts_per_tok`` experts per token, each expert projects that token's
queries for ``num_key_value_heads`` heads (``experts.map``) and later projects their
outputs back (``experts.reduce``, in place of an ``o_proj``). Keys and values come
from one shared ``kv_proj`` and are tiled (``repeat``, not interleaved) to every
query head before the shared eager forward, so the interface sees ``num_heads``
key/value heads and interface head ``h`` is routing slot ``h // num_kv_heads`` over
kv head ``h % num_kv_heads``: which expert's parameters a head's query came from
varies by token. The attention returns ``(attn_output, attn_weights,
router_logits)``. The MLP is a mixture of experts that returns the routed hidden
states plus a bias as one tensor.
"""

from transformers.models.jetmoe.modeling_jetmoe import JetMoeAttention, JetMoeDecoderLayer, JetMoeMoE

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("jetmoe",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "self_attention": "self_attn",
}


class Layer(Layer):
    """JetMoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """JetMoE's mixture of attention heads; the shared eager forward, a 3-tuple whose first element is the contribution, and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """A mixture of experts: the module returns the routed hidden states (plus its bias) as a bare tensor, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {JetMoeDecoderLayer: Layer, JetMoeAttention: Attention, JetMoeMoE: Mlp}
