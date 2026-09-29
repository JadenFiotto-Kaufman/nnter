"""Qwen3-Next (``Qwen3NextForCausalLM``).

Llama's tree with a hybrid block: three blocks in four carry a gated DeltaNet
mixer, ``linear_attn``, the fourth ordinary attention, ``self_attn``
(``config.layer_types``). Each block has one or the other, never both, so on
a linear block every ``self_attn`` value is reported missing and the linear
values live at ``layers[i].linear_attn`` (see `nnter.LinearAttention`). The MLP is a mixture of experts (a dense MLP class exists for the shared expert).
"""

from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention, Qwen3NextDecoderLayer, Qwen3NextGatedDeltaNet, Qwen3NextMLP, Qwen3NextSparseMoeBlock

from ..components import Attention, Layer, LinearAttention, Mlp

MODEL_TYPES = ("qwen3_next",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Qwen3-Next's block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen3-Next's softmax attention (one block in four); the shared eager forward, so the base holds."""


class LinearAttention(LinearAttention):
    """Qwen3-Next's gated DeltaNet mixer; transformers' pure-torch chunked rule, so the base holds."""


class Mlp(Mlp):
    """Qwen3-Next's mixture of experts or dense MLP; both return the hidden states, added in the block."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen3NextDecoderLayer: Layer, Qwen3NextAttention: Attention, Qwen3NextGatedDeltaNet: LinearAttention, Qwen3NextMLP: Mlp, Qwen3NextSparseMoeBlock: Mlp}
