"""Cohere (``CohereForCausalLM``, Command-R).

``model.{embed_tokens, layers[i].{input_layernorm, self_attn, mlp}, norm}`` and
``lm_head``: Llama's names. The block is parallel: one LayerNorm (not RMS)
feeds both sublayers and the block returns ``x + attn + mlp``, so the
contributions are the two modules' outputs and the base holds. There is no
``post_attention_layernorm``. The attention runs the shared interface, after
an optional per-head ``q_norm`` / ``k_norm`` (``use_qk_norm``) and an
interleaved rotary.

The model multiplies the head's output by ``config.logit_scale`` to make the
logits, so the family defines `finish_logits` and ``project_on_vocab`` stays
the logit lens.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.cohere.modeling_cohere import CohereAttention, CohereDecoderLayer, CohereMLP

from ..components import Attention, Layer, Mlp

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("cohere",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Cohere's parallel block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Cohere's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Cohere's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {CohereDecoderLayer: Layer, CohereAttention: Attention, CohereMLP: Mlp}


def finish_logits(model: "StandardizedTransformer", raw: torch.Tensor) -> torch.Tensor:
    """The logits are ``lm_head``'s output times ``logit_scale``."""
    return raw * model.config.logit_scale
