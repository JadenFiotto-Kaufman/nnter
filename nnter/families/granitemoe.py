"""GraniteMoE (``GraniteMoeForCausalLM``, IBM Granite 3.x MoE).

Granite's block and multipliers (``granite.py``) with a mixture of experts for
the MLP: ``block_sparse_moe`` (``GraniteMoeMoE``, routed experts only) is
``mlp``. The block adds ``h * residual_multiplier`` for each sublayer, so
``attention_output`` and ``mlp_output`` are the module's output times the
multiplier, computed copies that divide on assignment and carry an in-place edit
back through a transform, as on Granite. The mixture keeps no config, so the
multiplier is read off the block (`_block`). ``embedding_multiplier`` scales the
embedding module's output before the first block and ``logits_scaling`` divides
the head's output (the family's ``project_on_vocab``, Granite's). Every block has
the mixture, so the config's ``intermediate_size`` is the experts' width.
"""

import torch
from nnsight.intervention.envoy import Envoy
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeAttention, GraniteMoeDecoderLayer, GraniteMoeMoE

from ..components import Attention, EProperty, Layer, Mlp, Residual, first_tensor, rewrap
from .granite import Attention as GraniteAttention
from .granite import project_on_vocab  # noqa: F401  the logit lens divides by logits_scaling, as Granite's

MODEL_TYPES = ("granitemoe",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "block_sparse_moe": "mlp",
}


def _block(envoy: Envoy) -> torch.nn.Module:
    """The decoder block module ``envoy``'s module sits in: its parent by path, in the same tree."""
    parent = envoy.path.rsplit(".", 1)[0]
    return next(other._module for other in list(envoy.interleaver.envoys.values()) if other.path == parent)


def scaled_back(edited: torch.Tensor, raw, multiplier: float):
    """The value an edited scaled copy stands for: ``edited / multiplier``, rewrapped; an unedited copy hands ``raw`` back untouched."""
    served = first_tensor(raw)
    if torch.equal(edited, served * multiplier):
        return raw
    unscaled = edited / multiplier
    return (unscaled, *raw[1:]) if isinstance(raw, tuple) else unscaled


class Layer(Layer):
    """GraniteMoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(GraniteAttention):
    """GraniteMoE's attention: Granite's, the block adds its output times ``residual_multiplier``."""


class Mlp(Mlp):
    """GraniteMoE's mixture of experts; the block adds its output times ``residual_multiplier``."""

    @EProperty(key="output", description="What the MLP adds to the residual stream: its output times residual_multiplier")
    def mlp_output(self, value) -> Residual:
        return first_tensor(value) * _block(self).residual_multiplier

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value / _block(self).residual_multiplier)

    @mlp_output.transform
    def mlp_output(self, value, raw):
        return scaled_back(value, raw, _block(self).residual_multiplier)


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GraniteMoeDecoderLayer: Layer, GraniteMoeAttention: Attention, GraniteMoeMoE: Mlp}
