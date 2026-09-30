"""Nemotron-H (``NemotronHForCausalLM``; Nemotron-H, Nemotron-Nano-2, Nemotron-3 Nano and Super).

Every block holds one norm and one sublayer, the ``mixer``, whose class the
block's entry in ``config.layers_block_type`` picks::

    model.embeddings
    model.layers[i]          NemotronHBlock
        .norm                RMSNorm
        .mixer               one of:
                               NemotronHMamba2Mixer   ("linear_attention", an SSD mixer)
                               NemotronHAttention     ("full_attention", no rotary embedding)
                               NemotronHMoE           ("moe": routed experts plus a shared expert)
                               NemotronHMLP           ("mlp")
    model.norm_f
    lm_head

Each block is ``hidden + mixer(norm(hidden))``. One native name, four
meanings, so the standard name is bound by the block from its mixer's class
(`Layer.child_aliases`): ``linear_attn`` on a Mamba-2 block, ``self_attn`` on
an attention block, ``mlp`` on an MoE or MLP block; ``norm`` is
``input_layernorm`` on every block. A block has exactly one of the three, so
`status` reports the other two missing on it, per block, and the
contribution identity is ``input + <the one sublayer's output> ==
layer_output``.

The attention runs the shared eager forward (no rotary embedding), so the
base `Attention` holds; the Mamba-2 mixer is transformers' Mamba-2 code, so the
base `StateSpace` holds; the MoE and the MLP return what the block adds. The
model makes its logits in float32 (``lm_head(...).float()``), so
`project_on_vocab` does too. Sizes: the attention's heads are the config's
``num_attention_heads`` / ``num_key_value_heads`` / ``head_dim``; the dense
MLP is ``intermediate_size`` wide, an MoE block's experts
``moe_intermediate_size``.
"""

import torch
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention,
    NemotronHBlock,
    NemotronHMamba2Mixer,
    NemotronHMLP,
    NemotronHMoE,
)

from ..components import Attention, Layer, Mlp, StateSpace

MODEL_TYPES = ("nemotron_h",)

RENAME = {
    "model.embeddings": "embed_tokens",
    "model.layers": "layers",
    "model.norm_f": "norm",
}

#: The standard name of a block's ``mixer``, by the mixer's class.
MIXER_NAMES = {
    NemotronHMamba2Mixer: "linear_attn",
    NemotronHAttention: "self_attn",
    NemotronHMoE: "mlp",
    NemotronHMLP: "mlp",
}


class Layer(Layer):
    """Nemotron-H's block: ``norm`` then one ``mixer``, named by what the mixer is; returns a bare tensor."""

    def child_aliases(self) -> dict[str, str]:
        return {"norm": "input_layernorm", "mixer": MIXER_NAMES[type(self._module.mixer)]}


class Attention(Attention):
    """Nemotron-H's attention: the shared eager forward without a rotary embedding, so the base holds."""


class StateSpace(StateSpace):
    """Nemotron-H's Mamba-2 mixer: transformers' Mamba-2 scan, so the base holds."""


class Mlp(Mlp):
    """Nemotron-H's mixture of experts (routed plus shared) or dense MLP; either returns what the block adds."""


def project_on_vocab(model, hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then float32."""
    return model.lm_head(model.norm(hidden)).float()


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    NemotronHBlock: Layer,
    NemotronHAttention: Attention,
    NemotronHMamba2Mixer: StateSpace,
    NemotronHMoE: Mlp,
    NemotronHMLP: Mlp,
}
