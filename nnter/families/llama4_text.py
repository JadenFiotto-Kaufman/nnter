"""Llama 4, text (``Llama4ForCausalLM``, model_type ``llama4_text``): Scout and Maverick's language model.

``model.{embed_tokens, layers[i].{input_layernorm, self_attn,
post_attention_layernorm, feed_forward}, norm}`` and ``lm_head``: Llama's
tree with the feed-forward called ``feed_forward``, renamed to ``mlp``. A
``llama4`` checkpoint (``Llama4ForConditionalGeneration``, the published
Scout and Maverick repos) nests its config's ``text_config``, of this type;
the text-generation task builds ``Llama4ForCausalLM`` from it, and a wrapper
module passed in already loaded keeps the text model at
``language_model.{model.*, lm_head}``, so ``RENAME`` carries both spellings
and whichever the tree has binds.

The block adds both residuals itself, attention through the shared eager
forward (Llama 4's own copy, which keeps the softmax in the model dtype).
What differs from Llama, inside the attention and the feed-forward:

* **iRoPE.** ``no_rope_layers[i] == 0`` marks a NoPE block: no rotary
  embedding, no qk-norm, full causal attention, and *attention temperature
  tuning* (``attn_temperature_tuning``) scales its queries by a factor that
  grows with the position. The other blocks apply the rotary embedding, then
  an L2 qk-norm (``use_qk_norm``), and attend within chunks of
  ``attention_chunk_size`` tokens (``config.layer_types``). The queries and
  keys served are what the interface receives, after all of that; a chunked
  block's pattern is zero across chunk boundaries.
* **Interleaved mixture of experts.** Blocks in ``moe_layers`` carry a
  ``Llama4TextMoe`` (a shared expert plus routed experts) returning
  ``(out, router_logits)`` with ``out`` flattened to ``[batch * seq, hidden]``;
  the others a dense ``Llama4TextMLP``. The block views either back into the
  residual's shape before adding it, so ``mlp_output`` is read at that view in
  the block's forward: ``[batch, seq, hidden]`` on both kinds, and edits to it
  reach the feed-forward's own output tensor.
  The shared expert is a ``Llama4TextMLP`` too, so it is an `Mlp`; its
  ``mlp_output`` is unavailable, since the block adds the mixture's sum.
* **Sizes.** ``intermediate_size`` in this config is the experts' width
  (and the shared expert's); the dense MLP's is ``intermediate_size_mlp``, which
  the root's ``intermediate_size`` reports. Scout has a mixture on every block,
  so there it names a width the model never uses, as on Qwen3-MoE; Maverick
  alternates dense and mixture blocks.
"""

from typing import TYPE_CHECKING

from transformers.models.llama4.modeling_llama4 import (
    Llama4TextAttention, Llama4TextDecoderLayer, Llama4TextMLP, Llama4TextMoe,
)

from ..components import Attention, Layer, Mlp, RelativeEProperty, Residual

if TYPE_CHECKING:
    from nnsight.intervention.envoy import Envoy

    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("llama4_text",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # A Llama4ForConditionalGeneration module: the same text model under ``language_model``.
    "language_model.model.embed_tokens": "embed_tokens",
    "language_model.model.layers": "layers",
    "language_model.model.norm": "norm",
    "language_model.lm_head": "lm_head",
    "feed_forward": "mlp",
}

#: The block's ``hidden_states.view(residual.shape)``: the feed-forward's output in the residual's shape.
FEED_FORWARD_VIEW = "hidden_states_view_0"


def _not_a_block_feed_forward(envoy: "Envoy") -> str | None:
    if envoy.path.rsplit(".", 1)[-1] == "feed_forward":
        return None
    return "this is the shared expert inside a mixture of experts; what the block adds is the mixture's output, at layers[i].mlp"


class Layer(Layer):
    """Llama 4's decoder block; returns a bare tensor, so the base holds.

    Its forward is source-instrumented when the envoy is built (and again when
    real weights replace meta ones), because `Mlp.mlp_output` is an operation
    in it: an operation is only served on a call whose forward was already
    instrumented when the call began, and the MLP's value is read after the
    block has started.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.source

    def _update(self, module) -> None:
        super()._update(module)
        self.source


class Attention(Attention):
    """Llama 4's attention; the shared eager forward and the residual added in the block, so the base holds.

    The queries and keys it serves are after the rotary embedding and the
    qk-norm on a RoPE block, and the queries after temperature tuning on a
    NoPE block.
    """


class Mlp(Mlp):
    """Llama 4's dense MLP or mixture of experts: the contribution is the block's view of its output in the residual's shape.

    The mixture returns ``(out, router_logits)`` with ``out`` flattened over
    batch and sequence; the block views it back before adding it. That view is
    the value on every block, dense or not, so the layout is the same.
    """

    @RelativeEProperty(
        f"../source.{FEED_FORWARD_VIEW}.output",
        description="What the MLP adds to the residual stream: its output viewed in the residual's shape, [batch, seq, hidden]",
        unavailable=_not_a_block_feed_forward,
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Llama4TextDecoderLayer: Layer, Llama4TextAttention: Attention, Llama4TextMLP: Mlp, Llama4TextMoe: Mlp}


# -- sizes: what Llama 4's config calls them ------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The dense MLP's width, ``intermediate_size_mlp``; this config's ``intermediate_size`` is the experts'."""
    return model.config.intermediate_size_mlp
