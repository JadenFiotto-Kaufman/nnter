"""Gemma 4, text (``Gemma4ForCausalLM``, model_type ``gemma4_text``): E2B, E4B, 26B-A4B and 31B.

``model.{embed_tokens, layers[i].{input_layernorm, self_attn, post_attention_layernorm,
pre_feedforward_layernorm, mlp, post_feedforward_layernorm}, norm}`` and ``lm_head``:
Gemma-3's sandwich tree. A ``gemma4`` checkpoint (``Gemma4ForConditionalGeneration``,
the published repos) nests its config's ``text_config``, of this type; the
text-generation task builds the wrapper, whose text stack sits at
``model.language_model.{embed_tokens, layers, norm}`` with ``lm_head`` at the root, so
``RENAME`` carries both spellings and whichever the tree has binds.

The block, in order::

    x1 = x  + post_attention_layernorm(self_attn(input_layernorm(x)))
    x2 = x1 + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x1)) [+ experts])
    x3 = x2 + post_per_layer_input_norm(per_layer_projection(gelu(per_layer_input_gate(x2)) * per_layer_input))
    out = x3 * layer_scalar          # in place, on x3

* **The contributions** are the post-norms' outputs, as on Gemma-2/3. On a
  mixture-of-experts checkpoint (26B-A4B, ``enable_moe_block``) the block
  also runs ``router`` and ``experts``, direct children of the block, not of
  ``mlp``: ``post_feedforward_layernorm_1(mlp(...)) +
  post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(x1)))``, and
  ``post_feedforward_layernorm`` norms that sum, so ``mlp_output`` is the dense
  MLP and the experts together, what the block adds.
* **Per-layer embeddings** (E2B/E4B, ``hidden_size_per_layer_input``): a third
  add, served as ``layers[i].per_layer_output`` (the post-per-layer-input norm's
  output). ``per_layer_input`` is the block's second argument, one slice of the
  model's per-layer embedding table mixed with a projection of the token
  embeddings. On a checkpoint without them the value is unavailable.
* **``layer_scalar``**, a per-block buffer, multiplies the sum in place. The
  contributions are served unscaled, so the identity is
  ``(input + attention_output + mlp_output [+ per_layer_output]) * layer_scalar
  == layer_output``.
* **Keys and values that are not the block's own.** The last
  ``num_kv_shared_layers`` blocks have no ``k_proj``/``v_proj``: each borrows
  the keys and values of the last earlier block of the same kind (sliding or
  full) before the sharing starts. On a checkpoint with ``attention_k_eq_v``
  (26B-A4B, 31B) the full-attention blocks have no ``v_proj``: their values are
  ``v_norm(k_proj(x))``, the keys' projection before ``k_norm`` and the rotary
  embedding. ``attention_keys`` and ``attention_values`` are what the attention
  receives on every block, borrowed or not.
* **Per-layer sizes.** Sliding and full blocks differ in ``head_dim`` (and on
  26B-A4B/31B in ``num_key_value_heads``), and on E2B (``use_double_wide_mlp``)
  the KV-sharing blocks' MLP is twice ``intermediate_size`` wide; the config
  marks the attention sizes per-layer and refuses a plain ``config.head_dim``.
  The root's ``head_dim`` and ``num_kv_heads`` report the config's top-level
  values (the sliding blocks'); each block's own are on ``layers[i].self_attn``
  and ``layers[i].mlp``, read off the module.

``final_logit_softcapping`` is in the text config; the root's ``project_on_vocab``
reads it there.
"""

from typing import TYPE_CHECKING

from nnsight.intervention.envoy import Envoy
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention, Gemma4TextDecoderLayer, Gemma4TextMLP

from ..components import Attention, EProperty, Layer, Mlp, Residual

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("gemma4_text",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # A Gemma4ForConditionalGeneration: the same text model under ``model.language_model``.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
}


def _no_per_layer_input(envoy: Envoy) -> str | None:
    if envoy._module.hidden_size_per_layer_input:
        return None
    return "this checkpoint has no per-layer embeddings (hidden_size_per_layer_input is 0); the block adds attention_output and mlp_output only"


class Layer(Layer):
    """Gemma-4's decoder block; returns a bare tensor (the sum times ``layer_scalar``), so the base holds.

    Adds ``per_layer_output``, the third thing the block adds on a checkpoint
    with per-layer embeddings.
    """

    @EProperty(
        "post_per_layer_input_norm.output",
        description="What the per-layer-embedding branch adds to the residual stream: the post-per-layer-input norm's output",
        unavailable=_no_per_layer_input,
    )
    def per_layer_output(self, value) -> Residual:
        return value


class Attention(Attention):
    """Gemma-4's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output.

    The keys and values served are what the interface receives: on a
    KV-sharing block, the tensors an earlier block computed; on an
    ``attention_k_eq_v`` full block, values from ``k_proj``.
    """

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Gemma-4's MLP: what reaches the residual stream is the post-feedforward norm's output (with the experts', on a mixture block)."""

    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP (and, on a mixture block, the experts) adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Gemma4TextDecoderLayer: Layer, Gemma4TextAttention: Attention, Gemma4TextMLP: Mlp}


# -- sizes: per-layer on this config ---------------------------------------------
# A plain ``config.head_dim`` raises (a per-layer attribute); the stored top-level value does not.

def head_dim(model: "StandardizedTransformer") -> int:
    """The config's top-level ``head_dim`` (the sliding blocks'), as stored; full blocks' are on ``layers[i].self_attn``."""
    return vars(model.config.get_text_config())["head_dim"]


def num_kv_heads(model: "StandardizedTransformer") -> int:
    """The config's top-level ``num_key_value_heads`` (the sliding blocks'), as stored; full blocks' are on ``layers[i].self_attn``."""
    return vars(model.config.get_text_config())["num_key_value_heads"]
