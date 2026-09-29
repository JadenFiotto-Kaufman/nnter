"""Falcon (``FalconForCausalLM``): the 7B layout (parallel attention), with or without alibi, and the 40B layout.

``transformer.{word_embeddings, h[i].{input_layernorm, self_attention, mlp},
ln_f}`` and ``lm_head``. One norm feeds both sublayers and the block sums
``x + attn + mlp``, but it does so by adding the attention output *into the
MLP's output tensor in place*, so ``mlp_output`` reads a copy taken as the
MLP returns, and a transform carries edits to it back into the model.

The attention does its own arithmetic, and ``config.alibi`` picks one of two
branches of it with different operations, so each interior value names its
operation by that flag. Without alibi the queries and keys leave
``apply_rotary_pos_emb`` and the pattern is the first softmax, which has no
dropout after it; with alibi there is no rotary, the pattern is the dropout
after the second softmax, and the head outputs are flattened over batch and
heads. The 40B layout (``ln_attn`` / ``ln_mlp``, ``new_decoder_architecture``)
runs the same forward with its key/value heads already broadcast.
"""

import torch
from jaxtyping import Float
from torch import Tensor
from transformers.models.falcon.modeling_falcon import FalconAttention, FalconDecoderLayer, FalconMLP

from ..components import Attention, EProperty, Layer, Mlp, SourceEProperty, first_tensor, needs_eager, rewrap, seq_first

MODEL_TYPES = ("falcon",)

RENAME = {
    "transformer.word_embeddings": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "self_attention": "self_attn",
}


class Layer(Layer):
    """Falcon's parallel block; returns a tuple, which the base unwraps."""

    returns_tuple = True


def alibi(envoy) -> bool:
    return bool(envoy._module.config.alibi)


def by_alibi(without: str, with_alibi: str):
    """An op for `SourceEProperty` chosen by the checkpoint's ``alibi`` flag, a config value read at load."""

    def choose(envoy):
        return with_alibi if alibi(envoy) else without

    choose.__name__ = f"{without}|{with_alibi}"
    return choose


class Attention(Attention):
    """Falcon's attention: the residual is added in the block; the pattern and the interior on its own ops, per ``alibi``."""

    # Without alibi, queries and keys are ``apply_rotary_pos_emb``'s two
    # returns and the values the binding just before it (so read the values
    # before the queries or keys in one trace: they bind first); with alibi
    # there is no rotary and all three are the reshaped bindings, in forward
    # order. Keys and values are ``num_kv_heads`` wide (1 under multi-query).
    # The scores are the softmax's input after the mask, the pattern the
    # softmax itself (no alibi) or the dropout after it (alibi). The head
    # outputs are the ``scores @ values`` product: heads first without alibi,
    # flattened over batch and heads with it; both are served ``[batch, seq,
    # heads, head_dim]`` as a view, so in-place edits land.

    @SourceEProperty(by_alibi("apply_rotary_pos_emb_0", "query_layer_0"), attribute="output", description=Attention.attention_queries.description, unavailable=needs_eager)
    def attention_queries(self, value) -> Float[Tensor, "batch heads seq qk_head_dim"]:
        return value if alibi(self) else value[0]

    @attention_queries.postprocess
    def attention_queries(self, value):
        if alibi(self):
            return value
        _, keys = self.source.apply_rotary_pos_emb_0.output
        return value, keys

    @SourceEProperty(by_alibi("apply_rotary_pos_emb_0", "key_layer_0"), attribute="output", description=Attention.attention_keys.description, unavailable=needs_eager)
    def attention_keys(self, value) -> Float[Tensor, "batch kv_heads seq qk_head_dim"]:
        return value if alibi(self) else value[1]

    @attention_keys.postprocess
    def attention_keys(self, value):
        if alibi(self):
            return value
        queries, _ = self.source.apply_rotary_pos_emb_0.output
        return queries, value

    @SourceEProperty("value_layer_0", description=Attention.attention_values.description, unavailable=needs_eager)
    def attention_values(self, value) -> Float[Tensor, "batch kv_heads seq head_dim"]:
        return value

    @SourceEProperty(by_alibi("F_softmax_0", "F_softmax_1"), attribute="input", description=Attention.attention_scores.description, unavailable=needs_eager)
    def attention_scores(self, value) -> Float[Tensor, "batch heads query key"]:
        return value

    @SourceEProperty(by_alibi("attn_output_1", "flatten_0"), description=Attention.attention_head_outputs.description, unavailable=needs_eager)
    def attention_head_outputs(self, value) -> Float[Tensor, "batch seq heads head_dim"]:
        if alibi(self):  # [batch * heads, seq, head_dim] -> a [batch, seq, heads, head_dim] view
            heads = self._module.num_heads
            return value.view(-1, heads, *value.shape[1:]).transpose(1, 2)
        return seq_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        if alibi(self):
            return value.transpose(1, 2).reshape(-1, *value.shape[1:2], value.shape[3])
        return seq_first(value)

    @SourceEProperty(
        by_alibi("F_softmax_0", "self_attention_dropout_0"),
        description="The attention pattern the values are mixed with, [batch, heads, query, key]",
        unavailable=needs_eager,
    )
    def attention_probabilities(self, value) -> Float[Tensor, "batch heads query key"]:
        return value


class Mlp(Mlp):
    """Falcon's MLP: the block later adds the attention into this tensor in place, so read a copy.

    The copy keeps a saved read honest. So that in-place edits to it still
    reach the model, a transform hands a copy of the edited copy back to be
    swapped in once the block is done with the read; the second copy is what
    keeps the user's tensor clean when the block then adds into it.
    """

    @EProperty(key="output", description="What the MLP adds to the residual stream (a copy, since the block adds the attention into the live tensor in place)")
    def mlp_output(self, value) -> Float[Tensor, "batch seq hidden"]:
        return first_tensor(value).clone()

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value)

    @mlp_output.transform
    def mlp_output(self, value, raw):
        # Fires on the model side, after the read. The module returns a bare
        # tensor, so ``raw`` needs no rebuilding around the edited copy.
        return value.clone()


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FalconDecoderLayer: Layer, FalconAttention: Attention, FalconMLP: Mlp}
