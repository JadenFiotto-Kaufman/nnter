"""Falcon (7B layout), end to end: parallel block, multi-query, in-place add into the MLP output."""

import torch
from suite import FamilySuite, rows, PROMPT

from nnter.families import falcon


class TestFalcon(FamilySuite):
    REPO = "Rocketknight1/tiny-random-falcon-7b"
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln2=None)
    MLP_NORM = "input_layernorm"             # parallel (7B layout)

    def test_mlp_output_is_a_copy_the_block_does_not_touch(self, model):
        """The block adds the attention into the MLP's live tensor in place; the value is a copy."""
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            live = model.layers[0].mlp.output.save()
        torch.testing.assert_close(live, mlp + attn)

    def test_in_place_mlp_edit_reaches_the_model_through_the_transform(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            model.layers[0].mlp.mlp_output[:] = 0
            kept = model.layers[0].mlp.mlp_output.save()
            edited = model.logits.save()
        assert not torch.equal(clean, edited)
        assert torch.equal(kept, torch.zeros_like(kept))  # the user's copy stays what they made it

    def test_values_bind_before_the_rotary(self, model):
        """In one trace the values must be read before the queries or keys."""
        with model.trace(PROMPT):
            v = model.layers[0].self_attn.attention_values.save()
            q = model.layers[0].self_attn.attention_queries.save()
        assert v.shape[1] == 1 and q.shape[1] == model.num_heads  # multi-query

    def test_alibi_makes_the_interior_unavailable(self, model):
        config = model.layers[0].self_attn._module.config
        config.alibi = True
        try:
            assert "alibi" in model.status()["self_attn.attention_probabilities"][0]
        finally:
            config.alibi = False


class TestFalcon40B(FamilySuite):
    """The 40B layout: ``new_decoder_architecture``, with ``ln_attn`` / ``ln_mlp`` in place of one norm."""

    REPO = "Rocketknight1/tiny-random-falcon-40b"
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln1=None, ln2=None)
    KV_HEADS_EXPANDED = True  # the new layout broadcasts its 8 kv heads to all 128 before the rotary
    ATTENTION_NORM = "ln_attn"
    MLP_NORM = "ln_mlp"
    MLP_NORM_BEFORE_ATTENTION = True   # both norms are taken from the block input before either sublayer runs

    def test_new_decoder_architecture(self, model):
        assert model.config.new_decoder_architecture
        assert hasattr(model.layers[0], "ln_attn") and hasattr(model.layers[0], "ln_mlp")
