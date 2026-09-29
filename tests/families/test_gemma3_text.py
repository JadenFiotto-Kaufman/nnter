"""Gemma 3 text, end to end: a sandwich block."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.families import gemma3_text


class TestGemma3(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
    FAMILY = gemma3_text
    NATIVE = LLAMA_ROWS
    MLP_NORM = "pre_feedforward_layernorm"   # sandwich: post_attention_layernorm follows the attention

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_attention_layernorm.output.save()
        assert torch.equal(attn, post_attn)
