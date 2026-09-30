"""Gemma 4, text, end to end: sandwich block, per-layer embeddings, mixture of experts, KV sharing, layer_scalar."""

import glob
import json
import os
import tempfile

import pytest
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT, contributions, rows

from nnter.families import gemma4_text


def _ple_checkpoint(repo="hf-tiny-v2/tiny-random-Gemma4ForCausalLM"):
    """The tiny text checkpoint with its per-layer embedding table covering the tokenizer's vocabulary.

    As published, its ``vocab_size_per_layer_input`` is 99, so any prompt with a
    token id of 99 or more (every real one) fails in plain transformers with an
    index error in ``embed_tokens_per_layer``. The table is tiled to
    ``vocab_size`` rows (row ``i`` is row ``i % 99``) and the config says so;
    everything else is the checkpoint's own. Written once per snapshot.
    """
    from safetensors.torch import load_file, save_file

    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = os.path.join(tempfile.gettempdir(), f"nnter-gemma4-ple-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "model.safetensors")):
        return patched
    os.makedirs(patched, exist_ok=True)
    for name in os.listdir(snapshot):
        if name not in ("config.json", "model.safetensors") and not os.path.exists(os.path.join(patched, name)):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["vocab_size_per_layer_input"] = config["vocab_size"]
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    weights = load_file(os.path.join(snapshot, "model.safetensors"))
    table = weights["model.embed_tokens_per_layer.weight"]
    weights["model.embed_tokens_per_layer.weight"] = table[torch.arange(config["vocab_size"]) % table.shape[0]].contiguous()
    partial = os.path.join(patched, f"model.safetensors.{os.getpid()}")
    save_file(weights, partial, metadata={"format": "pt"})
    os.replace(partial, os.path.join(patched, "model.safetensors"))
    return patched


WRAPPER_ROWS = rows("model.language_model", "layers", "embed_tokens", "norm")


class Gemma4Suite(FamilySuite):
    FAMILY = gemma4_text
    NATIVE = LLAMA_ROWS
    MLP_NORM = "pre_feedforward_layernorm"   # sandwich: post_attention_layernorm follows the attention

    def expected_values(self, model):
        return super().expected_values(model) | ({"per_layer_output"} if hasattr(self.FAMILY.Layer, "per_layer_output") else set())

    def has_ple(self, model):
        return model.status().get("per_layer_output", "absent") is None

    def test_contribution_identity(self, model):
        """``(input + attention_output + mlp_output [+ per_layer_output]) * layer_scalar == layer_output``."""
        ple = self.has_ple(model)
        parts = {}
        with model.trace(PROMPT):
            for i, layer in enumerate(model.layers):
                x = layer.input.save()
                added = [getattr(host, value).save() for host, value in contributions(layer)]
                if ple:
                    added.append(layer.per_layer_output.save())
                parts[i] = (x, added, layer.layer_output.save())
        for i, (x, added, out) in parts.items():
            scalar = model.layers[i]._module.layer_scalar.float()
            eps = torch.finfo(out.dtype).eps
            total = (x.float() + sum(part.float() for part in added)) * scalar
            torch.testing.assert_close(total, out.float(), rtol=8 * eps, atol=8 * eps, msg=f"layer {i}")

    def test_skip_layers_with_a_given_stream(self, model):
        """On a KV-sharing checkpoint: skipping a borrowing block works; skipping a block others borrow from leaves them nothing.

        Block 0 is the last sliding block before the sharing starts, so the
        first sharing sliding block reads its keys and values out of the
        forward's ``shared_kv_states``; skipped, block 0 never stores them and
        that block fails with transformers' own ``KeyError``."""
        attention = [layer.self_attn._module for layer in model.layers]
        if not any(module.is_kv_shared_layer for module in attention):
            return super().test_skip_layers_with_a_given_stream(model)
        assert attention[0].store_full_length_kv and attention[0].layer_type == "sliding_attention"
        borrower = next(i for i, module in enumerate(attention) if module.is_kv_shared_layer)
        assert borrower + 1 < len(attention)
        with model.trace(PROMPT):
            clean = model.layers[borrower].layer_output.save()
        with model.trace(PROMPT):
            model.skip_layers(borrower, borrower, skip_with=clean * 0)
            out = model.layers[borrower].layer_output.save()
            nxt = model.layers[borrower + 1].input.save()
        assert torch.equal(out, torch.zeros_like(out)) and torch.equal(nxt, torch.zeros_like(nxt))
        with pytest.raises(KeyError, match="sliding_attention"):
            with model.trace(PROMPT):
                model.skip_layers(0, 0)
                model.logits.save()

    def test_contributions_are_the_post_norms(self, model):
        layer = model.layers[0]
        with model.trace(PROMPT):
            attn = layer.self_attn.attention_output.save()
            post_attn = layer.post_attention_layernorm.output.save()
            mlp = layer.mlp.mlp_output.save()
            post_ff = layer.post_feedforward_layernorm.output.save()
        assert torch.equal(attn, post_attn) and torch.equal(mlp, post_ff)

    def test_logits_are_softcapped_from_the_text_config(self, model):
        cap = model.config.get_text_config().final_logit_softcapping
        with model.trace(PROMPT):
            resid = model.layers[-1].layer_output.save()
            raw = model.lm_head.output.save()
            logits = model.logits.save()
        if cap:
            torch.testing.assert_close(logits, cap * torch.tanh(raw / cap))
            assert not torch.equal(raw, logits)
        else:
            assert torch.equal(raw, logits)
        torch.testing.assert_close(model.project_on_vocab(resid), logits)

    def test_sizes_are_the_top_level_ones(self, model):
        """``head_dim`` / ``num_kv_heads`` are the config's top-level values; a full block's tensors say its own."""
        text = model.config.get_text_config()
        assert model.head_dim == text._getattr_without_heterogeneous_validation("head_dim")
        assert model.num_kv_heads == text._getattr_without_heterogeneous_validation("num_key_value_heads")
        assert model.hidden_size == text.hidden_size and model.vocab_size == text.vocab_size
        full = next(i for i, kind in enumerate(text.layer_types) if kind == "full_attention")
        layer_config = text.per_layer_config[full]
        with model.trace(PROMPT):
            keys = model.layers[full].self_attn.attention_keys.save()
        with model.trace(PROMPT):
            queries = model.layers[full].self_attn.attention_queries.save()
        assert queries.shape[1] == model.num_heads and queries.shape[-1] == layer_config.head_dim
        assert keys.shape[1] == layer_config.num_key_value_heads and keys.shape[-1] == layer_config.head_dim


class TestGemma4Text(Gemma4Suite):
    """``Gemma4ForCausalLM``: per-layer embeddings, a mixture of experts on every block, the last two blocks sharing keys and values."""

    REPO = _ple_checkpoint()

    def test_the_checkpoint_has_every_quirk(self, model):
        text = model.config
        assert type(model._module).__name__ == "Gemma4ForCausalLM"
        assert text.hidden_size_per_layer_input and text.enable_moe_block and text.num_kv_shared_layers == 2
        assert text.layer_types == ["sliding_attention", "full_attention"] * 2
        assert [text.per_layer_config[i].head_dim for i in range(4)] == [16, 32, 16, 32]
        assert [layer.self_attn._module.is_kv_shared_layer for layer in model.layers] == [False, False, True, True]
        assert all(not hasattr(layer.self_attn._module, "k_proj") for layer in model.layers[2:])

    def test_mlp_output_is_the_mlp_and_the_experts(self, model):
        """On a mixture block the post-feedforward norm norms the dense MLP's and the experts' normed sum."""
        layer = model.layers[0]
        with model.trace(PROMPT):
            dense = layer.post_feedforward_layernorm_1.output.save()
            experts = layer.post_feedforward_layernorm_2.output.save()
            summed = layer.post_feedforward_layernorm.input.save()
            mlp = layer.mlp.mlp_output.save()
        torch.testing.assert_close(summed, dense + experts)
        torch.testing.assert_close(mlp, layer.post_feedforward_layernorm._module(dense + experts))

    def test_per_layer_output_is_the_third_add(self, model):
        layer = model.layers[1]
        with model.trace(PROMPT):
            x = layer.input.save()
            attn = layer.self_attn.attention_output.save()
            mlp = layer.mlp.mlp_output.save()
            gate_in = layer.per_layer_input_gate.input.save()
            ple = layer.per_layer_output.save()
            norm_out = layer.post_per_layer_input_norm.output.save()
        torch.testing.assert_close(gate_in, x + attn + mlp)  # the branch reads the stream after both sublayers
        assert torch.equal(ple, norm_out) and ple.shape == x.shape

    def test_per_layer_output_writes_land(self, model):
        layer = model.layers[0]
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            x = layer.input.save()
            attn = layer.self_attn.attention_output.save()
            mlp = layer.mlp.mlp_output.save()
            layer.per_layer_output = layer.per_layer_output * 0
            out = layer.layer_output.save()
            assigned = model.logits.save()
        with model.trace(PROMPT):
            layer.per_layer_output[:, -1] = 0
            inplace = model.logits.save()
        torch.testing.assert_close(out, x + attn + mlp)
        assert not torch.equal(clean, assigned)
        assert not torch.equal(clean[:, -1], inplace[:, -1])
        torch.testing.assert_close(clean[:, :-1], inplace[:, :-1])

    def test_scaled_identity(self, model):
        """With ``layer_scalar`` away from one, the contributions stay unscaled and the block's output is their sum times it."""
        scalars = [layer._module.layer_scalar for layer in model.layers]
        saved = [s.clone() for s in scalars]
        try:
            factors = [0.5, 0.75, 1.5, 2.0]
            for scalar, factor in zip(scalars, factors):
                scalar.fill_(factor)
            parts = {}
            with model.trace(PROMPT):
                for i, layer in enumerate(model.layers):
                    parts[i] = (
                        layer.input.save(), layer.self_attn.attention_output.save(), layer.mlp.mlp_output.save(),
                        layer.per_layer_output.save(), layer.layer_output.save(),
                    )
            for i, (x, attn, mlp, ple, out) in parts.items():
                total = x + attn + mlp + ple
                assert not torch.allclose(total, out)
                torch.testing.assert_close(total * factors[i], out)
            assert torch.equal(parts[1][0], parts[0][-1])  # the next block reads the scaled stream
        finally:
            for scalar, value in zip(scalars, saved):
                scalar.copy_(value)

    def test_kv_sharing_blocks_attend_to_borrowed_keys_and_values(self, model):
        """Blocks 2 and 3 receive blocks 0's and 1's keys and values: the same tensors, not copies."""
        read = {}
        for name in ("attention_keys", "attention_values"):
            with model.trace(PROMPT):
                for layer in model.layers:
                    read[layer.path, name] = getattr(layer.self_attn, name).save()
        paths = [layer.path for layer in model.layers]
        for name in ("attention_keys", "attention_values"):
            assert torch.equal(read[paths[2], name], read[paths[0], name])
            assert torch.equal(read[paths[3], name], read[paths[1], name])
            assert not torch.equal(read[paths[0], name][..., :4], read[paths[1], name][..., :4])

    def test_editing_borrowed_keys(self, model):
        """In place on the source block's keys reaches the borrowing block; an assignment is that block's alone."""
        source, borrower = model.layers[0].self_attn, model.layers[2].self_attn
        with model.trace(PROMPT):
            clean = borrower.attention_keys.save()
        with model.trace(PROMPT):
            source.attention_keys[:, :, -1] = 0
            inplace = borrower.attention_keys.save()
        with model.trace(PROMPT):
            source.attention_keys = source.attention_keys * 0
            assigned = borrower.attention_keys.save()
        assert torch.equal(inplace[:, :, -1], torch.zeros_like(inplace[:, :, -1]))
        torch.testing.assert_close(inplace[:, :, :-1], clean[:, :, :-1])
        torch.testing.assert_close(assigned, clean)
        with model.trace(PROMPT):
            clean_logits = model.logits.save()
        with model.trace(PROMPT):
            borrower.attention_values = borrower.attention_values * 0
            edited = model.logits.save()
        assert not torch.equal(clean_logits, edited)


class TestGemma4Wrapper(Gemma4Suite):
    """``Gemma4ForConditionalGeneration``: the text stack at ``model.language_model``, softcapped logits in ``text_config``."""

    REPO = "trl-internal-testing/tiny-Gemma4ForConditionalGeneration"
    NATIVE = WRAPPER_ROWS

    def test_the_text_stack_is_under_language_model(self, model):
        assert type(model._module).__name__ == "Gemma4ForConditionalGeneration"
        assert model.layers is model.get("model.language_model.layers")
        assert model.config.model_type == "gemma4" and model.config.get_text_config().final_logit_softcapping == 30.0
        assert model.num_layers == model.config.text_config.num_hidden_layers


class TestGemma4KEqV(Gemma4Suite):
    """A 26B-A4B-shaped wrapper: no per-layer embeddings, a mixture of experts, values from ``k_proj`` on full blocks,
    and full blocks with their own ``num_key_value_heads``."""

    REPO = "yujiepan/gemma-4-moe-tiny-random"
    NATIVE = WRAPPER_ROWS
    EXPECTED_UNAVAILABLE = {"per_layer_output": "no per-layer embeddings"}

    def test_values_come_from_k_proj_on_full_blocks(self, model):
        layer = model.layers[1]
        attn = layer.self_attn._module
        assert attn.use_alternative_attention and attn.v_proj is None
        with model.trace(PROMPT):
            normed = layer.input_layernorm.output.save()
            values = layer.self_attn.attention_values.save()
        batch, seq, _ = normed.shape
        projected = attn.k_proj(normed).view(batch, seq, -1, attn.head_dim)
        torch.testing.assert_close(values, attn.v_norm(projected).transpose(1, 2))
        assert values.shape[1] == model.config.text_config.per_layer_config[1].num_key_value_heads != model.num_kv_heads

    def test_per_layer_output_is_unavailable(self, model):
        from nnter import Unavailable

        with pytest.raises(Unavailable, match="no per-layer embeddings"):
            model.layers[0].per_layer_output
