"""Checks every family with a Mamba-2 (SSD) mixer passes, written once: mix into a `FamilySuite` subclass.

The mixer is `nnter.StateSpace` at ``layers[i].linear_attn``. Its values are
read inside transformers' pure-torch scan kernels, so the class routes the
family's kernels to them for its whole run (``mamba_ssm`` is installed in the
test environment, and its kernels have no Python source) and restores the
default after.
"""

import pytest
import torch
from suite import LINEAR, PROMPT

from nnter import Unavailable, route_kernels


class StateSpaceChecks:
    """Set `SSD_BLOCK` to a block holding a Mamba-2 mixer."""

    #: A block whose ``linear_attn`` is the SSD mixer.
    SSD_BLOCK = 0

    @pytest.fixture(scope="class", autouse=True)
    def torch_kernels(self, request):
        """Transformers' pure-torch scan kernels for the class, before the model is loaded or traced."""
        route_kernels(request.cls.FAMILY, "torch")
        yield
        route_kernels(request.cls.FAMILY, "default")

    def ssd(self, model):
        return model.layers[self.SSD_BLOCK].linear_attn

    def _read_all(self, model):
        mix = self.ssd(model)
        got = {}
        for name in LINEAR:
            if name in ("state", "states"):
                continue  # not materialized by the SSD kernels; see the availability test
            with model.trace(PROMPT):  # one trace each: reads within a trace follow forward order
                value = getattr(mix, name)
                got[name] = value.save() if value is not None else None
        return got

    def test_ssd_values_shapes(self, model):
        mix = self.ssd(model)
        m = mix._module
        got = self._read_all(model)
        batch, seq = 1, len(model.tokenizer(PROMPT).input_ids)
        assert got["attention_queries"].shape == got["attention_keys"].shape == (batch, seq, m.n_groups, m.ssm_state_size)
        assert got["attention_values"].shape == got["attention_head_outputs"].shape == (batch, seq, m.num_heads, m.head_dim)
        assert got["betas"].shape == got["decays"].shape == (batch, seq, m.num_heads)
        assert got["decays"].dtype == torch.float32 and (got["decays"] <= 0).all() and (got["betas"] > 0).all()
        torch.testing.assert_close(got["decays"], -torch.exp(m.A_log.float()) * got["betas"].float())
        assert got["state_input"] is None  # a fresh prompt starts from nothing
        assert got["state_output"].shape == (batch, m.num_heads, m.ssm_state_size, m.head_dim)  # key side first
        assert got["attention_output"].shape == (batch, seq, model.hidden_size)

    def test_ssd_writes_are_causal(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        for name in ("attention_queries", "attention_keys", "attention_values", "attention_head_outputs"):
            with model.trace(PROMPT):
                mix = self.ssd(model)
                setattr(mix, name, getattr(mix, name) * 0)
                edited = model.logits.save()
            assert not torch.equal(clean, edited), name
        with model.trace(PROMPT):
            self.ssd(model).attention_head_outputs[:, -1] = 0
            inplace = model.logits.save()
        assert not torch.equal(clean, inplace)
        with pytest.raises(AttributeError, match="read-only"):
            with model.trace(PROMPT):
                self.ssd(model).betas = torch.zeros(1)

    def test_state_hands_off_under_generate(self, model):
        """A prompt runs the chunk scan and each decode step the token update; the state one step leaves is
        the state the next starts from, and each step's update is SSD's recurrence over the step's values."""
        mix = self.ssd(model)
        m = mix._module
        with model.trace(PROMPT):
            traced = mix.state_output.save()
        ins, outs, seqs, parts = [], [], [], []
        with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            for step in tracer.iter[:]:
                entering = mix.state_input
                ins.append(entering.save() if entering is not None else None)
                seqs.append(mix.attention_queries.shape[1])
                parts.append((mix.attention_keys.save(), mix.attention_values.save(), mix.betas.save(), mix.decays.save()))
                outs.append(mix.state_output.save())
        assert seqs == [len(model.tokenizer(PROMPT).input_ids), 1, 1]
        assert torch.equal(outs[0], traced)
        assert ins[0] is None and all(torch.equal(ins[k], outs[k - 1]) for k in (1, 2))
        repeat = m.num_heads // m.n_groups
        for k in (1, 2):
            B, x, dt, log_decay = parts[k]
            B = B[:, 0].float().repeat_interleave(repeat, dim=1)             # [batch, heads, state_dim]
            write = dt[:, 0].float()[..., None, None] * B[..., :, None] * x[:, 0].float()[..., None, :]
            expected = log_decay[:, 0].exp()[..., None, None] * ins[k].float() + write
            torch.testing.assert_close(outs[k].float(), expected, rtol=2e-2, atol=2e-3)

    def test_mixer_input_before_the_kernel_values(self, model):
        """The kernel choice is read inside the forward, so the mixer's own input can be read first, on a prompt and a decode step."""
        mix = self.ssd(model)
        seqs = []
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[:2]:
                x = mix.input
                seqs.append((x.shape[1], mix.attention_values.shape[1]))
        assert seqs == [(len(model.tokenizer(PROMPT).input_ids),) * 2, (1, 1)]

    def test_written_state_input_moves_the_step(self, model):
        mix = self.ssd(model)
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[1]:
                clean = mix.attention_head_outputs.save()
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[1]:
                mix.state_input = torch.zeros_like(mix.state_input)
                edited = mix.attention_head_outputs.save()
        assert clean.shape[1] == 1 and not torch.equal(clean, edited)

    def test_per_token_state_is_not_materialized(self, model):
        reason = model.status(layer=self.SSD_BLOCK)
        for name in ("linear_attn.state", "linear_attn.states"):
            assert "do not materialize the state per token" in reason[name]
        with pytest.raises(Unavailable, match="per token"):
            self.ssd(model).state_after(0)

    def test_optimized_kernels_are_reported(self, model):
        """With ``mamba_ssm``'s kernels bound, every kernel value says so and names the way out."""
        pytest.importorskip("mamba_ssm")
        route_kernels(self.FAMILY, "default")
        try:
            reason = model.status(layer=self.SSD_BLOCK)["linear_attn.attention_queries"]
            assert "mamba_ssm" in reason and "route_kernels" in reason
            assert model.status(layer=self.SSD_BLOCK)["linear_attn.attention_output"] is None
        finally:
            route_kernels(self.FAMILY, "torch")

    def test_ssd_values_listed_in_the_repr(self, model):
        text = repr(self.ssd(model))
        for name in LINEAR:
            assert f"({name}):" in text, name
