"""Mixtral, end to end: a sparse mixture of experts."""

import torch
from suite import FamilySuite, LLAMA_ROWS

from nnter.families import mixtral


class TestMixtral(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-MixtralForCausalLM"
    FAMILY = mixtral
    NATIVE = LLAMA_ROWS


def test_a_lazy_load_reports_the_experts_implementation_it_will_run():
    """Before the weights load, ``status`` already says what ``experts_implementation=`` gives the loaded model."""
    from nnter import StandardizedTransformer

    lazy = StandardizedTransformer(TestMixtral.REPO, experts_implementation="eager")
    assert "experts_implementation=" in lazy.status(layer=0)["mlp.expert_outputs"]
    assert StandardizedTransformer(TestMixtral.REPO).status(layer=0)["mlp.expert_outputs"] is None


def routed_by_hand(moe, x, idx, w):
    """``routed_output`` from the experts' weights: the sum over slots of ``w * expert(x)``, token by token."""
    from suite import expert_by_hand

    out = torch.zeros(idx.shape[1], x.shape[-1], dtype=torch.float32, device=x.device)
    for t in range(idx.shape[1]):
        for j in range(idx.shape[2]):
            out[t] += expert_by_hand(moe, x[t], int(idx[0, t, j]), w[0, t, j]).float()
    return out[None]


def routing_from_written_logits(model, moe, scoring):
    """Write random logits; return what the router made of them, and the hand computation of each."""
    from suite import PROMPT, near

    with model.trace(PROMPT):
        logits = moe.router_logits.save()
    written = torch.randn(logits.shape, generator=torch.Generator().manual_seed(0)).to(logits)
    with model.trace(PROMPT):
        x = moe.experts.inputs[0][0].save()
    with model.trace(PROMPT):
        moe.router_logits = written
        idx = moe.expert_indices.save()
        w = moe.expert_weights.save()
        routed = moe.routed_output.save()
    expected_idx, expected_w = scoring(written.float())
    assert torch.equal(idx.sort(-1).values, expected_idx.sort(-1).values)
    near(w.float(), _align(idx, expected_idx, expected_w), w)
    near(routed.float(), routed_by_hand(moe, x, idx, w), routed, ulps=64)


def _align(idx, expected_idx, expected_w):
    """``expected_w`` reordered to ``idx``'s slot order."""
    position = (idx[..., :, None] == expected_idx[..., None, :]).float().argmax(-1)
    return expected_w.gather(-1, position)


def test_router_logits_write_by_hand():
    """Mixtral: softmax over the written logits, top-k, renormalized; the routed sum from the experts' weights."""
    from nnter import StandardizedTransformer

    model = StandardizedTransformer(TestMixtral.REPO, dispatch=True)
    moe = model.layers[1].mlp

    def scoring(logits):
        top, idx = logits.softmax(-1).topk(moe.top_k, dim=-1)
        return idx, top / top.sum(-1, keepdim=True)

    routing_from_written_logits(model, moe, scoring)
