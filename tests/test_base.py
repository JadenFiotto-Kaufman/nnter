"""The base envoys and descriptors, apart from any family."""

import pytest
import torch
from nnsight import TransformersModel  # nnsight before any transformers submodule
from transformers.models.gptj.modeling_gptj import GPTJBlock
from transformers.models.llama.modeling_llama import LlamaAttention

from nnter import Layer, StandardizedTransformer, Unavailable, unavailable
from nnter.components import Attention


@pytest.fixture(scope="module")
def tuple_model():
    """GPT-J's block returns ``(hidden_states, present)``; the base Layer on a plain TransformersModel."""
    return TransformersModel(
        "hf-internal-testing/tiny-random-GPTJForCausalLM", task="text-generation",
        dispatch=True, envoys={GPTJBlock: Layer},
    )


def test_tuple_block_unwrapped_and_rewrapped(tuple_model):
    m = tuple_model
    block = m.transformer.h[0]
    with m.trace("Hello world"):
        raw = block.output.save()
        std = block.layer_output.save()
        clean = m.lm_head.output.save()
    assert isinstance(raw, tuple) and torch.equal(std, raw[0])
    with m.trace("Hello world"):
        block.layer_output = block.layer_output * 0
        after = block.output.save()
        edited = m.lm_head.output.save()
    assert isinstance(after, tuple) and len(after) == len(raw)
    assert torch.equal(after[0], torch.zeros_like(raw[0]))
    assert not torch.equal(clean, edited)


def test_unavailable_marker_is_listed_and_raises():
    class Linear(Attention):
        attention_probabilities = unavailable("no softmax: the attention is linear")

    model = StandardizedTransformer("hf-internal-testing/tiny-random-LlamaForCausalLM", envoys={LlamaAttention: Linear})
    attn = model.layers[0].self_attn
    assert type(attn) is Linear
    assert "(attention_probabilities): Unavailable: no softmax" in repr(attn)
    assert attn.status()["attention_probabilities"] == "no softmax: the attention is linear"
    assert model.status()["self_attn.attention_probabilities"] == {i: "no softmax: the attention is linear" for i in range(model.num_layers)}
    with pytest.raises(Unavailable, match="the attention is linear"):
        attn.attention_probabilities
    with pytest.raises(Unavailable):  # TODO in nnter.components.Unavailable: hasattr should be False instead
        hasattr(attn, "attention_probabilities")


@pytest.fixture(scope="module")
def gpt2_paths():
    """A GPT-2 attention with one value per kind of path an `EProperty` key can take."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    from nnter.components import EProperty, Pattern, Residual
    from nnter.families import gpt2

    class Paths(gpt2.Attention):
        @EProperty("../ln_2.output", description="A sibling module's output")
        def sibling(self, value) -> Residual:
            return value

        @EProperty("source.attention_interface_1.source.nn_functional_softmax_0.output", description="An op two drills down")
        def softmax(self, value) -> Pattern:
            return value

        @EProperty(lambda self: "source.attention_interface_1.inputs", select="scaling", description="A function key, a keyword argument")
        def scaling(self, value) -> float:
            return value

        @EProperty("source.attention_interface_1.source.nn_functional_softmax_0.input", description="A call's first argument")
        def scores(self, value) -> Pattern:
            return value

    model = StandardizedTransformer("hf-internal-testing/tiny-random-gpt2", dispatch=True, attn_implementation="eager", envoys={GPT2Attention: Paths})
    return model, Paths


def test_eproperty_paths_resolve_and_write(gpt2_paths):
    model, Paths = gpt2_paths
    attn = model.layers[0].self_attn
    assert Paths.softmax.inside_forward() and Paths.scaling.inside_forward() and not Paths.sibling.inside_forward()
    assert model.status()["self_attn.sibling"] is None and model.status()["self_attn.softmax"] is None
    with model.trace("Hello world"):  # forward order: the call's arguments, then the ops inside it, then the sibling norm
        scaling = attn.scaling.save()
        scores = attn.scores.save()
        softmax = attn.softmax.save()
        sibling = attn.sibling.save()
        ln_2 = model.layers[0].ln_2.output.save()
        clean = model.logits.save()
    assert torch.allclose(scores.softmax(-1).to(softmax.dtype), softmax) and torch.equal(sibling, ln_2)
    assert isinstance(scaling, float) and scaling == attn._module.head_dim ** -0.5
    with model.trace("Hello world"):
        attn.scores = attn.scores * 0                       # a first-argument write
        edited = model.logits.save()
    assert not torch.equal(clean, edited)
    with model.trace("Hello world"):
        attn.scaling = 0.0                                   # a keyword-argument write repacks the call
        pattern = attn.softmax.save()
    causal = torch.ones_like(pattern).tril()                 # with no scaling every score is 0: uniform over the causal keys
    assert torch.allclose(pattern, causal / causal.sum(-1, keepdim=True))
