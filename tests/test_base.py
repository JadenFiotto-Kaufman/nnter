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
