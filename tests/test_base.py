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

        @EProperty(lambda self: "source.attention_interface_1.inputs", select=lambda self: "scaling", description="A select function")
        def scaling_selected(self, value) -> float:
            return value

        @EProperty("source.attention_interface_1.source.nn_functional_softmax_0.input", description="A call's first argument")
        def scores(self, value) -> Pattern:
            return value

        @EProperty("source.attention_interface_1.inputs", select=lambda self: 1, description="A selector function: the queries")
        def queries(self, value):
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


def test_select_can_be_a_function_of_the_host(gpt2_paths):
    """``select`` given as a function picks the element at read time, for a read and for a write."""
    model, _ = gpt2_paths
    attn = model.layers[0].self_attn
    with model.trace("Hello world"):
        queries = attn.queries.save()
        clean = model.logits.save()
    with model.trace("Hello world"):
        standard = attn.attention_queries.save()
    assert torch.equal(queries, standard)
    with model.trace("Hello world"):
        attn.queries = attn.queries * 0
        edited = model.logits.save()
    assert not torch.equal(clean, edited)


def test_select_function_picks_the_element_per_access(gpt2_paths):
    """A `select` that is a function of the host picks the element at each read and write (`StateSpace`'s two kernels)."""
    model, Paths = gpt2_paths
    attn = model.layers[0].self_attn
    with model.trace("Hello world"):
        by_function = attn.scaling_selected.save()
    assert by_function == attn._module.head_dim ** -0.5
    with model.trace("Hello world"):
        attn.scaling_selected = 0.0
        pattern = attn.softmax.save()
    causal = torch.ones_like(pattern).tril()
    assert torch.allclose(pattern, causal / causal.sum(-1, keepdim=True))


def test_route_kernels_binds_each_state_space_kernel_to_its_own_torch_function():
    """A mixer with no per-token state (`StateSpace`) keeps its prompt kernel: each name gets its own pure-torch function."""
    import sys

    from nnter import StateSpace, route_kernels
    from nnter.families import mamba2

    module = sys.modules[mamba2.Mamba2Mixer.__module__]
    names = [StateSpace.CHUNK_KERNEL.rsplit("_", 1)[0], StateSpace.RECURRENT_KERNEL.rsplit("_", 1)[0]]
    before = {name: getattr(module, name) for name in names}
    try:
        route_kernels(mamba2, "torch")
        assert all(getattr(module, name) is before[name].__wrapped__ for name in names)
        route_kernels(mamba2, "default")
        assert {name: getattr(module, name) for name in names} == before
    finally:
        route_kernels(mamba2, "default")


def test_recurrent_mixer_without_a_state_op_reports_the_state_unavailable():
    """A `RecurrentMixer` whose kernels do not materialize the state per token says so, and still serves its call's values."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from nnter import LinearAttention, RecurrentMixer
    from nnter.components import EProperty
    from nnter.components.recurrent import kernel, needs_torch_kernels

    class NoState(RecurrentMixer):
        CHUNK_KERNEL = LinearAttention.CHUNK_KERNEL
        RECURRENT_KERNEL = LinearAttention.RECURRENT_KERNEL

        @EProperty(kernel("output"), select=1, unavailable=needs_torch_kernels)
        def state_output(self, value):
            return value

    assert NoState.STATE_OP is None and NoState.KERNEL.__name__ == "branched(use_precomputed_states_0)"
    model = StandardizedTransformer("yujiepan/qwen3.5-tiny-random", dispatch=True, envoys={Qwen3_5GatedDeltaNet: NoState})
    mix = model.layers[0].linear_attn
    assert type(mix) is NoState
    reason = "this mixer's kernels do not materialize the state per token"
    for name in ("state", "states"):
        assert mix.status()[name] == reason
        assert model.status()[f"linear_attn.{name}"][0] == reason
    assert mix.status()["state_output"] is None
    with pytest.raises(Unavailable, match="do not materialize the state per token"):
        mix.state_after(0)
    with pytest.raises(Unavailable, match="do not materialize the state per token"):
        mix.set_state_after(0, torch.zeros(1))
    with model.trace("Hello world"):
        final = mix.state_output.save()
    assert final.dim() == 4


def test_route_kernels_round_trips_the_bindings():
    """``"torch"`` binds both kernel names to the token-by-token loop; ``"default"`` restores what the module bound."""
    import sys

    from nnter import LinearAttention, route_delta_rule, route_kernels
    from nnter.families import qwen3_5_text

    module = sys.modules[qwen3_5_text.Qwen3_5GatedDeltaNet.__module__]
    chunk, recurrent = LinearAttention.CHUNK_KERNEL.rsplit("_", 1)[0], LinearAttention.RECURRENT_KERNEL.rsplit("_", 1)[0]
    before = {chunk: getattr(module, chunk), recurrent: getattr(module, recurrent)}
    loop = before[recurrent].__wrapped__  # functools.wraps on the dispatcher: the pure-torch token-by-token rule
    try:
        route_kernels(qwen3_5_text, "torch")
        assert getattr(module, chunk) is loop and getattr(module, recurrent) is loop
        route_kernels(qwen3_5_text, "default")
        assert {name: getattr(module, name) for name in before} == before
        route_delta_rule(module, "recurrent")  # the DeltaNet spelling, given the modeling module itself
        assert getattr(module, chunk) is loop
        route_delta_rule(module, "chunked")
        assert {name: getattr(module, name) for name in before} == before
        with pytest.raises(ValueError, match="'torch' or 'default'"):
            route_kernels(qwen3_5_text, "recurrent")
    finally:
        route_kernels(qwen3_5_text, "default")


def test_route_kernels_binds_a_single_step_decode_kernel_to_its_own():
    """On Mamba-1 (``STEP_STATE_OP`` set) the scan is the token loop: each name is bound to its own pure-torch function."""
    import sys

    from nnter import SelectiveScan, route_kernels
    from nnter.families import mamba

    module = sys.modules[mamba.MambaMixer.__module__]
    names = [op.rsplit("_", 1)[0] for op in (SelectiveScan.CHUNK_KERNEL, SelectiveScan.RECURRENT_KERNEL)]
    before = {name: getattr(module, name) for name in names}
    try:
        route_kernels(mamba, "torch")
        assert all(getattr(module, name) is before[name].__wrapped__ for name in names)
        assert SelectiveScan._loop_kernel() == SelectiveScan.CHUNK_KERNEL
        assert SelectiveScan._state_op(SelectiveScan.RECURRENT_KERNEL) == SelectiveScan.STEP_STATE_OP
        route_kernels(mamba, "default")
        assert {name: getattr(module, name) for name in names} == before
    finally:
        route_kernels(mamba, "default")


def test_decode_step_whose_first_read_is_relaxed_takes_its_own_branch():
    """A decode step that first reads ``input`` (a relaxed read) and then a kernel value binds on that
    step's own kernel: `per_call` counts the relaxed read as the step of the last pinned one."""
    import warnings

    from nnter import route_kernels
    from nnter.families import qwen3_5_text

    route_kernels(qwen3_5_text, "torch")
    try:
        model = StandardizedTransformer("yujiepan/qwen3.5-tiny-random", dispatch=True)
        mix = model.layers[0].linear_attn
        inputs, keys = [], []  # made outside the block: names bound inside do not survive it
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with model.generate("Hello world", max_new_tokens=3, do_sample=False) as tracer:
                for step in tracer.iter[:3]:
                    inputs.append(mix.input.save())
                    keys.append(mix.attention_keys.save())
        assert not [str(w.message) for w in caught if "cut short" in str(w.message).lower()]
        assert len(inputs) == 3 and len(keys) == 3
        assert all(value is not None for value in inputs + keys)
        assert keys[0].shape[1] == len(model.tokenizer("Hello world").input_ids)
        assert keys[1].shape[1] == 1 and keys[2].shape[1] == 1
    finally:
        route_kernels(qwen3_5_text, "default")
