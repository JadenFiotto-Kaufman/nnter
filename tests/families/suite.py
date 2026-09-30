"""Every check a family must pass, written once.

A family's test file subclasses `FamilySuite`, names its pinned checkpoint and
native paths, states its quirks, and adds what is specific to it. Each method
here is an end-to-end statement about a standardized model: aliases reach the
native modules, the standard values read and write and mean what they say,
availability is honest, sizes match the tensors.
"""

import pytest
import torch
from nnsight import TransformersModel  # nnsight before any transformers submodule
from nnsight.intervention.envoy import Envoy

from nnter import StandardizedTransformer, Unavailable
from nnter.components import (
    Attention, EProperty, Layer, LinearAttention, Mlp, RecurrentMixer, SelectiveScan, Standard, StateSpace,
)
from nnter.components.standard import in_width

PROMPT = "Hello world there"

VALUES = {
    "logits", "token_embeddings", "next_token_probs", "input_ids", "attention_mask", "input_size", "layer_output",
    "self_attn.attention_output", "self_attn.attention_probabilities", "mlp.mlp_output",
    "self_attn.attention_queries", "self_attn.attention_keys", "self_attn.attention_values",
    "self_attn.attention_scores", "self_attn.attention_head_outputs",
}
INTERIOR = ("attention_queries", "attention_keys", "attention_values", "attention_scores", "attention_head_outputs")
LINEAR = ("attention_output", "attention_queries", "attention_keys", "attention_values", "decays", "betas", "state_input", "attention_head_outputs", "state_output", "state", "states")


def mixer(layer):
    """The block's sequence mixer: ``self_attn``, else ``linear_attn`` (a hybrid's recurrent block), else None (an MLP-only block)."""
    attn = getattr(layer, "self_attn", None)
    return attn if attn is not None else getattr(layer, "linear_attn", None)


def contributions(layer):
    """The block's contributions to the residual stream, as (host, value name): every sublayer the block has, in forward order."""
    hosts = [(getattr(layer, name, None), value) for name, value in
             (("linear_attn", "attention_output"), ("self_attn", "attention_output"), ("mlp", "mlp_output"))]
    return [(host, value) for host, value in hosts if host is not None]


def recurrent_mixer(family):
    """The family's recurrent mixer envoy class (`LinearAttention`, `SelectiveScan` or `StateSpace`), or None."""
    return next((envoy for envoy in family.ENVOYS.values() if issubclass(envoy, RecurrentMixer)), None)


def rows(container, layers, embed, norm, attn="self_attn", mlp="mlp", ln1="input_layernorm", ln2="post_attention_layernorm"):
    """Standard path -> native path, for the modules a family has."""
    out = {
        "embed_tokens": f"{container}.{embed}",
        "layers": f"{container}.{layers}",
        "layers.0.self_attn": f"{container}.{layers}.0.{attn}",
        "norm": f"{container}.{norm}",
        "lm_head": "lm_head",
    }
    if mlp:
        out["layers.0.mlp"] = f"{container}.{layers}.0.{mlp}"
    if ln1:
        out["layers.0.input_layernorm"] = f"{container}.{layers}.0.{ln1}"
    if ln2:
        out["layers.0.post_attention_layernorm"] = f"{container}.{layers}.0.{ln2}"
    return out


LLAMA_ROWS = rows("model", "layers", "embed_tokens", "norm")


class FamilySuite:
    """Subclass per family: set the class attributes, add the family's own tests."""

    #: The pinned tiny checkpoint.
    REPO: str
    #: The family module the checkpoint must resolve to.
    FAMILY = None
    #: Standard path -> native path (see `rows`).
    NATIVE: dict
    #: Values this checkpoint lacks: status key -> a substring of the reason.
    EXPECTED_UNAVAILABLE: dict = {}
    #: torch refuses in-place edits on q/k/v that come out of a multi-view op (split, chunk).
    REFUSES_IN_PLACE_QKV = False
    #: An attention sink: the pattern's rows sum to less than one.
    ATTENTION_SINK = False
    #: The keys and values are read already expanded to ``num_heads`` (latent attention; Falcon's 40B layout).
    KV_HEADS_EXPANDED = False
    #: A config key naming the first block's MLP width when it is not `intermediate_size` (an all-MoE family's experts).
    MLP_WIDTH_KEY = None
    #: Extra load arguments for this checkpoint (a dtype a degenerate tiny checkpoint needs).
    LOAD_KWARGS: dict = {}
    #: Gated attention: ``q_proj`` produces the query and a gate side by side, twice the width.
    QUERY_GATED = False
    #: The block's own module whose output enters the attention (None: the block input itself, no pre-norm).
    ATTENTION_NORM = "input_layernorm"
    #: The block's own module whose output enters the MLP: the pre-MLP norm, whatever the family calls it.
    MLP_NORM = "post_attention_layernorm"
    #: The MLP's norm runs before the attention does (Falcon's 40B layout norms both inputs up front).
    MLP_NORM_BEFORE_ATTENTION = False

    @pytest.fixture(scope="class")
    def model(self, request):
        """Loaded eager so the pattern is available; the default is the checkpoint's own."""
        cls = request.cls
        return StandardizedTransformer(cls.REPO, dispatch=True, attn_implementation="eager", **cls.LOAD_KWARGS)

    @pytest.fixture(scope="class")
    def raw_model(self, request):
        """The same checkpoint with no standardization, loaded the same way."""
        cls = request.cls
        return TransformersModel(cls.REPO, task="text-generation", dispatch=True, attn_implementation="eager", **cls.LOAD_KWARGS)

    def has_mlp(self, model):
        """Every block has an MLP."""
        return model.status().get("mlp.mlp_output", "absent") is None

    def any_mlp(self, model):
        """Some block has an MLP (Nemotron-H's MLP blocks are their own blocks)."""
        return any(getattr(layer, "mlp", None) is not None for layer in model.layers)

    def attn_blocks(self, model):
        """The blocks with softmax attention: all of them, one in four on a hybrid, none on a pure state-space model."""
        return [layer for layer in model.layers if getattr(layer, "self_attn", None) is not None]

    def attn_block(self, model, last=False):
        """The first (or last) block with softmax attention; the test does not apply to a model with none."""
        blocks = self.attn_blocks(model)
        if not blocks:
            pytest.skip("no softmax attention on any block")
        return blocks[-1] if last else blocks[0]

    def expected_values(self, model):
        values = set(VALUES)
        if not self.attn_blocks(model):  # a module no block has is not listed (Mamba-2's self_attn)
            values -= {name for name in VALUES if name.startswith("self_attn.")}
        if recurrent_mixer(model.family) is not None:
            values |= {f"linear_attn.{name}" for name in LINEAR}
        if not self.any_mlp(model):  # a module no block has is not listed (OPT, Mamba-2)
            values.discard("mlp.mlp_output")
        return values

    # -- names ------------------------------------------------------------------

    def test_family_resolved(self, model):
        assert model.family is self.FAMILY

    def test_standard_names_alias_native_envoys(self, model):
        for standard, native in self.NATIVE.items():
            assert isinstance(model.get(standard), Envoy), standard
            assert model.get(standard) is model.get(native), (standard, native)

    def test_no_inner_model_alias(self, model):
        """The containers are lifted to the root; nothing is bound at ``model.model``."""
        assert "model" not in model._aliases

    def test_every_layer_is_renamed(self, model):
        assert len(model.layers) > 1
        names = [key.removeprefix("layers.0.") for key in self.NATIVE if key.startswith("layers.0.")]
        for layer in model.layers:
            for name in names:
                assert isinstance(getattr(layer, name), Envoy), (layer.path, name)

    def test_envoy_classes(self, model):
        assert all(type(layer) is self.FAMILY.Layer for layer in model.layers)
        assert issubclass(self.FAMILY.Layer, Layer)
        if self.attn_blocks(model):
            assert all(type(layer.self_attn) is self.FAMILY.Attention for layer in self.attn_blocks(model))
            assert issubclass(self.FAMILY.Attention, Attention)
        if self.any_mlp(model):
            assert issubclass(self.FAMILY.Mlp, Mlp)
            assert all(type(layer.mlp) is self.FAMILY.Mlp for layer in model.layers if getattr(layer, "mlp", None) is not None)
        recurrent = recurrent_mixer(self.FAMILY)
        for layer in model.layers:
            if getattr(layer, "linear_attn", None) is not None:
                assert type(layer.linear_attn) is recurrent
                assert issubclass(recurrent, (LinearAttention, SelectiveScan, StateSpace))

    def test_trace_through_standard_names(self, model):
        with model.trace(PROMPT):
            attn = mixer(model.layers[0]).output.save()
            resid = model.layers[-1].output.save()
            normed = model.norm.output.save()
            logits = model.lm_head.output.save()
        hidden = model.hidden_size
        assert (resid[0] if isinstance(resid, tuple) else resid).shape[-1] == hidden
        assert (attn[0] if isinstance(attn, tuple) else attn).shape[-1] == hidden
        assert normed.shape[-1] == hidden and logits.shape[-1] == model.vocab_size

    # -- availability -----------------------------------------------------------

    def test_status_lists_every_standard_value(self, model):
        status = model.status()
        expected = self.expected_values(model)
        assert set(status) == expected
        assert set(model.status(layer=0)) == expected - {"logits", "token_embeddings", "next_token_probs", "input_ids", "attention_mask", "input_size"}

    def test_status_is_what_this_family_expects(self, model):
        for name, reason in model.status().items():
            if name in self.EXPECTED_UNAVAILABLE:
                assert isinstance(reason, dict) and reason, name
                assert all(self.EXPECTED_UNAVAILABLE[name] in r for r in reason.values()), name
            else:
                assert reason is None, (name, reason)

    def test_status_matches_what_reads(self, model):
        """Every value reported available reads; every one reported unavailable raises with that reason."""
        layer = model.layers[0]
        for name, reason in model.status(layer=0).items():
            module, _, value = name.rpartition(".")
            host = getattr(layer, module, None) if module else layer
            if reason is None:
                assert isinstance(getattr(type(host), value), EProperty)
            elif host is None:
                assert "no" in reason and "module" in reason
            else:
                with pytest.raises(Unavailable, match=reason.split(";")[0][:40]):
                    getattr(host, value)

    # -- boundary values --------------------------------------------------------

    def test_layer_output_is_the_tensor(self, model):
        with model.trace(PROMPT):
            raw = model.layers[0].output.save()
            std = model.layers[0].layer_output.save()
        assert isinstance(std, torch.Tensor)
        assert torch.equal(std, raw[0] if isinstance(raw, tuple) else raw)
        assert std.shape[-1] == model.hidden_size

    def test_every_value_reads_a_tensor_on_every_layer(self, model):
        read = {}  # filled inside the block: a name bound there does not survive the trace
        expected = sum(len(contributions(layer)) + 1 for layer in model.layers)
        with model.trace(PROMPT):
            for layer in model.layers:
                for host, value in contributions(layer):
                    read[host.path, value] = getattr(host, value).save()
                read[layer.path, "layer_output"] = layer.layer_output.save()
        assert len(read) == expected and expected >= 2 * len(model.layers)
        for (path, name), value in read.items():
            assert isinstance(value, torch.Tensor) and value.shape[-1] == model.hidden_size, (path, name)

    def test_contribution_identity(self, model):
        """``input + attention_output + mlp_output == layer_output``: the definition of the contributions.

        Summed over the sublayers each block has: both mixers on a parallel
        hybrid (Falcon-H1), the one sublayer of a block that holds one
        (Nemotron-H). A model with no MLP anywhere has its own test (OPT's
        feed-forward sits on the block; Mamba-2's block is the mixer alone)."""
        if not self.any_mlp(model):
            pytest.skip("no mlp module on any block")
        parts = {}
        with model.trace(PROMPT):
            for i, layer in enumerate(model.layers):
                x = layer.input.save()
                added = [getattr(host, value).save() for host, value in contributions(layer)]
                parts[i] = (x, added, layer.layer_output.save())
        for i, (x, added, out) in parts.items():
            eps = torch.finfo(out.dtype).eps  # the block may sum in another order in its own dtype
            total = x.float() + sum(part.float() for part in added)
            torch.testing.assert_close(total, out.float(), rtol=8 * eps, atol=8 * eps, msg=f"layer {i}")

    def test_sublayer_inputs_are_the_normed_stream(self, model):
        """What enters each sublayer is `self_attn.input` / `mlp.input`; the norm that produces it is
        the family's own, and its name is not the standard thing (Gemma-2's `post_attention_layernorm`
        follows the attention; a parallel block has one norm for both)."""
        layer = model.layers[0]
        attn_norm = mlp_norm = mlp_in = None  # bound outside: a None bound inside the block would not survive it
        has_mlp = getattr(layer, "mlp", None) is not None
        shared = has_mlp and self.MLP_NORM == self.ATTENTION_NORM  # a parallel block: one norm, fired once, feeds both
        with model.trace(PROMPT):  # in forward order: each norm fires before the sublayer it feeds
            block_in = layer.input.save()
            if self.ATTENTION_NORM:
                attn_norm = layer.get(self.ATTENTION_NORM).output.save()
            if has_mlp and self.MLP_NORM and self.MLP_NORM_BEFORE_ATTENTION:
                mlp_norm = layer.get(self.MLP_NORM).output.save()
            attn_in = mixer(layer).input.save()
            if has_mlp and self.MLP_NORM and not shared and not self.MLP_NORM_BEFORE_ATTENTION:
                mlp_norm = layer.get(self.MLP_NORM).output.save()
            if has_mlp:
                mlp_in = layer.mlp.input.save()
        if shared:
            mlp_norm = attn_norm
        torch.testing.assert_close(attn_in, attn_norm if attn_norm is not None else block_in)
        if mlp_in is not None and mlp_norm is not None:
            torch.testing.assert_close(mlp_in, mlp_norm)

    def test_renamed_model_equals_raw_model(self, model, raw_model):
        with model.trace(PROMPT):
            std_first = model.layers[0].layer_output.save()
            std_logits = model.lm_head.output.save()
        with raw_model.trace(PROMPT):
            raw_first = raw_model.get(model.layers[0].path.removeprefix(raw_model.path + ".")).output.save()
            raw_logits = raw_model.get(model.lm_head.path.removeprefix(raw_model.path + ".")).output.save()  # the head by its native name
        torch.testing.assert_close(std_first, raw_first[0] if isinstance(raw_first, tuple) else raw_first)
        torch.testing.assert_close(std_logits, raw_logits)

    def test_boundary_writes_land(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        names = ["attention_output", "layer_output"] + (["mlp_output"] if getattr(model.layers[0], "mlp", None) is not None else [])
        for name in names:
            with model.trace(PROMPT):
                layer = model.layers[0]
                host = {"attention_output": mixer(layer), "mlp_output": getattr(layer, "mlp", None), "layer_output": layer}[name]
                setattr(host, name, getattr(host, name) * 0)
                edited = model.logits.save()
            assert not torch.equal(clean, edited), name
        with model.trace(PROMPT):
            model.layers[0].layer_output[:] = 0
            inplace = model.logits.save()
        assert not torch.equal(clean, inplace)

    # -- the attention pattern --------------------------------------------------

    def test_probabilities_are_a_pattern(self, model):
        block = self.attn_block(model)
        with model.trace(PROMPT):
            tokens = model.layers[0].input.save()
            probs = block.self_attn.attention_probabilities.save()
        batch, heads, q, k = probs.shape
        assert heads == model.num_heads and q == k == tokens.shape[1] > 1
        assert probs.dtype == model.lm_head.weight.dtype  # after the cast back to the model dtype
        sums = probs.sum(-1).float()
        eps = torch.finfo(probs.dtype).eps  # a bf16 row can land one ulp short of one
        if self.ATTENTION_SINK:
            assert (sums < 1).all() and (sums > 0).all()
        else:
            assert torch.allclose(sums, torch.ones_like(sums), atol=8 * eps, rtol=0)
        assert torch.equal(probs.tril(), probs)  # causal

    def test_pattern_across_layers_and_traces(self, model):
        block = self.attn_block(model)
        last = self.attn_block(model, last=True)
        with model.trace(PROMPT):
            first = block.self_attn.attention_probabilities.save()
            last_probs = last.self_attn.attention_probabilities.save()
        with model.trace(PROMPT):
            again = block.self_attn.attention_probabilities.save()
        assert first.shape == last_probs.shape and torch.equal(first, again)
        if block is not last:
            assert not torch.equal(first, last_probs)

    def test_written_pattern_moves_the_logits(self, model):
        """A read can be causally inert (weights a mixer merely returns); only a write tells."""
        block = self.attn_block(model)
        with model.trace(PROMPT):
            probs = block.self_attn.attention_probabilities.save()
            clean = model.logits.save()
        torch.manual_seed(0)
        random = torch.randn_like(probs).softmax(-1)
        with model.trace(PROMPT):
            block.self_attn.attention_probabilities = random
            written = model.logits.save()
        with model.trace(PROMPT):
            block.self_attn.attention_probabilities[:, 0] = 0
            inplace = model.logits.save()
        assert not torch.allclose(clean, written) and not torch.equal(clean, inplace)

    def test_every_source_value_resolves_on_every_layer(self, model):
        """The op names inside a forward are what releases rename; every one must resolve."""
        blocks = self.attn_blocks(model)
        status = model.status(layer=int(self.attn_block(model).path.rsplit(".", 1)[1]))
        names = [
            name for name, attr in self.FAMILY.Attention.values().items()
            if attr.inside_forward() and status[f"self_attn.{name}"] is None
        ]
        assert "attention_probabilities" in names
        read = {}
        for name in names:  # one trace per value: reads within a trace must follow forward order
            with model.trace(PROMPT):
                for layer in blocks:
                    read[layer.path, name] = getattr(layer.self_attn, name).save()
        assert len(read) == len(names) * len(blocks)
        assert all(isinstance(value, torch.Tensor) for value in read.values())

    # -- the attention interior -------------------------------------------------

    def _read_interior(self, model):
        attn = self.attn_block(model).self_attn
        got = {}
        for name in INTERIOR + ("attention_probabilities",):
            with model.trace(PROMPT):  # one trace each: families bind these at different points of the forward
                got[name] = getattr(attn, name).save()
        return got

    def pattern_from_scores(self, model, scores):
        """What the softmax makes of the scores; a sink family adds its column."""
        return scores.float().softmax(-1)

    def test_interior_shapes(self, model):
        got = self._read_interior(model)
        q, k, v = got["attention_queries"], got["attention_keys"], got["attention_values"]
        scores, probs, heads = got["attention_scores"], got["attention_probabilities"], got["attention_head_outputs"]
        batch, seq = q.shape[0], q.shape[2]
        kv_heads = model.num_heads if self.KV_HEADS_EXPANDED else model.num_kv_heads
        assert q.shape == (batch, model.num_heads, seq, model.qk_head_dim)
        assert k.shape == (batch, kv_heads, seq, model.qk_head_dim)
        assert v.shape[:3] == (batch, kv_heads, seq) and v.shape[3] in (model.head_dim, model.qk_head_dim)
        assert scores.shape == probs.shape == (batch, model.num_heads, seq, seq)
        assert heads.shape[:3] == (batch, seq, model.num_heads) and heads.shape[3] in (model.head_dim, model.qk_head_dim)
        torch.testing.assert_close(self.pattern_from_scores(model, scores).to(probs.dtype), probs)

    def test_interior_writes_are_causal(self, model):
        block = self.attn_block(model)
        with model.trace(PROMPT):
            clean = model.logits.save()
        seen = {}
        for name in INTERIOR:
            with model.trace(PROMPT):
                attn = block.self_attn
                setattr(attn, name, getattr(attn, name) * 0)
                seen[name] = (attn.attention_output.save(), model.logits.save())
        for name, (_, logits) in seen.items():
            assert not torch.equal(clean, logits), name
        # Zeroed head outputs: the attention contribution is the projection's bias alone, the same on every position.
        heads_zeroed = seen["attention_head_outputs"][0]
        torch.testing.assert_close(heads_zeroed[:, 0], heads_zeroed[:, -1])

    def test_interior_in_place_edits(self, model):
        block = self.attn_block(model)
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            block.self_attn.attention_scores[:, 0] = 0
            scores = model.logits.save()
        with model.trace(PROMPT):
            block.self_attn.attention_head_outputs[:, :, 0] = 0
            heads = model.logits.save()
        assert not torch.equal(clean, scores) and not torch.equal(clean, heads)
        if self.REFUSES_IN_PLACE_QKV:
            with pytest.raises(RuntimeError, match="view"):
                with model.trace(PROMPT):
                    block.self_attn.attention_queries[:, 0] = 0
            return
        with model.trace(PROMPT):
            block.self_attn.attention_queries[:, 0] = 0
            queries = model.logits.save()
        assert not torch.equal(clean, queries)

    # -- methods over the values -------------------------------------------------

    def test_skip_layers_hands_the_stream_straight_through(self, model):
        last = model.num_layers - 1
        with model.trace(PROMPT):
            first = model.layers[0].layer_output.save()
            model.skip_layers(1, last)
            skipped_last = model.layers[last].layer_output.save()
            logits = model.logits.save()
        assert torch.equal(skipped_last, first)
        torch.testing.assert_close(logits, model.project_on_vocab(first))

    def test_skip_layers_with_a_given_stream(self, model):
        with model.trace(PROMPT):
            clean = model.layers[0].layer_output.save()
        with model.trace(PROMPT):
            model.skip_layers(0, 0, skip_with=clean * 0)
            out = model.layers[0].layer_output.save()
            nxt = model.layers[1].input.save()
        assert torch.equal(out, torch.zeros_like(out)) and torch.equal(nxt, torch.zeros_like(nxt))

    def test_steer(self, model):
        torch.manual_seed(0)
        with model.trace(PROMPT):
            clean = model.layers[0].layer_output.save()
            clean_logits = model.logits.save()
        vector = torch.randn(model.hidden_size).to(clean)
        with model.trace(PROMPT):
            model.steer(0, vector, factor=2.0, token_positions=-1)
            steered = model.layers[0].layer_output.save()
            logits = model.logits.save()
        torch.testing.assert_close(steered[:, -1], clean[:, -1] + 2.0 * vector)
        torch.testing.assert_close(steered[:, :-1], clean[:, :-1])
        assert not torch.equal(clean_logits, logits)

    def test_project_on_vocab_is_the_logit_lens(self, model):
        with model.trace(PROMPT):
            resid = model.layers[-1].layer_output.save()
            logits = model.logits.save()
        torch.testing.assert_close(model.project_on_vocab(resid), logits)
        top = model.get_topk_closest_tokens(resid[0, -1], k=3)
        assert len(top) == 1 and len(top[0]) == 3
        assert all(isinstance(token, str) and 0 < p <= 1 for token, p in top[0].items())
        assert 0 < sum(top[0].values()) <= 1 + 1e-4

    # -- layouts -------------------------------------------------------------------

    def axis_sizes(self, model, host):
        """What each axis name in a value's annotation must be on this model."""
        seq = len(model.tokenizer(PROMPT).input_ids)
        sizes = {"batch": 1, "seq": seq, "query": seq, "key": seq, "hidden": model.hidden_size, "vocab": model.vocab_size}
        if self.attn_blocks(model):  # a state-space model has no heads to size
            sizes.update(heads=model.num_heads, kv_heads=model.num_heads if self.KV_HEADS_EXPANDED else model.num_kv_heads,
                         head_dim=model.head_dim, qk_head_dim=model.qk_head_dim)
        module = getattr(host, "_module", None)
        if isinstance(host, LinearAttention):
            sizes.update(heads=module.num_v_heads, key_dim=module.head_k_dim, value_dim=module.head_v_dim)
        if isinstance(host, SelectiveScan):
            sizes.update(groups=1, channels=module.intermediate_size, state_dim=module.ssm_state_size)
        if isinstance(host, StateSpace):
            sizes.update(heads=module.num_heads, groups=module.n_groups, state_dim=module.ssm_state_size,
                         head_dim=module.head_dim, key_dim=module.ssm_state_size, value_dim=module.head_dim)
        return sizes

    def test_values_match_their_annotations(self, model):
        """Every value's tensor has the rank, dtype and axis sizes its `Float[Tensor, "..."]` annotation says.

        One value per trace: the values fire at different points of the forward."""
        attention = bool(self.attn_blocks(model))
        block = self.attn_blocks(model)[0] if attention else model.layers[0]
        hosts = [model, block] + ([block.self_attn] if attention else [])
        mlp = next((layer.mlp for layer in model.layers if getattr(layer, "mlp", None) is not None), None)
        if mlp is not None:
            hosts.append(mlp)
        linear = next((layer.linear_attn for layer in model.layers if getattr(layer, "linear_attn", None) is not None), None)
        if linear is not None:
            hosts.append(linear)
        root_status = {k: v for k, v in model.status().items() if "." not in k}
        checked = 0
        for host in hosts:
            unavailable = root_status if host is model else host.status()
            for name, value in Standard.values.__func__(type(host)).items():
                if value.layout is None or unavailable.get(name):
                    continue
                saved = None  # bound outside: a name bound inside the block does not survive it
                with model.trace(PROMPT):
                    tensor = getattr(host, name)
                    if tensor is not None:
                        saved = tensor.save()
                if saved is None:
                    continue  # a value that is legitimately None here (state_input on a fresh prompt)
                assert isinstance(saved, value.layout), (host.path, name, tuple(saved.shape), value.dims)
                sizes = self.axis_sizes(model, host)
                for axis, size in zip(value.dims, saved.shape):
                    if axis in sizes and not (axis == "head_dim" and self.KV_HEADS_EXPANDED):
                        assert size == sizes[axis], (host.path, name, axis, size, sizes[axis])
                checked += 1
        assert checked >= (11 if attention else 4)  # root 3 + layer 1 + attention 7, plus the MLP and linear hosts where present

    # -- the input -----------------------------------------------------------------

    def test_input_accessors(self, model):
        with model.trace(PROMPT):
            ids = model.input_ids.save()
            size = torch.tensor(model.input_size).save()  # a Size bound in the block would not survive it
            mask = model.attention_mask.save()
            first = model.token_embeddings.save()
        n = len(model.tokenizer(PROMPT).input_ids)
        assert tuple(ids.shape) == tuple(size.tolist()) == (1, n) == tuple(mask.shape)
        assert mask.bool().all() and first.shape[1] == n
        assert ids[0].tolist() == model.tokenizer(PROMPT).input_ids
        text = repr(model)
        assert all(f"({name}):" in text for name in ("input_ids", "attention_mask", "input_size"))

    def test_assigning_input_ids_runs_other_ids(self, model):
        other = "A completely different prompt here"
        with model.trace(other):
            other_ids = model.input_ids.save()
            other_logits = model.logits.save()
        with model.trace(PROMPT):
            model.input_ids = other_ids.clone()
            model.attention_mask = torch.ones_like(other_ids)
            logits = model.logits.save()
        torch.testing.assert_close(logits, other_logits)
        with pytest.raises(AttributeError, match="assign input_ids"):
            with model.trace(PROMPT):
                model.input_size = (1, 2)

    # -- the root ---------------------------------------------------------------

    def test_logits_are_the_models_output(self, model):
        with model.trace(PROMPT):
            last = model.layers[-1].layer_output.save()
            logits = model.logits.save()
            result = model.output.logits.save()
        assert torch.equal(logits, result) and logits.shape[-1] == model.vocab_size
        torch.testing.assert_close(logits, model.project_on_vocab(last))  # the lens on the last block is the model's own step after the head

    def test_assigning_logits_replaces_the_result(self, model):
        with model.trace(PROMPT) as tracer:
            model.logits = model.logits * 0
            result = tracer.result.logits.save()
        assert torch.equal(result, torch.zeros_like(result))

    def test_token_embeddings_are_the_embedding_output(self, model):
        with model.trace(PROMPT):
            emb = model.token_embeddings.save()
            native = model.embed_tokens.output.save()
            clean = model.logits.save()
        assert torch.equal(emb, native) and emb.shape[-1] == model.hidden_size
        with model.trace(PROMPT):
            model.token_embeddings = model.token_embeddings * 0
            edited = model.logits.save()
        assert not torch.equal(clean, edited)

    def test_next_token_probs(self, model):
        with model.trace(PROMPT):
            probs = model.next_token_probs.save()
            logits = model.logits.save()
        assert probs.shape == (1, model.vocab_size)
        torch.testing.assert_close(probs, logits[:, -1].softmax(-1))
        assert "(next_token_probs):" in repr(model)
        with pytest.raises(AttributeError, match="assign model.logits"):
            with model.trace(PROMPT):
                model.next_token_probs = torch.zeros(1, model.vocab_size)

    def test_sizes_match_the_model(self, model):
        assert model.num_layers == len(model.layers) == model.config.get_text_config().num_hidden_layers
        blocks = self.attn_blocks(model)
        block = blocks[0] if blocks else model.layers[0]
        if blocks:
            attn = block.self_attn._module
            with model.trace(PROMPT):
                probs = block.self_attn.attention_probabilities.save()
                resid = block.layer_output.save()
            assert probs.shape[1] == model.num_heads
            assert 1 <= model.num_kv_heads <= model.num_heads
            if getattr(attn, "q_proj", None) is not None:  # latent attention carries a q_proj set to None
                assert model.num_heads * model.qk_head_dim * (2 if self.QUERY_GATED else 1) == attn.q_proj.out_features
            if getattr(attn, "o_proj", None) is not None:
                assert model.num_heads * model.head_dim == attn.o_proj.in_features
        else:
            with model.trace(PROMPT):
                resid = block.layer_output.save()
        assert resid.shape[-1] == model.hidden_size
        # The MLP's hidden width appears in some projection's shape, wherever the
        # family keeps it (OPT: on the block; fused experts: a 3-d parameter;
        # Nemotron-H: its own block; Mamba: the mixer's inner width, its block's only one).
        block = next((layer for layer in model.layers if getattr(layer, "mlp", None) is not None), block)._module
        dims = {d for p in block.parameters() for d in p.shape}
        assert isinstance(model.intermediate_size, int)  # resolves on every family, whatever the config calls it
        width = getattr(model.config.get_text_config(), self.MLP_WIDTH_KEY) if self.MLP_WIDTH_KEY else model.intermediate_size
        assert width in dims, (self.MLP_WIDTH_KEY or "intermediate_size", width, dims)

    def test_per_module_sizes_match_each_block(self, model):
        """Each block's attention and MLP report the sizes of their own tensors.

        Outside a trace against every block's projections; inside one against
        the interior of one block of each kind (module type and parameter
        shapes), where the served values are available.
        """
        kinds = {}
        for layer in self.attn_blocks(model):
            attn, module = layer.self_attn, layer.self_attn._module
            heads, kv, head_dim, qk = attn.num_heads, attn.num_kv_heads, attn.head_dim, attn.qk_head_dim
            assert all(isinstance(size, int) and size > 0 for size in (heads, kv, head_dim, qk)), layer.path
            assert heads % kv == 0, (layer.path, heads, kv)
            out = in_width(module, "o_proj", "out_proj", "dense", "c_proj", "wo")
            assert out in (None, heads * head_dim), (layer.path, out, heads, head_dim)
            for name, width in (("q_proj", heads * qk * (2 if self.QUERY_GATED else 1)), ("k_proj", kv * qk), ("v_proj", kv * head_dim)):
                projection = getattr(module, name, None)
                if isinstance(projection, torch.nn.Linear):
                    assert projection.out_features == width, (layer.path, name, projection.out_features, width)
            kinds.setdefault((type(module), tuple(tuple(p.shape) for p in module.parameters())), layer)
        for layer in kinds.values():
            attn = layer.self_attn
            status = attn.status()
            kv = attn.num_heads if self.KV_HEADS_EXPANDED else attn.num_kv_heads
            widths = (attn.head_dim, attn.qk_head_dim) if self.KV_HEADS_EXPANDED else (attn.head_dim,)  # latent attention may pad values
            expected = {
                "attention_queries": lambda t: t.shape[1] == attn.num_heads and t.shape[3] == attn.qk_head_dim,
                "attention_keys": lambda t: t.shape[1] == kv and t.shape[3] == attn.qk_head_dim,
                "attention_values": lambda t: t.shape[1] == kv and t.shape[3] in widths,
                "attention_probabilities": lambda t: t.shape[1] == attn.num_heads,
                "attention_head_outputs": lambda t: t.shape[2] == attn.num_heads and t.shape[3] in widths,
            }
            for name, check in expected.items():
                if status[name] is not None:
                    continue
                with model.trace(PROMPT):  # one trace each: families bind these at different points of the forward
                    tensor = getattr(attn, name).save()
                assert check(tensor), (layer.path, name, tuple(tensor.shape), attn.num_heads, kv, attn.head_dim, attn.qk_head_dim)
        # The MLP's width is an axis of its own weights (its routed experts', on a
        # mixture, flattened over the experts on DBRX); the root's on a dense block;
        # the family's MLP_WIDTH_KEY on the first MLP block where the suite names one.
        mlps = [layer.mlp for layer in model.layers if getattr(layer, "mlp", None) is not None]
        for mlp in mlps:
            width = mlp.intermediate_size
            experts = getattr(mlp._module, "experts", None)
            params = list((experts if experts is not None else mlp._module).parameters())
            count = getattr(experts, "num_experts", None) or 1
            dims = {d for p in params for d in p.shape} | {d // count for p in params for d in p.shape if d % count == 0}
            assert width in dims, (mlp.path, width, dims)
            if experts is None:
                assert width == model.intermediate_size, (mlp.path, width, model.intermediate_size)
        if mlps and self.MLP_WIDTH_KEY:
            assert mlps[0].intermediate_size == getattr(model.config.get_text_config(), self.MLP_WIDTH_KEY)

    def test_repr_lists_the_values(self, model):
        blocks = self.attn_blocks(model)
        block = blocks[0] if blocks else model.layers[0]
        text = repr(block)
        own = ("attention_probabilities",) if blocks else ("betas", "decays", "state_output")
        for name in ("layer_output", "attention_output", "attention_queries", "attention_head_outputs") + own:
            assert f"({name}):" in text, name
        if getattr(block, "mlp", None) is not None:
            assert "(mlp_output):" in text
