"""The registry and the load path, across families."""

import types

import pytest
from nnsight.intervention.envoy import Envoy
from transformers import AutoModelForCausalLM

from nnter import StandardizedTransformer, UnsupportedFamily, families
from nnter.families import gpt2

GPT2 = "hf-internal-testing/tiny-random-gpt2"


def test_every_family_module_is_named_after_its_model_type():
    for name in families.known():
        family = getattr(families, name)   # lazy: imported here
        assert family.MODEL_TYPES == (name,), name
        assert families.lookup(name) is family
    assert len(families.known()) == len(families.all_families()) >= 31


def test_import_is_lazy():
    """`import nnter` pulls in no transformers modeling module; a family loads on first use."""
    import subprocess
    import sys

    code = (
        "import sys, nnsight, nnter\n"
        "before = sorted(m for m in sys.modules if m.startswith('transformers.models.') and 'modeling_' in m)\n"
        "nnter.families.lookup('gpt2')\n"
        "after = sorted(m for m in sys.modules if m.startswith('transformers.models.') and 'modeling_' in m)\n"
        "print(len(before), len(after), 'nnter.families.gpt2' in sys.modules, 'nnter.families.llama' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == "0" and int(out[1]) >= 1 and out[2] == "True" and out[3] == "False", out


def test_unknown_family_refused():
    with pytest.raises(UnsupportedFamily, match="glm4"):
        StandardizedTransformer("yujiepan/glm-4-tiny-random")


def test_register_adds_a_family_and_can_override():
    custom = types.SimpleNamespace(MODEL_TYPES=("gpt2",), RENAME=gpt2.RENAME, ENVOYS=gpt2.ENVOYS)
    try:
        families.register(custom)
        assert families.lookup("gpt2") is custom
        assert StandardizedTransformer(GPT2).family is custom
    finally:
        del families.REGISTRY["gpt2"]


def test_preloaded_module_uses_its_own_config():
    module = AutoModelForCausalLM.from_pretrained(GPT2)
    model = StandardizedTransformer(module)
    assert model.family is gpt2
    assert model.layers[0].self_attn is model.transformer.h[0].attn


def test_user_rename_merges_over_family():
    model = StandardizedTransformer(GPT2, rename={"mlp": "ffn"})
    block = model.layers[0]
    assert block.ffn is block.mlp is model.transformer.h[0].mlp


def test_user_envoys_merge_over_defaults():
    """A user's entry replaces the family's for the same key. nnsight tries type
    keys before path keys, so a family's type key is beaten by a type key."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

    class Marker(Envoy):
        pass

    model = StandardizedTransformer(GPT2, envoys={GPT2MLP: Marker})
    assert type(model.layers[0]) is model.family.Layer
    assert type(model.layers[0].mlp) is Marker


def test_remote_key_names_the_plain_transformers_model():
    """A server deploys a TransformersModel; the standardized wrapper must produce that key."""
    from nnsight.modeling.transformers import TransformersModel

    model = StandardizedTransformer(GPT2)
    assert model._remoteable_class() is TransformersModel


def test_default_load_keeps_the_checkpoints_attention():
    model = StandardizedTransformer("hf-internal-testing/tiny-random-LlamaForCausalLM")
    assert model.config._attn_implementation != "eager"
    assert "eager" in model.status()["self_attn.attention_probabilities"][0]
    assert model.status()["self_attn.attention_output"] is None
