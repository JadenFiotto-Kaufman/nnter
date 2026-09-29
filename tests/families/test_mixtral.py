"""Mixtral, end to end: a sparse mixture of experts."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import mixtral


class TestMixtral(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-MixtralForCausalLM"
    FAMILY = mixtral
    NATIVE = LLAMA_ROWS
