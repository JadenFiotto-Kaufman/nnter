"""GraniteMoE, end to end: Granite's scaled residual adds around a mixture of experts."""

import test_granite
from suite import FamilySuite, LLAMA_ROWS

from nnter.families import granitemoe

REPO = "hf-internal-testing/tiny-random-GraniteMoeForCausalLM"
NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.block_sparse_moe"}


class TestGraniteMoe(FamilySuite):
    REPO = REPO
    FAMILY = granitemoe
    NATIVE = NATIVE


class TestGraniteMoeScaled(test_granite.TestGraniteScaled):
    """The same weights with every multiplier away from 1.0 (Granite's rewrite): the identity holds only on the scaled terms."""

    REPO = test_granite._scaled_checkpoint(REPO)
    FAMILY = granitemoe
    NATIVE = NATIVE
