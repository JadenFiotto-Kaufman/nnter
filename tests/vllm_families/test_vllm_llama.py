"""Llama on vLLM, against the transformers engine."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnter.families.vllm import llama


class TestVLLMLlama(VLLMFamilySuite):
    REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
    FAMILY = llama
    NATIVE = LLAMA_ROWS


class TestVLLMLlamaWithoutPrefixCaching(TestVLLMLlama):
    """The same, on an engine built with prefix caching off: an edit's prompt is computed whole."""

    ENGINE = {"enable_prefix_caching": False}
