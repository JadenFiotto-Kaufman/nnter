"""Qwen2 on vLLM, against the transformers engine."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnter.families.vllm import qwen2


class TestVLLMQwen2(VLLMFamilySuite):
    REPO = "Qwen/Qwen2.5-0.5B"
    FAMILY = qwen2
    NATIVE = LLAMA_ROWS
