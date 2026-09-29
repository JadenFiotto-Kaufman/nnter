---
title: Families
one_liner: One row per shipped family: its model_type and module, public and pinned checkpoints, native names, which values it relocates and how, and what status() reports unavailable, verified by a status() sweep over every tiny checkpoint.
tags: [reference, families, quirks, availability, status]
related: [docs/usage/loading.md, docs/usage/vocabulary.md, docs/usage/residual-stream.md, docs/usage/availability.md, docs/usage/attention-interior.md, docs/usage/delta-net.md, docs/extending/adding-a-family.md, docs/extending/overriding-values.md, docs/developing/testing.md, docs/developing/transformers-compat.md, docs/reference/api-quick-reference.md, docs/reference/glossary.md]
sources: [nnter/families/__init__.py, nnter/families/gpt2.py, nnter/families/llama.py, nnter/families/gpt_neox.py, nnter/families/gemma2.py, nnter/families/gemma3_text.py, nnter/families/olmo2.py, nnter/families/olmo3.py, nnter/families/gpt_oss.py, nnter/families/deepseek_v2.py, nnter/families/deepseek_v3.py, nnter/families/glm4_moe.py, nnter/families/glm4_moe_lite.py, nnter/families/dbrx.py, nnter/families/opt.py, nnter/families/gptj.py, nnter/families/bloom.py, nnter/families/mpt.py, nnter/families/falcon.py, nnter/families/qwen3_next.py, nnter/families/qwen3_5_text.py, nnter/families/qwen3_5_moe_text.py, nnter/components/attention.py, nnter/components/linear_attention.py, tests/families/suite.py]
---

# Families

## What this is for

A family is one module under `nnter/families/`, named after `config.model_type`. This page is the lookup table for the 31 shipped families (32 pinned checkpoints: Falcon's 7B and 40B layouts share one module): where each one's native modules are, which standard values the family relocates and how, and what `status()` says a checkpoint of that family does not have. The second half groups the families by quirk, so a recipe that must survive one quirk (a tuple block, a sandwich norm, a sink) can see at once which families it must survive on.

Every row but OLMo-3's was checked by loading the pinned tiny checkpoint with `dispatch=True, attn_implementation="eager"` and diffing its `status()` against the table (OLMo-3's tiny checkpoint loads only through the config rewrite its test performs, so its row rests on the suite). Under an eager load, 27 of the 31 swept checkpoints report every value available; the exceptions are OPT (no MLP module, so no `mlp.*` key) and the three hybrids (each block has `self_attn` or `linear_attn`, and the per-token state needs `route_delta_rule`). The reason strings below are quoted verbatim from that sweep.

## Canonical pattern

```python
from nnter import StandardizedTransformer

model = StandardizedTransformer("meta-llama/Llama-3.1-8B", dispatch=True, attn_implementation="eager")

print(model.family.__name__)                                         # nnter.families.llama
print(model.family.RENAME)                                           # the aliases: {'model.embed_tokens': 'embed_tokens', 'model.layers': 'layers', 'model.norm': 'norm'}
print(model.layers[0].self_attn is model.model.layers[0].self_attn)  # True: an alias, not a copy
print({name: reason for name, reason in model.status().items() if reason})   # {} : Llama has every value under eager
```

The same four lines on a checkpoint of any other family print that family's module, its `RENAME`, `True`, and the dict the tables below predict.

## The families

One row per family, keyed on `model_type`. Public ids are checkpoints of that `model_type` on the Hub; the pinned checkpoint is the tiny random one the test suite (`tests/families/test_<model_type>.py`) runs against, also cached for every example in these docs.

| `model_type` | Module | Public checkpoints | Pinned tiny checkpoint |
|---|---|---|---|
| `gpt2` | `nnter.families.gpt2` | `openai-community/gpt2`, `openai-community/gpt2-xl` | `hf-internal-testing/tiny-random-gpt2` |
| `llama` | `nnter.families.llama` | `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-2-7b-hf` | `hf-internal-testing/tiny-random-LlamaForCausalLM` |
| `gpt_neox` | `nnter.families.gpt_neox` | `EleutherAI/pythia-70m-deduped`, `EleutherAI/gpt-neox-20b` | `hf-internal-testing/tiny-random-GPTNeoXForCausalLM` |
| `mistral` | `nnter.families.mistral` | `mistralai/Mistral-7B-v0.1` | `hf-internal-testing/tiny-random-MistralForCausalLM` |
| `mixtral` | `nnter.families.mixtral` | `mistralai/Mixtral-8x7B-v0.1` | `hf-internal-testing/tiny-random-MixtralForCausalLM` |
| `qwen2` | `nnter.families.qwen2` | `Qwen/Qwen2.5-7B`, `Qwen/Qwen2-7B` | `yujiepan/qwen2-tiny-random` |
| `qwen2_moe` | `nnter.families.qwen2_moe` | `Qwen/Qwen1.5-MoE-A2.7B` | `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM` |
| `qwen3` | `nnter.families.qwen3` | `Qwen/Qwen3-8B` | `trl-internal-testing/tiny-Qwen3ForCausalLM` |
| `qwen3_moe` | `nnter.families.qwen3_moe` | `Qwen/Qwen3-30B-A3B` | `trl-internal-testing/tiny-Qwen3MoeForCausalLM` |
| `gemma` | `nnter.families.gemma` | `google/gemma-2b`, `google/gemma-7b` | `trl-internal-testing/tiny-GemmaForCausalLM` |
| `gemma2` | `nnter.families.gemma2` | `google/gemma-2-2b` | `trl-internal-testing/tiny-Gemma2ForCausalLM` |
| `gemma3_text` | `nnter.families.gemma3_text` | `google/gemma-3-1b-pt` (the text-only Gemma-3; multimodal `gemma3` checkpoints are another task) | `hf-internal-testing/tiny-random-Gemma3ForCausalLM` |
| `gpt_oss` | `nnter.families.gpt_oss` | `openai/gpt-oss-20b` | `yujiepan/gpt-oss-tiny-random` |
| `deepseek_v2` | `nnter.families.deepseek_v2` | `deepseek-ai/DeepSeek-V2-Lite` | `hf-tiny-v2/tiny-random-DeepseekV2ForCausalLM` |
| `deepseek_v3` | `nnter.families.deepseek_v3` | `deepseek-ai/DeepSeek-V3` | `hf-internal-testing/tiny-random-DeepseekV3ForCausalLM` |
| `glm4_moe` | `nnter.families.glm4_moe` | `zai-org/GLM-4.5-Air`, `zai-org/GLM-4.6` | `trl-internal-testing/tiny-Glm4MoeForCausalLM` |
| `glm4_moe_lite` | `nnter.families.glm4_moe_lite` | `zai-org/GLM-4.7-Flash` | `hf-tiny-v2/tiny-random-Glm4MoeLiteForCausalLM` |
| `dbrx` | `nnter.families.dbrx` | `databricks/dbrx-base` | `yujiepan/dbrx-tiny256-random` (load with `dtype=torch.float32`) |
| `phi` | `nnter.families.phi` | `microsoft/phi-2`, `microsoft/phi-1_5` | `hf-internal-testing/tiny-random-PhiForCausalLM` |
| `phi3` | `nnter.families.phi3` | `microsoft/Phi-3-mini-4k-instruct` | `trl-internal-testing/tiny-Phi3ForCausalLM` |
| `olmo` | `nnter.families.olmo` | `allenai/OLMo-1B-hf` | `katuni4ka/tiny-random-olmo-hf` |
| `olmo2` | `nnter.families.olmo2` | `allenai/OLMo-2-1124-7B` | `hf-tiny-v2/tiny-random-Olmo2ForCausalLM` |
| `olmo3` | `nnter.families.olmo3` | the allenai OLMo-3 checkpoints (`model_type` `olmo3`) | `yujiepan/olmo-3-tiny-random`, with its config rewritten to the per-layer-type rope form by the test (`tests/families/test_olmo3.py`); not part of the sweep |
| `smollm3` | `nnter.families.smollm3` | `HuggingFaceTB/SmolLM3-3B` | `yujiepan/smollm3-tiny-random` |
| `stablelm` | `nnter.families.stablelm` | `stabilityai/stablelm-2-1_6b` | `stabilityai/tiny-random-stablelm-2` |
| `gptj` | `nnter.families.gptj` | `EleutherAI/gpt-j-6b` | `hf-internal-testing/tiny-random-GPTJForCausalLM` |
| `bloom` | `nnter.families.bloom` | `bigscience/bloom-560m` | `hf-internal-testing/tiny-random-BloomForCausalLM` |
| `mpt` | `nnter.families.mpt` | `mosaicml/mpt-7b` | `hf-internal-testing/tiny-random-MptForCausalLM` |
| `falcon` (7B layout) | `nnter.families.falcon` | `tiiuae/falcon-7b` | `Rocketknight1/tiny-random-falcon-7b` |
| `falcon` (40B layout, `new_decoder_architecture`) | `nnter.families.falcon` | `tiiuae/falcon-40b` | `Rocketknight1/tiny-random-falcon-40b` |
| `opt` | `nnter.families.opt` | `facebook/opt-125m` | `hf-internal-testing/tiny-random-OPTForCausalLM` |
| `qwen3_next` | `nnter.families.qwen3_next` | `Qwen/Qwen3-Next-80B-A3B-Instruct` | `yujiepan/qwen3-next-moe-tiny-random` |
| `qwen3_5_text` | `nnter.families.qwen3_5_text` | `Qwen/Qwen3.5-9B` (the text model; multimodal `qwen3_5` checkpoints are another task) | `yujiepan/qwen3.5-tiny-random` |
| `qwen3_5_moe_text` | `nnter.families.qwen3_5_moe_text` | the Qwen3.5-MoE text checkpoints (`model_type` `qwen3_5_moe_text`) | `yujiepan/qwen3.5-moe-tiny-random` |

## Names, shapes and quirks

The same rows again. "Native" is what `RENAME` maps onto the standard name, in the order embeddings / blocks / final norm, then attention / MLP inside a block, then the block's norms; `lm_head` is `lm_head` everywhere (`embed_out` on old GPT-NeoX releases also binds). A block norm listed here is aliased `input_layernorm` / `post_attention_layernorm` only where the row says so; what a norm produces is `self_attn.input` / `mlp.input` on every family regardless.

"Tuple" is `Layer.returns_tuple`: whether the block returns `(hidden_states, ...)`. The attention module returns `(attn_output, attn_weights)` on every non-hybrid family; a hybrid's `linear_attn` returns a bare tensor. `attention_output`, `mlp_output` and `layer_output` unwrap these on every family.

"Overrides" names the values the family points somewhere other than the base class does. Everything not named is the base: `layer_output` at the block's output, the contributions at the module outputs, the interior on `attention_interface_1`.

| `model_type` | Native names | Tuple | Overrides | Unavailable in `status()` (exact reason) and caveats |
|---|---|---|---|---|
| `gpt2` | `transformer.wte` / `transformer.h` / `transformer.ln_f`; `attn` / `mlp`; `ln_1` = `input_layernorm`, `ln_2` = `post_attention_layernorm` | no | `off_interface`: adds the `reorder_and_upcast_attn` reason | Interior values (queries, keys, values, scores, probabilities, head outputs): `"this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path"` on a checkpoint with that flag; `attention_output` stays available. Queries, keys and values are split views of one `c_attn` tensor: assign, do not edit in place. `wpe` and `drop` keep their own names. `intermediate_size` is `n_inner`, `4 * hidden_size` when `None`. |
| `llama` | `model.embed_tokens` / `model.layers` / `model.norm`; `self_attn` / `mlp`; `input_layernorm`, `post_attention_layernorm` (already the standard names) | no | none | none |
| `gpt_neox` | `gpt_neox.embed_in` / `gpt_neox.layers` / `gpt_neox.final_layer_norm`; `attention` / `mlp`; `input_layernorm`, `post_attention_layernorm` | no | none | none. Parallel block under `use_parallel_residual` (Pythia's default): both norms take the block input, so `post_attention_layernorm` does not follow the attention. |
| `mistral` | as Llama | no | none | none |
| `mixtral` | as Llama; the MLP is `MixtralSparseMoeBlock` | no | none | none. `mlp_output` is the routed hidden states. |
| `qwen2` | as Llama | no | none | none |
| `qwen2_moe` | as Llama; `Qwen2MoeSparseMoeBlock` (with a shared expert) | no | none | none |
| `qwen3` | as Llama; `q_norm` / `k_norm` inside the attention | no | none | none. `head_dim` is the config's, not `hidden_size // num_heads`. |
| `qwen3_moe` | as Llama; `Qwen3MoeSparseMoeBlock` | no | none | none. Every block is a mixture of experts: the experts are `moe_intermediate_size` wide; `intermediate_size` is the unused dense width (the suite's `MLP_WIDTH_KEY`). |
| `gemma` | as Llama | no | none | none. `head_dim` is the config's. |
| `gemma2` | as Llama, plus `pre_feedforward_layernorm` and `post_feedforward_layernorm` | no | `attention_output` = `../post_attention_layernorm.output`; `mlp_output` = `../post_feedforward_layernorm.output` (`RelativeEProperty`) | none. Sandwich block: `post_attention_layernorm` follows the attention, `pre_feedforward_layernorm` feeds the MLP. `final_logit_softcapping` (30.0): `logits` differs from `lm_head.output`; `project_on_vocab` applies the cap. Sliding and full attention layers alternate (`layer_types`). |
| `gemma3_text` | as Gemma-2 | no | as Gemma-2 | none. Sandwich block as Gemma-2. |
| `gpt_oss` | as Llama; the MLP is `GptOssMLP`, returning `(hidden_states, router_scores)` | no | `attention_scores` = `attention_interface_1.source.attn_weights_1`, the masked scores before the sink column joins; `Attention.SINK = True` | none. Pattern rows sum to less than one; `softmax(attention_scores)` is not the pattern (the sink column must be appended first). Sliding and full attention layers alternate. |
| `deepseek_v2` | as Llama; `DeepseekV2MLP` (dense, the first `first_k_dense_replace` blocks) and `DeepseekV2Moe`, both keyed to `Mlp` | no | none | none. Multi-head latent attention: the family defines `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` and `head_dim = v_head_dim` (the config's `head_dim` key is the latent width); the interface receives `num_heads` key/value heads, so `attention_keys.shape[1] == num_heads`, not `num_kv_heads`. `q_proj` can be `None` (`q_a_proj` / `q_b_proj`). |
| `deepseek_v3` | as DeepSeek-V2 (`DeepseekV3MLP`, `DeepseekV3MoE`) | no | none | none. Latent attention as V2; `head_dim` and `qk_head_dim` are imported from `deepseek_v2`. |
| `glm4_moe` | as Llama; `Glm4MoeMLP` (dense, the first `first_k_dense_replace` blocks) and `Glm4MoeMoE`, both keyed to `Mlp`; `q_norm` / `k_norm` inside the attention under `use_qk_norm` | no | none | none. `head_dim` is the config's (128 on GLM-4.5/4.6, not `hidden_size // num_heads`); rotary covers `partial_rotary_factor` of each head. `mlp_output` on a mixture block is the routed sum plus the shared expert (`mlp.shared_experts`, itself a `Glm4MoeMLP`). |
| `glm4_moe_lite` | as Llama; `Glm4MoeLiteMLP` and `Glm4MoeLiteMoE` (per `mlp_layer_types`), both keyed to `Mlp` | no | none | none. Latent attention as DeepSeek-V2; `head_dim` and `qk_head_dim` are imported from `deepseek_v2`, since the config's `head_dim` is an alias of `qk_rope_head_dim`. Shared expert as `glm4_moe`. |
| `dbrx` | `transformer.wte` / `transformer.blocks` / `transformer.norm_f`; `norm_attn_norm.attn` / `ffn`; `norm_attn_norm.norm_1` = `input_layernorm`, `norm_attn_norm.norm_2` = `post_attention_layernorm` | no | none | none. The residual is added in `norm_attn_norm`, outside the attention module, so the base holds. The FFN width is `ffn_config.ffn_hidden_size`, not `intermediate_size`. The pinned tiny checkpoint is fp16 with degenerate weights and needs `dtype=torch.float32`. |
| `phi` | `model.embed_tokens` / `model.layers` / `model.final_layernorm`; `self_attn` / `mlp`; `input_layernorm` only | no | none | none. Parallel block: one norm feeds both sublayers; no `post_attention_layernorm` exists. |
| `phi3` | as Llama; queries, keys and values are indexed out of one fused `qkv_proj` | no | none | none. |
| `olmo` | as Llama | no | none | none |
| `olmo2` | as Llama minus `input_layernorm`: `post_attention_layernorm` and `post_feedforward_layernorm` only | no | as Gemma-2 | none. Post-norms only: `self_attn.input` is the block input; there is no `input_layernorm` to alias. |
| `olmo3` | as OLMo-2 | no | as Gemma-2 | none (per the suite; not in the sweep). Post-norms only; sliding and full attention layers mix. |
| `smollm3` | as Llama | no | none | none. Some layers use sliding-window attention and some no positional embedding. |
| `stablelm` | as Llama; `post_attention_layernorm` only without `use_parallel_residual`; `q_layernorm` / `k_layernorm` inside the attention | no | none | none. Parallel block under `use_parallel_residual` (StableLM-2): one `input_layernorm` feeds both sublayers. |
| `gptj` | `transformer.wte` / `transformer.h` / `transformer.ln_f`; `attn` / `mlp`; `ln_1` = `input_layernorm` | yes | Interior on the module's own `_attn` call: queries / keys / values = `self__attn_0` arguments 0, 1, 2; `attention_scores` = `self__attn_0.source.nn_functional_softmax_0` input; `attention_probabilities` = `self__attn_0.source.self_attn_dropout_0`; `attention_head_outputs` = `self__attn_0` return 0, served through `seq_first` | none. Parallel block: `ln_1` feeds both sublayers. Interior needs eager (`needs_eager`). Keys and values are `num_heads` wide. `intermediate_size` is `n_inner`, `4 * hidden_size` when `None`. |
| `bloom` | `transformer.word_embeddings` / `transformer.h` / `transformer.ln_f`; `self_attention` / `mlp`; `input_layernorm`, `post_attention_layernorm` | yes | `attention_output` = `dropout_add_0` input; `mlp_output` = `dropout_add_0` input (both sublayers add the residual inside); queries / keys / values = `self__reshape_0` returns 0, 1, 2; `attention_scores` = `F_softmax_0` input; `attention_probabilities` = `self_attention_dropout_0`; `attention_head_outputs` = `torch_bmm_0`, reshaped from `[batch * heads, seq, head_dim]` | none. The module outputs are residual-stream states; the contributions are the pre-add tensors. The pattern has no `attn_implementation` predicate (own arithmetic; the checkpoint loads `eager`). `word_embeddings_layernorm` keeps its own name. `intermediate_size` is `4 * hidden_size`; the config has no key for it. |
| `mpt` | `transformer.wte` / `transformer.blocks` / `transformer.norm_f`; `attn` / `ffn`; `norm_1` = `input_layernorm`, `norm_2` = `post_attention_layernorm` | yes | `mlp_output` = `F_dropout_0` output (the MLP adds the residual inside); queries / keys / values = the `query_states_0` / `key_states_0` / `value_states_0` bindings; `attention_scores` = `nn_functional_softmax_0` input; `attention_probabilities` = `nn_functional_dropout_0`; `attention_head_outputs` = `torch_matmul_1`, through `seq_first` | none. Queries, keys and values come out of one `chunk()`: assign, do not edit in place. The pattern has no `attn_implementation` predicate. `intermediate_size` is `expansion_ratio * hidden_size`. |
| `falcon` (7B layout) | `transformer.word_embeddings` / `transformer.h` / `transformer.ln_f`; `self_attention` / `mlp`; `input_layernorm` | yes | `mlp_output`: a copy of the module output, with a transform carrying edits back (the block adds the attention into the MLP's live tensor in place); each interior value names its op by `config.alibi` through `by_alibi(without, with_alibi)`: queries / keys = `apply_rotary_pos_emb_0` returns 0 and 1, or the `query_layer_0` / `key_layer_0` bindings with alibi; `attention_values` = the `value_layer_0` binding on both; `attention_scores` = `F_softmax_0` input, or `F_softmax_1` input; `attention_probabilities` = `F_softmax_0` output (no dropout follows), or `self_attention_dropout_0` output; `attention_head_outputs` = `attn_output_1` through `seq_first`, or `flatten_0` output (`[batch * heads, seq, head_dim]`) served as a `[batch, seq, heads, head_dim]` view, a write reshaping back | nothing under eager, with or without alibi: all six interior values are guarded by `needs_eager` only. Parallel block: `input_layernorm` feeds both sublayers. The family defines `num_kv_heads` (1 under `multi_query`, `config.num_kv_heads` on the 40B layout, else `num_heads`) and `intermediate_size` (`ffn_hidden_size`). Read order in one trace: without alibi `attention_values` before `attention_queries` / `attention_keys` (the values bind before the rotary); with alibi queries, then keys, then values (`query_layer_0`, `key_layer_0`, `value_layer_0` bind in that order). Interior needs eager. |
| `falcon` (40B layout) | as the 7B layout, with `ln_attn` and `ln_mlp` in place of `input_layernorm` (neither is aliased) | yes | as the 7B layout | as the 7B layout. Both norms take the block input before either sublayer runs. Keys and values are broadcast to `num_heads` (128) before the rotary, so `attention_keys.shape[1] == num_heads` while `num_kv_heads` is the config's 8. |
| `opt` | `model.decoder.embed_tokens` / `model.decoder.layers` / `model.decoder.final_layer_norm`; `self_attn` / no MLP module (`fc1`, `activation_fn`, `fc2` on the block); `self_attn_layer_norm` = `input_layernorm`; the block's own `final_layer_norm` (pre-MLP) keeps its native name | no | none (`Mlp` exists for the class convention; nothing keys it) | nothing unavailable; `status()` lists no `mlp.*` key, since no block has the module. `layers[i].fc2.output` is what the block adds; `layers[i].fc1.input` is the pre-MLP norm's output. `intermediate_size` is `ffn_dim`. |
| `qwen3_next` | as Llama; `linear_attn` (`Qwen3NextGatedDeltaNet`) on `linear_attention` blocks, `self_attn` on `full_attention` blocks; `Qwen3NextSparseMoeBlock` and `Qwen3NextMLP` both keyed to `Mlp` | no | none (`LinearAttention` is the base) | Every `self_attn.*` value: `"no self_attn module on this block"` on the DeltaNet blocks; every `linear_attn.*` value: `"no linear_attn module on this block"` on the attention block; `linear_attn.state` and `linear_attn.states` on the DeltaNet blocks: `"the state after each token is materialized only by the token-by-token kernel; the chunked kernel a prompt runs through keeps one state per 64 tokens. Call nnter.route_delta_rule(model.family, 'recurrent') before tracing this layer (slower, like attn_implementation='eager')"`. Gated query: `q_proj` is `2 * num_heads * head_dim` wide. |
| `qwen3_5_text` | as Qwen3-Next with a dense `Qwen3_5MLP` | no | none | as Qwen3-Next |
| `qwen3_5_moe_text` | as Qwen3-Next (`Qwen3_5MoeSparseMoeBlock`, `Qwen3_5MoeMLP`) | no | none | as Qwen3-Next |

Two reasons apply to every family and are not repeated per row. Loaded without `attn_implementation="eager"`, every interface family's interior reports `"read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'"` (GPT-J and Falcon say the same through `needs_eager`; BLOOM and MPT do not, their pattern being their own dropout). A hybrid whose process has `flash-linear-attention` or `causal-conv1d` installed reports every `linear_attn` value but `attention_output` as `"read inside transformers' pure-torch gated delta rule, but this process dispatches to an optimized kernel (flash-linear-attention / causal-conv1d) with no Python source; uninstall it to read these"`.

## Quirks by theme

### Tuple blocks

`gptj`, `bloom`, `mpt`, `falcon` set `Layer.returns_tuple = True`: the block returns `(hidden_states, ...)`. `layer_output` is the first element on every family; assigning it keeps the other elements; `skip_layers` hands back `(hidden, None)` on these four and a bare tensor elsewhere. Separately, every softmax attention module returns `(attn_output, attn_weights)` and GPT-OSS's MLP returns `(hidden_states, router_scores)`; `attention_output` and `mlp_output` take the first element, so a recipe never sees the tuples.

### Sandwich norms

`gemma2`, `gemma3_text`, `olmo2`, `olmo3`: the residual stream receives a post-sublayer norm's output, not the module's, so `attention_output` is a `RelativeEProperty` at `../post_attention_layernorm.output` and `mlp_output` at `../post_feedforward_layernorm.output`. On these families `post_attention_layernorm` *follows* the attention. Gemma-2/3 also norm before each sublayer (`input_layernorm`, `pre_feedforward_layernorm`); OLMo-2/3 have only the post-norms, so `self_attn.input` is the block input and no `input_layernorm` exists.

### Residual added inside the module

`bloom` (both sublayers, via `dropout_add`) and `mpt` (the MLP only) take the residual as an argument and return a residual-stream state. The contributions are `SourceEProperty` values at the tensor before the add: BLOOM's `dropout_add_0` input for both, MPT's `F_dropout_0` output for the MLP. On `dbrx` the residual is added by `norm_attn_norm`, outside the attention module, so the module's own output is its contribution and the base holds.

### Own attention arithmetic

`gptj`, `bloom`, `mpt`, `falcon` do not call `attention_interface_1`; each maps the same six interior names onto its own operations (the table above lists them). Three present `attention_head_outputs` as a transposed view (`seq_first`) because their arithmetic keeps heads first; BLOOM reshapes from `[batch * heads, seq, head_dim]`. GPT-J's and Falcon's values need an eager load (`needs_eager`); BLOOM's and MPT's carry no `attn_implementation` predicate. Falcon's pattern is the softmax output itself (no dropout follows). A family that has not mapped an interior value would mark it `unavailable(NOT_ON_INTERFACE)`; none of the shipped ones does.

### Attention sink

`gpt_oss`: each head carries a learned logit that joins the softmax as one extra key column and is dropped afterwards. `attention_probabilities` (the dropout output) is already without that column, so its rows sum to less than one (`Attention.SINK = True`); `attention_scores` is read one step earlier, at the `attn_weights_1` binding inside the interface, the masked scores just before the sink column is concatenated. `softmax(attention_scores)` therefore does not reproduce the pattern without appending the sink column and subtracting the row max in the scores' dtype.

### Latent attention

`deepseek_v2`, `deepseek_v3`, `glm4_moe_lite`: queries and keys are `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` wide, values `v_head_dim` (what the family's `head_dim` returns), and the interface receives `num_heads` key/value heads whatever `num_key_value_heads` says (`KV_HEADS_EXPANDED` in the suite). The root publishes `qk_head_dim` for this; `attention_queries` is laid out `batch heads seq qk_head_dim` on every family for the same reason.

### Mixtures of experts

`mixtral`, `qwen2_moe`, `qwen3_moe`, `gpt_oss`, `dbrx`, `deepseek_v2`, `deepseek_v3` and `glm4_moe` (after the first `first_k_dense_replace` dense blocks), `glm4_moe_lite` (dense where `mlp_layer_types` says `dense`), `qwen3_next`, `qwen3_5_moe_text`. `mlp_output` is the routed hidden states on all of them. On `glm4_moe` and `glm4_moe_lite` that includes the shared expert's output, which the mixture adds before it returns. `intermediate_size` is the dense MLP's width; the experts are `config.moe_intermediate_size` wide (`ffn_config.ffn_hidden_size` on DBRX). On an all-MoE family (Qwen3-MoE) `intermediate_size` names a width the model never uses; the suite reads `MLP_WIDTH_KEY = "moe_intermediate_size"` there.

### Gated query

`qwen3_next`, `qwen3_5_text`, `qwen3_5_moe_text`: `q_proj` produces the query and a gate side by side, so its `out_features` is `2 * num_heads * head_dim` (`QUERY_GATED`). `attention_queries` on the attention block is still `[batch, heads, seq, head_dim]`: the gate is split off before the interface.

### Parallel blocks

One norm's output feeds both sublayers and the block sums `x + attn + mlp`: `gpt_neox` (`use_parallel_residual`; two norms, both of the block input), `phi` (one `input_layernorm`), `gptj` (`ln_1`), `stablelm` (`use_parallel_residual`; `input_layernorm`), `falcon` 7B (`input_layernorm`) and 40B (`ln_attn`, `ln_mlp`, both of the block input, before either sublayer runs). `mlp.input` is that norm's output, not a mid-block stream, and the contribution identity `input + attention_output + mlp_output == layer_output` holds as on a sequential block.

### Hybrids

`qwen3_next`, `qwen3_5_text`, `qwen3_5_moe_text`: `config.layer_types` is `linear_attention` on three blocks in four and `full_attention` on the fourth. A block has `linear_attn` (a `LinearAttention`) or `self_attn` (an `Attention`), never both, so `status()` reads as per-block dicts: `"no self_attn module on this block"` on the DeltaNet blocks, `"no linear_attn module on this block"` on the attention block. The DeltaNet values are read at the delta-rule kernel call, chunked on a prompt and recurrent on a decode step, and need transformers' pure-torch kernels (`needs_torch_kernels`). The per-token `state` and `states` need `route_delta_rule(model.family, "recurrent")` before tracing the layer (`needs_recurrent_routing`); `status()` reports that instruction until then. Decide which blocks have `self_attn` outside the trace.

### No MLP module

`opt`: `fc1`, `activation_fn` and `fc2` sit on the block, so `layers[i].mlp` does not exist and `status()` lists no `mlp.*` key (a module no block has is not listed; read `status().get("mlp.mlp_output")`). `layers[i].fc2.output` is what the block adds (the suite checks the contribution identity with it). The block's own `final_layer_norm` is the pre-MLP norm and keeps its native name, since a single-component alias would also bind on the decoder's final norm.

### dtype

`dbrx`'s pinned tiny checkpoint is fp16 with weights of std 0.02: its attention scores round to a uniform pattern, which makes the queries inert and every layer's pattern identical, so the suite and the sweep load it with `dtype=torch.float32`. Real DBRX checkpoints need nothing special.

### In-place edits and read order

- `gpt2` (split views of `c_attn`) and `mpt` (one `chunk()`): torch refuses in-place edits on `attention_queries`, `attention_keys`, `attention_values`; assign instead (`REFUSES_IN_PLACE_QKV`). `attention_scores` and `attention_head_outputs` accept in-place edits on every family.
- `falcon`: `mlp_output` is a copy, since the block adds the attention into the MLP's live output in place; assignment and in-place edits reach the model through an `eproperty` transform, and the copy you saved stays what you made it.
- `falcon`: in one trace, `attention_values` before `attention_queries` or `attention_keys` (the values bind before the rotary embedding that produces the queries and keys).
- Every family: a block's interior values before that block's `attention_output`; `model.input_ids` / `attention_mask` / `input_size` before any block.

### Softcapping and sizes from the config

`gemma2` softcaps its logits (`final_logit_softcapping`): `model.logits` is the capped output, `model.lm_head.output` the raw projection, and `project_on_vocab` applies the cap so a logit lens at the last block equals `logits`. `qwen3`, `qwen3_moe` and `gemma` set `head_dim` in the config to something other than `hidden_size // num_heads`; the root's plain rule reads the config's.

Each root size is a `StandardizedProperty`: the plain rule over the config, unless the family module defines a function of the same name. The families that do: `falcon` (`num_kv_heads`: `config.num_kv_heads` on the 40B layout, 1 under `multi_query`, else `num_heads`; `intermediate_size`: `ffn_hidden_size`), `deepseek_v2`, `deepseek_v3` and `glm4_moe_lite` (`head_dim`: `v_head_dim`; `qk_head_dim`: `qk_nope_head_dim + qk_rope_head_dim`), `gpt2` and `gptj` (`intermediate_size`: `n_inner`, `4 * hidden_size` when `None`), `opt` (`ffn_dim`), `mpt` (`expansion_ratio * hidden_size`), `bloom` (`4 * hidden_size`). Every other family, and every other size on these, is the plain rule ([../usage/root-values.md](../usage/root-values.md#sizes)).

## Reproducing the sweep

The table's `status()` column comes from one script that loads every pinned checkpoint above (olmo3 excepted) with `dispatch=True, attn_implementation="eager"` (plus `dtype=torch.float32` for DBRX) and prints `model.status()`. The one-family version:

```python
import torch
from nnter import StandardizedTransformer

model = StandardizedTransformer("yujiepan/dbrx-tiny256-random", dispatch=True, attn_implementation="eager", dtype=torch.float32)
print({name: reason for name, reason in model.status().items() if reason})   # {}
```

`HF_HUB_OFFLINE=1 pytest tests/families` runs the suite behind every row: each family's test file pins its checkpoint, native paths and quirks as class attributes (`NATIVE`, `EXPECTED_UNAVAILABLE`, `REFUSES_IN_PLACE_QKV`, `ATTENTION_SINK`, `KV_HEADS_EXPANDED`, `MLP_WIDTH_KEY`, `LOAD_KWARGS`, `QUERY_GATED`, `ATTENTION_NORM`, `MLP_NORM`, `MLP_NORM_BEFORE_ATTENTION`), and `tests/families/suite.py` is the executable form of every claim above.

## Gotchas

- `status()` on a hybrid returns per-block dicts for every `self_attn.*` and `linear_attn.*` key; `status(layer=i)` is the flat view for one block.
- The interior values report unavailable under the checkpoint's default `sdpa` on every interface family; `StandardizedTransformer` does not force eager. BLOOM and MPT load eager and their pattern has no such predicate.
- Falcon's interior values name their operations by `config.alibi` (`by_alibi`), so an alibi checkpoint reports nothing unavailable under eager; the read order inside one trace differs by branch (see the row). The suite runs the 7B checkpoint on both branches (`TestFalconAlibi` on a copy with `alibi: true` in its config).
- `num_kv_heads` is the config's number; `attention_keys.shape[1]` is `num_heads` on DeepSeek and Falcon's 40B layout, where the keys are expanded before the interface.
- A family's `input_layernorm` / `post_attention_layernorm` aliases are names, not meanings: on a sandwich block the post-attention norm follows the attention, on a parallel block one norm feeds both sublayers, on OLMo-2/3 and Falcon-40B there is no `input_layernorm`. Use `self_attn.input` and `mlp.input` for what enters a sublayer.
- `olmo3`'s tiny checkpoint predates the per-layer-type rope config and loads only with the rewrite `tests/families/test_olmo3.py` performs; examples in these docs use the other families.

## Related

- [api-quick-reference.md](api-quick-reference.md): every value's layout and descriptor.
- [glossary.md](glossary.md): sandwich block, parallel block, tuple block, interface, sink, MLA, hybrid.
- [../usage/vocabulary.md](../usage/vocabulary.md), [../usage/residual-stream.md](../usage/residual-stream.md), [../usage/availability.md](../usage/availability.md), [../usage/attention-interior.md](../usage/attention-interior.md), [../usage/delta-net.md](../usage/delta-net.md).
- [../extending/adding-a-family.md](../extending/adding-a-family.md) and [../extending/overriding-values.md](../extending/overriding-values.md) for writing a row of your own; [../developing/testing.md](../developing/testing.md) for the suite; [../developing/transformers-compat.md](../developing/transformers-compat.md) for the operation names releases rename.
- nnsight `docs/usage/rename-modules.md` and `docs/usage/source.md` for the two mechanisms every row is built on.
