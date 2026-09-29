# nnter: new-family proposals (scouted 2026-09-29)

Ground truth: transformers **5.17.0** in `ndif2` (`site-packages/transformers/models/`, 518 dirs, 178 `MODEL_FOR_CAUSAL_LM` entries). Each block, attention and mixer class below was read out of the `modeling_*.py` files with an AST pass and, for the leading candidates, by reading the `forward` directly. Popularity comes from the Hub API: the top 3,000 `text-generation` and top 2,000 `image-text-to-text` models by downloads, grouped by `config.model_type`. The NDIF signal is the live `https://api.ndif.us/status` (HOT/WARM/COLD).

**How the checks were run:**
- **Configs.** Every tiny checkpoint named below had its `config.json` fetched and parsed with `AutoConfig` on 5.17, and all of them pass except where the entry says otherwise.
- **Forward passes.** 50 tiny checkpoints ran one fp32 CPU forward pass through `AutoModelForCausalLM`.
- **Smoke test.** 10 families were registered as throwaway `families.register(SimpleNamespace(...))` families from the scratchpad, not from `~/wd/nnter`. Each was then loaded through `StandardizedTransformer` (eager, CPU), and I checked `status()`, pattern rows summing to 1, and `input + attention_output + mlp_output == layer_output`. The results are marked **[smoke]** below.

No code was written into `~/wd/nnter`.

---

## 0. Cross-cutting findings (read these first; they gate several families)

1. **Multimodal wrappers put the text stack at `model.language_model.*`.**
   - `lookup` already uses `text_config.model_type`, but RENAME keys spell `model.layers`. With `task="text-generation"`, `AutoModelForCausalLM` builds `Gemma4ForConditionalGeneration` / `Gemma3ForConditionalGeneration` / `Gemma3nForConditionalGeneration`, whose blocks sit at `model.language_model.layers` (verified on meta). A key that resolves nowhere is skipped, so a family can carry **both** spellings (`"model.language_model.layers": "layers"`, and the same for `embed_tokens` / `norm`).
   - That one change turns the shipped `gemma3_text` family on for the `gemma3` checkpoints. Those are 9.7M downloads, and NDIF lists gemma-3-4b/12b/27b pt/it (COLD).
   - `qwen3_vl`, `mistral3` and `llava` are **not** in the causal-LM auto map at all (`AutoModelForCausalLM` raises "Unrecognized configuration class"). They need an image-text-to-text load path first. That matters because NDIF has Qwen3-VL-8B **HOT** and Mistral-Small-3.1, Llava-1.5 and Qwen3-VL-2B COLD.
2. **Logit transforms other than softcapping.** `project_on_vocab` reads `config.final_logit_softcapping` only, and reads it off the top-level config.
   - Cohere/Cohere2 multiply by `logit_scale`, Granite / GraniteMoE* divide by `logits_scaling`, and Falcon-H1 multiplies by `lm_head_multiplier`.
   - Gemma-4 keeps `final_logit_softcapping` inside `text_config`, so on a multimodal load the current lookup returns `None`.
   - Suggest a family-level `def logits(model, x)` hook, or a lookup that reads `get_text_config()`.
3. **Scaled residual adds break the contribution identity.** The affected families are Granite (`residual + h * residual_multiplier`, 0.22 on granite-4.1-3b), HyperCLOVAX, MiniMax (alpha/beta), Doge (learned `input_residual`) and Zaya (`ZayaResidualScaling`).
   - `attention_output` and `mlp_output` must be the *scaled* tensors. That means either a `SourceEProperty` on the block's multiply or a transform-carrying copy (the Falcon `mlp_output` pattern).
   - The public tiny Granite checkpoints have `residual_multiplier = 1.0`, so the identity passes there **[smoke]**. A test needs a config rewrite to exercise the scale.
4. **One pure-torch fallback for every recurrent mixer in 5.17.** Every mixer family now calls one `@use_kernel_func_from_hub_with_fallback` function, the same way `LinearAttention` already reads `torch_chunk_gated_delta_rule`:

   | Mixer | Calls | Families |
   |---|---|---|
   | Mamba-1 | `mamba_selective_scan` / `mamba_selective_state_update` | mamba, falcon_mamba, jamba, zamba |
   | Mamba-2 | `mamba2_chunk_scan` / `mamba2_selective_state_update` | mamba2, nemotron_h, granitemoehybrid, bamba, falcon_h1, zamba2 |
   | KDA | `chunk_kimi_delta_attention` / `recurrent_kimi_delta_attention` | kimi_linear, glm5_next |
   | Gated DeltaNet | `torch_chunk_gated_delta_rule` (same names as Qwen3-Next) | olmo_hybrid, qwen4_exp |
   | Short conv | `causal_conv1d_fn` only | lfm2, lfm2_moe, inkling |

   So one `SSMMixer` component, the twin of `LinearAttention`, covers 11 families.
   - **Environment trap:** `ndif2` has `mamba_ssm` installed. Every Mamba-family tiny checkpoint then **fails on CPU** (`u.is_cuda()` / `exchangeDevice`) because the hub dispatch picks the CUDA kernel.
   - With `sys.modules['mamba_ssm'] = None`, all 9 run on the pure-torch path. The component needs a `needs_torch_kernels` analogue, and the suite needs the fallback forced.
5. **One block holds one sublayer (NemotronH).** Each `NemotronHBlock` has a single `mixer` that is Mamba2, attention, MoE or MLP depending on `layers_block_type`.
   - RENAME is name-based, so it cannot alias `mixer` to `self_attn` on some blocks and to `mlp` on others. It needs either a type-keyed alias in nnsight/nnter or a documented `layers[i].mixer` with typed envoys.
   - The per-block identity is `input + mixer_output == layer_output`.
6. **Hyper-connection residuals.** DeepSeek-V4 (`hc_mult = 4`), GLM-5-Next (`hc_mult = 4`) and Qwen4-Exp (`Qwen4ExpTextGatedResidual`) carry an `[B, T, hc, D]` or `[B, T, hc·D]` residual. That residual is mixed by learned `pre` / `post` / `comb` maps and collapsed by an `hc_head` before `norm`.
   - `layer_output` is not a `[B, T, D]` stream, contributions are not additive, and a logit lens has to go through `hc_head`.
   - This is a design decision, not a family module. Treat all three as one project.
7. **Per-layer sizes.** Gemma-4 has per-layer `head_dim` / `num_key_value_heads` (`config.per_layer_config`; the tiny checkpoint has head_dim 16 on sliding layers and 32 on full layers), so root sizes stop being scalars. DeepSeek-V4 and the MLA families already sit near this edge.

Nothing a candidate needs is deprecated in 5.17 (`DEPRECATED_MODELS == []`, `models/deprecated/` is empty).

---

## 1. Ranked top 16

Effort scale:
- **T (trivial):** Llama-like; the 3 container keys plus class names.
- **S:** small overrides; ≤½ day.
- **D:** own-arithmetic or per-block overrides; about a day.
- **C:** needs a new component or a core change.

| # | model_type | Why (signal) | Effort | Test checkpoint (config parses on 5.17) |
|---|---|---|---|---|
| 1 | `gemma4_text` | 42.1M dl (#6 type on the Hub); NDIF lists gemma-4-31B, -31B-it, -26B-A4B, -12B-it | D + core (§0.1, §0.2, §0.7) | `hf-tiny-v2/tiny-random-Gemma4ForCausalLM` (text; exercises PLE, MoE, KV sharing, per-layer head_dim); `trl-internal-testing/tiny-Gemma4ForConditionalGeneration` (wrapper path) |
| 2 | `glm4_moe` | GLM-4.5/4.6-Air; NDIF COLD GLM-4.5-Air; 0.8M dl | **T [smoke: status {}, identity 0]** | `trl-internal-testing/tiny-Glm4MoeForCausalLM` |
| 3 | `glm4_moe_lite` | GLM-4.7-Flash 1.85M dl | **T (MLA sizes from deepseek_v2) [smoke]** | `hf-tiny-v2/tiny-random-Glm4MoeLiteForCausalLM` |
| 4 | `glm_moe_dsa` + `deepseek_v32` | GLM-5.x 4.3M dl; DeepSeek-V3.2 3.3M dl | S | `tiny-random/glm-5`, `hf-tiny-v2/tiny-random-GlmMoeDsaForCausalLM`, `hf-tiny-v2/tiny-random-DeepseekV32ForCausalLM` (all run) |
| 5 | `olmoe` | NDIF COLD OLMoE-1B-7B; AllenAI, an open-MoE interpretability staple | **T [smoke]** | `hf-internal-testing/tiny-random-OlmoeForCausalLM` |
| 6 | `olmo_hybrid` | AllenAI Olmo-Hybrid-7B (15.7k dl); NDIF already hosts the OLMo line | S **[smoke: every DeltaNet value resolves with the base `LinearAttention`]** | `hf-tiny-v2/tiny-random-OlmoHybridForCausalLM` |
| 7 | `nemotron_h` | 12.9M dl (Nemotron-3 Nano/Super/Ultra); NDIF COLD Nemotron-3-Super-120B | C (Mamba2 component + per-block alias) | `hf-tiny-v2/tiny-random-NemotronHForCausalLM`, `trl-internal-testing/tiny-NemotronHForCausalLM-nano` (run only with `mamba_ssm` hidden) |
| 8 | `qwen3_vl_text` (+`qwen3_vl_moe_text`) | 31.1M + 2.3M dl; **NDIF HOT Qwen3-VL-8B** | T family, blocked on the multimodal load path (§0.1) | `trl-internal-testing/tiny-Qwen3VLForConditionalGeneration`, `hf-tiny-v2/tiny-random-Qwen3VLTextModel`, `yujiepan/qwen3-vl-moe-tiny-random` |
| 9 | `granite` (+`granitemoe`, `granitemoeshared`) | Granite 3.x/4.1/4.2 dense, 1.8M dl; NDIF COLD granite-4.2-30b | S + logit hook | `hf-internal-testing/tiny-random-GraniteForCausalLM`, `hf-internal-testing/tiny-random-GraniteMoeForCausalLM`, `hf-tiny-v2/tiny-random-GraniteMoeSharedForCausalLM` |
| 10 | `mamba` (+`falcon_mamba`) | The SSM interpretability papers (ROME in Mamba, IOI in Mamba, SAE studies); state-spaces/mamba-130m…2.8b-hf | C (SSM component) | `hf-internal-testing/tiny-random-MambaForCausalLM`, `trl-internal-testing/tiny-FalconMambaForCausalLM` |
| 11 | `deepseek_v4` | 8.6M dl (V4-Flash); NDIF COLD DeepSeek-V4-Flash | C (hyper-connection residual, §0.6) | `trl-internal-testing/tiny-DeepseekV4ForCausalLM` (runs) |
| 12 | `cohere` / `cohere2` | Command-R (183k), tiny-aya; classic parallel block | T + logit hook **[smoke cohere: identity 0]** | `trl-internal-testing/tiny-CohereForCausalLM`, `trl-internal-testing/tiny-Cohere2ForCausalLM` |
| 13 | `gpt_neo` | TinyStories-1M…33M (218k dl; a staple of interpretability tutorials and papers); GPT-Neo 125M–2.7B | D (own arithmetic, GPT-J pattern) | `hf-internal-testing/tiny-random-GPTNeoForCausalLM` |
| 14 | `llama4_text` | Llama-4 Scout/Maverick (377k dl) | T code; test checkpoint is a problem | `trl-internal-testing/tiny-Llama4ForCausalLM` parses but **NaNs in plain transformers 5.17**; the yujiepan / optimum / tiny-random ones fail config validation (`attn_temperature_tuning` int → needs a rewrite) |
| 15 | `lfm2` (+`lfm2_moe`) | LiquidAI LFM2/2.5 (1.2M dl on native configs, far more as GGUF) | D (short-conv mixer + renames) | `trl-internal-testing/tiny-Lfm2ForCausalLM`, `optimum-intel-internal-testing/tiny-random-lfm2-moe` |
| 16 | `minimax_m2` | MiniMax-M2.5/2.7, 1.9M dl | **T [smoke]**; public checkpoints carry `auto_map` but the class is native | `hf-tiny-v2/tiny-random-MiniMaxM2ForCausalLM` |

Honourable mention at the same tier: **`exaone4`** is **T [smoke with Gemma-2-style overrides: identity 0]**, with the OLMo-2 post-norm layout (`hf-tiny-v2/tiny-random-Exaone4ForCausalLM`).

---

## 2. Per-candidate detail

### 1. `gemma4_text` (Gemma 4)

- **Classes.** `Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration`; block `Gemma4TextDecoderLayer`; attention `Gemma4TextAttention`; MLP `Gemma4TextMLP`. The MoE is **not** a module under `mlp`: `router` (`Gemma4TextRouter`) and `experts` (`Gemma4TextExperts`) are direct block children.
- **Structure.** A Gemma-2-style sandwich (`input_layernorm → attn → post_attention_layernorm`, `pre_feedforward_layernorm → mlp → post_feedforward_layernorm`). Three additions:
  - **MoE blocks** (`enable_moe_block`, 26B-A4B): `post_feedforward_layernorm_1(mlp) + post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(x)))`, then `post_feedforward_layernorm`. The router reads the residual, not the normed input.
  - **Per-layer embeddings (PLE)** (`hidden_size_per_layer_input`, E2B/E4B = 256): a third residual add from `per_layer_input_gate → act → * per_layer_input → per_layer_projection → post_per_layer_input_norm`.
  - `hidden_states *= layer_scalar`, an in-place multiply by a loaded buffer (1.0 in the tiny checkpoint; not verified on real weights).
- **Attention.** On the shared interface. `scaling = 1.0` (q_norm/k_norm/v_norm). `attention_k_eq_v` on global layers of 31B/26B means `v_proj` is `None` and the values come from `k_proj` before `k_norm`. KV sharing: the last `num_kv_shared_layers` layers (18 on E4B, 20 on E2B) take keys and values from `shared_kv_states` and have no `k_proj` / `v_proj`. Sliding and full layers alternate, and `head_dim` / `num_kv_heads` differ between them.
- **Family needs:**
  - RENAME with both spellings (`model.*` and `model.language_model.*`).
  - `Attention.attention_output = ../post_attention_layernorm.output` and `Mlp.mlp_output = ../post_feedforward_layernorm.output`, which is correct on MoE blocks because that norm follows the sum.
  - A new block-level value or an explicit carve-out for the PLE contribution. [smoke] the identity misses by about 2.7 on every layer of the tiny checkpoint because of PLE.
  - A decision on `layer_scalar`.
  - Per-layer `head_dim`.
  - Softcap read from `text_config`.
- **Expected `unavailable`:** none under eager on the smoke run (`status() == {}`). The keys and values on KV-shared layers are another layer's, which should be documented rather than marked unavailable.
- **Also:** `gemma4_unified_text` (Gemma-4-12B) is a separate `model_type` with a simpler block (sandwich plus `layer_scalar`, no PLE or MoE). It could share the module through `MODEL_TYPES = ("gemma4_text", "gemma4_unified_text")` if the registry allows it, or be a second trivial file (`hf-tiny-v2/tiny-random-Gemma4UnifiedForCausalLM`).

### 2. `glm4_moe` (GLM-4.5 / 4.6 / Air)

- **Classes.** `Glm4MoeForCausalLM`; `Glm4MoeDecoderLayer` / `Glm4MoeAttention` / `Glm4MoeMoE` + `Glm4MoeMLP` (the first `first_k_dense_replace` blocks are dense).
- **Structure.** Llama names. Attention on the interface, with partial rotary and an optional `q_norm`. The MoE has a shared expert, and its "residual" variable is internal to it.
- **Family needs:** the 3 container keys, and both MLP classes keyed to `Mlp`.
- **Unavailable:** none **[smoke]**.

### 3. `glm4_moe_lite` (GLM-4.7-Flash)

- **Structure.** DeepSeek-V3-style MLA (`kv_lora_rank`) with `mlp_layer_types` dense/sparse. Attention on the interface; bare-tensor block.
- **Family needs:** the 3 keys, plus `head_dim` / `qk_head_dim` imported from `deepseek_v2`. Keys are expanded to `num_heads`, so it is `KV_HEADS_EXPANDED`.
- **Unavailable:** none **[smoke]**.

### 4. `glm_moe_dsa` (GLM-5) and `deepseek_v32` (DeepSeek-V3.2)

- **Structure.** MLA plus DeepSeek Sparse Attention. A `GlmMoeDsaIndexer` picks the top-k keys, and under eager/sdpa the result is folded into `attention_mask` before `attention_interface`. The interior values are therefore on the interface and the pattern is the sparse one.
  - **GLM-5 differences:** its block returns `(hidden_states, topk_indices)`, so it needs `returns_tuple = True`. Its attention returns a 3-tuple, which the base takes `[0]` of. Layers without an indexer reuse `prev_topk_indices`.
  - **DeepSeek-V3.2:** its block returns a bare tensor.
- **Family needs:** the MLA sizes from `deepseek_v2` and `returns_tuple` (GLM-5).
- **Worth adding:** an optional `attention_topk_indices` value, since interpretability work will want the index selection.
- **Unavailable:** none expected. Not smoke-run; the forward passes run.

### 5. `olmoe`

- **Structure.** `OlmoeDecoderLayer` / `OlmoeAttention` (q_norm/k_norm) / `OlmoeSparseMoeBlock`. Pure Llama layout.
- **Family needs:** the 3 keys.
- **Unavailable:** none **[smoke]**.

### 6. `olmo_hybrid`

- **Structure.** Two block classes.
  - `OlmoHybridLinearAttentionDecoderLayer` is **pre-norm**: `input_layernorm → linear_attn` (`OlmoHybridGatedDeltaNet`) and `post_attention_layernorm → mlp`.
  - `OlmoHybridAttentionDecoderLayer` is **OLMo-3 post-norm**: `self_attn → post_attention_layernorm` and `mlp → post_feedforward_layernorm`.
  - The GDN calls `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` exactly as Qwen3-Next does.
- **[smoke]** With the base `LinearAttention`, every `linear_attn.*` value resolves on the linear block, `state` / `states` report the usual `route_delta_rule` instruction, and the identity holds (err 0). On the attention block the identity fails (0.59) until the post-norm overrides are added.
- **Family needs:**
  - `Attention.attention_output = ../post_attention_layernorm.output`.
  - `Mlp.mlp_output` chosen **by the parent block's class**, because both block types share `OlmoHybridMLP`: `post_feedforward_layernorm.output` on attention blocks and `mlp.output` on linear blocks. A `by_alibi`-style callable op would do it.
- **Unavailable:** as Qwen3-Next.

### 7. `nemotron_h` (Nemotron-3 / Nemotron-H)

- **Native names.** `model.embeddings`, `model.layers[i].{norm, mixer}`, `model.norm_f`.
- **Structure.** The mixer is `NemotronHMamba2Mixer`, `NemotronHAttention` (on the interface), `NemotronHMoE` or `NemotronHMLP`, and the block is `residual + mixer(norm(x))`.
- **Family needs:**
  - RENAME `model.embeddings → embed_tokens`, `model.norm_f → norm`.
  - The per-type alias problem in §0.5.
  - A Mamba-2 component reading `mamba2_chunk_scan` inputs (x, dt, A, B, C, D), its output and final state. The state is per chunk under the chunked kernel, so it has the same `route`-style caveat as DeltaNet.
- **Unavailable:** `self_attn.*` on non-attention blocks, and `mlp.*` on mixer blocks.
- **Also unlocks** `granitemoehybrid` (Granite-4.0-H), `bamba`, `falcon_h1`, `zamba2` and `mamba2` with little extra work.

### 8. `qwen3_vl_text` / `qwen3_vl_moe_text`

- **Structure.** The text block is Qwen3 (q_norm/k_norm) with interleaved MRoPE.
- **DeepStack.** In `Qwen3VLTextModel.forward`, visual features are added to the hidden states **between** blocks (`_deepstack_process` after layer *i*). On image tokens, `layers[i].layer_output != layers[i+1].input`. Nothing changes for text-only prompts.
- **Family needs:** trivial, apart from the `model.language_model.layers` path. The hard part is loading: there is no causal-LM auto class, so it needs the image-text-to-text path (§0.1).

### 9. `granite` (+`granitemoe`, `granitemoeshared`)

- **Native names.** Llama names. The MoE variants use `block_sparse_moe` (+ `shared_mlp`), so they need RENAME `block_sparse_moe → mlp`. The shared MLP is a second contribution that stays native, or gets summed.
- **Structure.** `residual + h * residual_multiplier` on both sublayers, `embedding_multiplier` on the embeddings, and `logits / logits_scaling`.
- **Family needs:** scaled `attention_output` / `mlp_output` (§0.3) and a logit hook (§0.2).
- **Test:** rewrite `residual_multiplier` in the test config (the tiny checkpoints have 1.0).
- **Related, same pattern:** `granite_swa` / `granitemoe_swa`, which also add attention sinks.

### 10. `mamba` / `falcon_mamba`

- **Native names.** `backbone.embeddings`, `backbone.layers[i].{norm, mixer}`, `backbone.norm_f`, `lm_head`.
- **Structure.** No attention and no MLP; the block is `residual + mixer(norm(x))`.
- **Family needs:** RENAME for the 3 containers, `mixer → ssm` (or a new standard name), and the Mamba-1 SSM component on `mamba_selective_scan`.
- **Unavailable:** all `self_attn.*` and `mlp.*` are absent (not listed).
- **Watch:** the `mamba_ssm` routing trap (§0.4). `falcon_mamba` adds weightless B/C/dt RMSNorms and is otherwise identical.

### 11. `deepseek_v4`

- **Structure.** `DeepseekV4DecoderLayer` holds `attn_hc` / `ffn_hc` (`DeepseekV4HyperConnection`, a Sinkhorn-normalised `comb`) and returns `post·f(x) + combᵀ·streams` over 4 streams. The model ends with `hc_head → norm`.
- **Attention.** On the interface, with `s_aux = sinks` (a GPT-OSS-style sink). Keys **and values are the same tensor** (`kv, kv`). Compressed KV columns (HCA/CSA compressors) are appended, so keys are longer than the sequence. The output is de-rotated after the interface and projected through grouped `o_a` / `o_b`.
- **MoE.** Includes a hash router.
- **Family needs:** the §0.6 design, plus `SINK = True` and a scores override like `gpt_oss`.

### 12. `cohere` / `cohere2`

- **Structure.** A parallel block with one `input_layernorm` (LayerNorm, not RMS): `x + attn(n) + mlp(n)`.
- **Family needs:** the 3 keys, and the logit hook for `logit_scale` (§0.2).
- **cohere2** additionally mixes sliding and global layers, with no RoPE on global layers.
- `cohere2_moe` (`hf-tiny-v2/tiny-random-Cohere2MoeForCausalLM`) has a dense prefix, then MoE.

### 13. `gpt_neo`

- **Native names.** `transformer.{wte, wpe, h, ln_f}`. The block (`ln_1`, `attn`, `ln_2`, `mlp`) returns a **tuple**.
- **Attention.** `attn` is a `GPTNeoAttention` wrapper around `attn.attention` (`GPTNeoSelfAttention`), which does its own arithmetic in `_attn`: an fp32 `matmul`, a causal/local mask by `torch.where`, `softmax`, `attn_dropout`, `matmul`. There is **no 1/√d scaling**, and local and global layers alternate (`attention_types`).
- **Family needs:**
  - RENAME `attn.attention → self_attn`, or key `Attention` on the inner class. The wrapper passes the output straight through.
  - The six interior values mapped on `self__attn_0`, like GPT-J: the `matmul` input for scores, the `softmax` output, `attn_dropout_0` for the pattern.
  - `returns_tuple = True`, and `intermediate_size` from `intermediate_size` or `4 * hidden_size`.

### 14. `llama4_text`

- **Native names.** `feed_forward` instead of `mlp` (RENAME `feed_forward → mlp`). The MoE (`Llama4TextMoe`) returns `(out, router_logits)`, which the base unwraps. Dense layers use `Llama4TextMLP`.
- **Attention.** On the interface, with L2 qk-norm, NoPE layers (`no_rope_layers`), chunked attention and temperature tuning.
- **Family needs:** the `language_model` path for `llama4` wrapper checkpoints.
- **Blocker:** a usable tiny checkpoint. Both a config-rewrite route and a forward pass without NaN need checking.

### 15. `lfm2` / `lfm2_moe`

- **Native names.** Block `{operator_norm, self_attn | conv, ffn_norm, feed_forward}`. The **final norm is `model.embedding_norm`**; despite the name it runs after the last block.
- **Structure.** Conv blocks use `Lfm2ShortConv` (a gated `causal_conv1d_fn`).
- **Family needs:** RENAME `feed_forward → mlp`, `operator_norm → input_layernorm`, `ffn_norm → post_attention_layernorm`, `model.embedding_norm → norm`. Either a small `ShortConv` component or a native-only conv (with `attention_output` taken from `conv.output`).
- **Unavailable:** `self_attn.*` on conv blocks.

### 16. `minimax_m2`

- **Structure.** Llama names, a sparse MoE block, q_norm/k_norm, attention on the interface.
- **Family needs:** the 3 keys.
- **Unavailable:** none **[smoke]**.

---

## 3. Also possible (by shape)

**Llama-like, trivial (3 keys, maybe one alias):**
- `ministral` (`hf-tiny-v2/tiny-random-MinistralForCausalLM`) and `ministral3` (`hf-tiny-v2/tiny-random-Ministral3ForCausalLM`, runs).
- `mistral4`: MLA like DeepSeek. **Not** in the causal-LM auto map (`AutoModelForCausalLM` rejects `Mistral4Config`). Checkpoint `hf-tiny-v2/tiny-random-Mistral4ForCausalLM`.
- `glm` (`hf-tiny-v2/tiny-random-GlmForCausalLM`).
- `helium` (`hf-internal-testing/tiny-random-HeliumForCausalLM`).
- `arcee` (`hf-tiny-v2/tiny-random-ArceeForCausalLM`).
- `hunyuan_v1_dense` / `hunyuan_v1_moe` (`hf-tiny-v2/tiny-random-HunYuan*`).
- `ernie4_5` / `ernie4_5_moe` (`hf-tiny-v2/tiny-random-Ernie4_5*`).
- `seed_oss` (`hf-tiny-v2/tiny-random-SeedOssForCausalLM`).
- `starcoder2` (LayerNorm; `hf-internal-testing/tiny-random-Starcoder2ForCausalLM`).
- `nemotron` (LayerNorm1P; `hf-tiny-v2/tiny-random-NemotronForCausalLM`).
- `persimmon` (qk-LayerNorm; `hf-internal-testing/tiny-random-PersimmonForCausalLM`).
- `phimoe` (`hf-tiny-v2/tiny-random-PhimoeForCausalLM`).
- `dots1` (`hf-tiny-v2/tiny-random-Dots1ForCausalLM`).
- `laguna` (`hf-tiny-v2/tiny-random-LagunaForCausalLM`; public checkpoints carry `auto_map`).
- `solar_open` (`onnx-internal-testing/tiny-random-SolarOpenForCausalLM`).
- `youtu`, `axk1`, `hy_v3`, `mellum`, `cwm`, `jais2`, `bitnet`, `nanochat` (softcap), `mimo_v2_flash` (attention sinks → `SINK = True`, `hf-tiny-v2/tiny-random-MiMoV2FlashForCausalLM`).
- `apertus` (RENAME `attention_layernorm → input_layernorm`, `feedforward_layernorm → post_attention_layernorm`; `hf-tiny-v2/tiny-random-ApertusForCausalLM`).
- `jetmoe` (attention is `self_attention` and returns a 3-tuple; `hf-tiny-v2/tiny-random-JetMoeForCausalLM`).

**Post-norm / sandwich, S (Gemma-2 / OLMo-2 overrides):**
- `exaone4` **[smoke]**.
- `flex_olmo` (OLMo-2 layout with MoE; `hf-tiny-v2/tiny-random-FlexOlmoForCausalLM`).
- `glm4`: sandwich with extra norms. `post_self_attn_layernorm` follows the attention; `post_attention_layernorm` is the *pre*-MLP norm; `post_mlp_layernorm` follows the MLP. Checkpoint `hf-tiny-v2/tiny-random-Glm4ForCausalLM`.
- `afmoe` (`pre_mlp_layernorm` / `post_mlp_layernorm`).
- `vaultgemma`: Gemma with pre-norms only, so the base holds. Checkpoint `hf-tiny-v2/tiny-random-VaultGemmaForCausalLM`.

**Scaled residual, S–D (§0.3):** `hyperclovax` (sandwich + multiplier), `minimax` (alpha/beta + lightning attention with its own arithmetic → C), `doge`, `zaya` (tuple block + `ZayaResidualScaling`), `granite_swa`, `granitemoe_swa`.

**Own attention arithmetic, D:**
- `codegen`: GPT-J-like `_attn`. Checkpoint `hf-internal-testing/tiny-random-CodeGenForCausalLM`.
- `xglm`: own softmax; `fc1` / `fc2` on the block, like OPT. Checkpoint `hf-internal-testing/tiny-random-XGLMForCausalLM`.
- `gpt_neox_japanese` (own).
- `gpt_bigcode` is **on** the interface (StarCoder-1 / SantaCoder, MQA): T/S. Checkpoint `hf-internal-testing/tiny-random-GPTBigCodeForCausalLM`.
- `diffllama`: calls the interface **twice** per layer (two value halves), then differential lambda plus groupnorm. The base would bind `attention_interface_1` to the first map only, so it needs a second set of values or an explicit "first map" caveat.

**New mixer component, C:**
- Mamba-2 hybrids `granitemoehybrid`, `bamba`, `falcon_h1` (+ `lm_head_multiplier`), `zamba`, `zamba2` (shared attention blocks), `jamba`. All run on the pure-torch path with `mamba_ssm` hidden.
- `kimi_linear` (KDA + MLA). **Neither tiny config parses on 5.17** (`yujiepan/kimi-linear-tiny-random` and `tiny-random/kimi-linear` fail strict validation on `layer_types`); public checkpoints carry `auto_map`.
- `recurrent_gemma` (RG-LRU, own; `hf-tiny-v2/tiny-random-RecurrentGemmaForCausalLM`).
- `rwkv` (own WKV with a CPU fallback; `hf-internal-testing/tiny-random-RwkvForCausalLM`).
- `xlstm` (`hf-tiny-v2/tiny-random-xLSTMForCausalLM`).
- `inkling_text`: short convs wrapped around both sublayers, inside the residual. Checkpoint `hf-tiny-v2/tiny-random-InklingForCausalLM`.

**Hyper-connection residual, C (§0.6):**
- `glm5_next_text` (GLM-5.3-Flash, 5.3M dl, KDA + DSA). No tiny checkpoint found.
- `qwen4_exp_text` (Qwen3.8-Flash-Next, 2.7M dl; GDN + MoE + PLE + `Qwen4ExpTextGatedResidual`, and no norm modules in the block). No tiny checkpoint found.
- `hy_v4`. No tiny checkpoint found.

**Hard structure:**
- `gemma3n_text`: AltUp predict/correct over 4 streams, LAuReL, PLE. Checkpoints `hf-tiny-v2/tiny-random-Gemma3nForCausalLM`, `optimum-intel-internal-testing/tiny-random-gemma3n-text`.
- `longcat_flash`: two attentions and two MLPs per block as ModuleLists, plus a zero-compute MoE.
- `hrm_text`: H/L recurrent stacks, not one layer list.
- `blt`: byte latent, three stacks.

---

## 4. Not native in 5.17 (need `trust_remote_code`, or absent)

| What | Status in 5.17 | Note |
|---|---|---|
| `internlm2`, `exaone` (3.x), `openelm`, `qwen` (v1), `minicpm`/`minicpmv`, `phi3_v`, `llada`, `bailing_hybrid`, `sarvam_moe`, `mimo_v2`, `talkie` | Remote-code only | — |
| `kimi_k2` / `kimi_k25` / `kimi_k3` | Remote-code only | NDIF lists Kimi-K2.5 COLD. The text model is DeepSeek-V3-shaped, so a registered alias `MODEL_TYPES = ("kimi_k2",)` reusing `deepseek_v3` would work *if* loaded with `trust_remote_code`. |
| `deepseek_v41` (DeepSeek-V4.1-Flash, 0.8M dl), `apertus1p5` | Not in 5.17 at all | Needs a newer transformers |
| `minimax_m2`, `nemotron_h`, `laguna` | Native | Public checkpoints still ship `auto_map` |

---

## 5. Suggested order

1. **Batch of trivial families**, each a copy of `llama.py`: `glm4_moe`, `glm4_moe_lite`, `olmoe`, `minimax_m2`, `exaone4`, `cohere`/`cohere2` (after the logit hook), plus the Llama-like list in §3. This takes about a day in total. Four of them are already smoke-verified.
2. **Multimodal RENAME keys and the `get_text_config()` softcap fix.** This lights up `gemma3` checkpoints right away and is a prerequisite for `gemma4_text`. Then `gemma4_text` itself.
3. **`glm_moe_dsa` / `deepseek_v32`, `granite`*, `olmo_hybrid`, `gpt_neo`.**
4. **An `SSMMixer` component** (Mamba-1 / Mamba-2 on the hub-fallback functions, with a kernel-routing guard). Then `mamba`, `nemotron_h` (plus the per-type alias), `granitemoehybrid`, `falcon_h1`, `bamba`, `zamba2`, `jamba`.
5. **Design the hyper-connection residual**, then `deepseek_v4`, `glm5_next_text` and `qwen4_exp_text`.

Web sources used:
- [Hub API](https://huggingface.co/api/models)
- [NDIF status](https://api.ndif.us/status)
- [Gemma Scope 2 announcement](https://www.lesswrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2)
- [Locating and Editing Factual Associations in Mamba](https://arxiv.org/pdf/2404.03646)
- [Investigating the IOI circuit in Mamba](https://arxiv.org/pdf/2407.14008)
