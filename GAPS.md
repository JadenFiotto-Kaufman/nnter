# nnterp vs nnter — gap report

Produced 2026-09-28 by a read-only comparison of `~/wd/nnterp` (branch
`internals-accessors`, HEAD `2db11d8`) against this package. File:line
references into nnterp are as of that commit. Sections: (1) nnterp's surface,
(2) what nnter lacks, (3) deliberate differences, (4) fragile spots in nnter
seen from nnterp's experience.

## 1. nnterp's surface, inventoried

### Package entry (`nnterp/__init__.py`)
| symbol | what | where |
|---|---|---|
| `StandardizedTransformer`, `StandardizedVLM`, `load_model`, `detect_automodel`, `get_rename_dict`, `ModuleAccessor` | `__all__` | `__init__.py:11-18` |
| `load_model(model, use_vllm, allow_experimental_vllm, text_only, **kw)` | picks `StandardizedVLLM` / `StandardizedVLM` / `StandardizedTransformer` by `detect_automodel` | `__init__.py:21-66` |

### `standardized_transformer.py`
| symbol | what | where |
|---|---|---|
| `COMPATIBILITY_PROPERTIES = ("logits", "token_embeddings")` | the two rows exposed as tensor properties, not accessors | `:47` |
| `StandardizationMixin` | everything shared by HF / VLM / vLLM wrappers | `:50` |
| published ints: `num_layers, attention_layers, linear_attention_layers, num_heads, hidden_size, vocab_size, head_dim, qk_head_dim, num_kv_heads, intermediate_size, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim, linear_attention_kernels, block_structure, is_vllm, remote` | set in `_init_standardization` | `:111-127`, `:220-252` |
| 23 accessor attributes `internals, embeddings_input, embeddings_output, ln_final_output, lm_head_output, layers_input, layers_mid, layers_output, attentions, attentions_input, attentions_norm_output, attentions_premix, attentions_output, attention_queries, attention_keys, attention_scores, attention_probabilities, attention_head_outputs, mlps, mlps_input, mlps_norm_output, mlps_activation, mlps_neurons, mlps_output` | one `LayerAccessor` per row of the address table, `setattr` on the model; a clash with an existing attribute is a `RenamingError` | `:131-154`, `:209-218` |
| `_init_standardization(...)` | builds addresses, vLLM row surgery, `Internals`, sizes, kernel pinning, ignores, `check_model_renaming`, `check_attention_probabilities` or `.disable()` | `:156-296` |
| `_get_rename`, `_prepare_init_kwargs` | merges `RenameConfig` + user `rename=`; `device_map="auto"` default; `enable_attention_probs` forces `attn_implementation="eager"`, non-eager raises | `:298-328` |
| `detect_layer_output_type()` | records tuple-ness of unread layers for `skip_layers` | `:330-353` |
| `add_prefix_false_tokenizer` | lazy second tokenizer with `add_prefix_space=False` | `:355-368` |
| `attn_probs_available` | `unavailable_on(attention_layers[0]) is None` | `:370-375` |
| `input_ids`, `input_size`, `attention_mask` | from `self.inputs[1]`; vLLM shapes `(seq,)` | `:377-402` |
| `token_embeddings` (get/set) | reads/writes the `embeddings_output` row | `:404-413` |
| `next_token_probs` | `logits[:, -1].softmax(-1)`, assumes left padding | `:415-419` |
| `skip_layer`, `skip_layers(start, end, skip_with, layer_returns_tuple)` | `.skip()` with `(tensor, DummyCache())` on tuple layers | `:421-464` |
| `steer(layers, vector, factor, positions*, token_positions, batch_index)` | in-place `+=` on HF, clone+assign on vLLM | `:466-535` |
| `project_on_vocab(h)` | `lm_head(ln_final(h))`; refuses outside trace on vLLM | `:537-551` |
| `probs_to_dict`, `get_topk_closest_tokens(h, k)` | | `:553-595` |
| `StandardizedTransformer(model, check_renaming, remote, allow_dispatch, enable_attention_probs, check_attn_probs_with_trace, rename_config, automodel, text_only, tokenizer_kwargs, **kw)` | VLM warning via `detect_automodel`; `task="text-generation"` | `:598-727` |
| `_remoteable_class`, `dispatch()` (re-pins kernels), `logits` property | | `:729-743` |
| `StandardizedVLM(...)` | same mixin over `task="image-text-to-text"`, `allow_multimodal` | `:746-824` |

### `rename_utils.py` (1964 lines)
| symbol | what | where |
|---|---|---|
| `RenamingError` | | `:34` |
| `AttnProbFunction` (ABC) | legacy custom probability op | `:38-50` |
| `RenameConfig` fields: `attn_name, mlp_name, ln_final_name, lm_head_name, model_name, layers_name, attn_prob_source, ignore_mlp, ignore_attn, attn_head_config_key, hidden_size_config_key, vocab_size_config_key, attn_output_source, mlp_output_source, addresses` | | `:53-156` |
| `MODEL_NAMES = [transformer, gpt_neox, decoder, language_model]` → `model` | | `:159` |
| `expand_path_with_model` | expands `model.x` to every container spelling | `:162-171` |
| `default_*_config_keys` | `n_heads/num_attention_heads/n_head/num_heads`; `hidden_size/d_model/n_embd`; `vocab_size/n_vocab` | `:175-184` |
| `bloom_slow_but_exact(model)` | BLOOM `pretraining_tp>1 & slow_but_exact` has no contribution module | `:187-195` |
| `ATTENTION_NAMES = [attn, self_attention, attention, norm_attn_norm]` → `self_attn`; `LINEAR_ATTENTION_NAME = "linear_attn"` (never renamed); `LAYER_NAMES` (`h, blocks, model.layers`) → `layers`; `LN_NAMES` (`final_layer_norm, final_layernorm, ln_f, norm_f, norm, embedding_norm, model.ln_final`) → `ln_final`; `LM_HEAD_NAMES` (`embed_out, model.lm_head`) → `lm_head`; `MLP_NAMES` (`block_sparse_moe, feed_forward, ffn`) → `mlp`; `EMBED_TOKENS_NAMES` (`wte, embed_in, word_embeddings, model.embed_tokens`) → `embed_tokens` | | `:199-236` |
| `get_rename_dict(rename_config)` | user names first, then every list | `:239-269` |
| `text_config`, `get_num_attention_heads`, `get_hidden_size`, `get_vocab_size` | config lookups honoring `RenameConfig` keys | `:272-350` |
| `get_head_dim` (`v_head_dim` → `head_dim` → `hidden//heads`), `get_qk_head_dim` (MLA), `get_num_kv_heads` (Falcon `multi_query` → 1), `get_intermediate_size` (`n_inner`, DBRX `ffn_config`, MPT `expansion_ratio`, BLOOM 4×) | | `:353-409` |
| `IOType {INPUT, INPUTS, OUTPUT}` | nnsight's spellings | `:412-425` |
| `get_attention_layers(layers)` → `(softmax idx, linear idx)`; `linear_attention_error` | | `:428-447` |
| `Selection` ABC (`get`/`put`), `Index(*steps)`, `Copied()`, `FirstIfTuple()` | where the tensor sits in the value | `:450-513`, `:634-651` |
| `Address` dataclass: `module, io, op, select, order, unavailable, tags, per_layer, seq_axis, width, heads, keys, needs, scan` | one row of the table | `:516-631` |
| `LayerAccessor(model, address, io_type, name)`: `unavailable_on(layer)`, `num_heads`, `width`, `scan(layer, cuts, edit)`, `disable(reason)`, `get_module`, `get_operation(layer, containing_source)`, `__getitem__`/`__setitem__`/`__call__`, `returns_tuple(layer)`, `print_source(layer)` | | `:654-902` |
| `check_attention_probabilities(model, layer, allow_dispatch, use_trace)` | shape, rows sum to 1 (or `<1` with `sink` tag), and a random-pattern write moves the logits | `:905-986` |
| `_INTERFACE = "attention_interface_1"`, `INTERFACE_ROWS` | the shared attention call, as nnsight 0.8 names it | `:993-1001` |
| `delta_rule_call`, `pin_linear_attention_kernels`, `delta_rule_scan` | Gated DeltaNet: kernel picked by decode step; reference kernels pinned at load; state at any prompt position | `:1003-1162` |
| `DEFAULT_ADDRESSES` (25 rows: 5 whole-model, 15 block, 9 `linear_attn`) | | `:1198-1291` |
| `BlockStructure` = `pre_norm/sandwich_norm/post_norm/parallel/residual_inside`; `STRUCTURAL_ADDRESSES` | candidates per structure for `attentions_norm_output, attentions_premix, layers_mid, mlps_norm_output, mlps_activation, mlps_neurons` | `:1293-1373` |
| `FAMILY_ADDRESSES`; `POST_SUBLAYER_NORM_MODEL_TYPES` | | `:1407-1512` |
| `get_block_structure`, `structural_addresses`, `addresses_for`, `get_ignores` | | `:1515-1643` |
| `check_io`, `_check_has_module`, `_check_attention_layers`, `_warn_heterogeneous_types`, `_check_output_source`, `check_model_renaming` | load-time validation | `:1646-1943` |
| `HF_TO_VLLM_KWARGS_MAP`, `hf_kwargs_to_vllm_kwargs` | `max_new_tokens` → `max_tokens` | `:1946-1964` |

### `internals.py`
`Internals(dict[str, LayerAccessor])` ordered by `Address.order`; `__getitem__` raises `RenamingError` listing rows; `status(layer=None)`; `rank(name, layer)` (`internals.py:14-70`).

### `nnsight_utils.py`
`get_embed_tokens :19, get_layers :26, get_num_layers :35, get_layer :46, get_layer_input :58, get_layer_output :70, get_attention :83, get_attention_output :95, get_mlp :111, get_mlp_output :118, get_logits :127, get_unembed_norm :138, get_unembed :151, project_on_vocab :162, get_next_token_probs :175, set_layer_output :186, ModuleAccessor :203-256, get_token_activations :260, collect_last_token_activations_session :320, collect_token_activations_batched :381, compute_next_token_probs :434`. Most accept a raw `TransformersModel`.

### `interventions.py`
`logit_lens :29, TargetPrompt :71, repeat_prompt :76, it_repeat_prompt :102, TargetPromptBatch :169, patchscope_lens :234, patchscope_generate :304, patch_object_attn_lens :358`.

### `prompt_utils.py`
`TokenizationError :14, get_first_tokens :18, Prompt :97 (from_strings, has_no_collisions, get_target_probs, run), next_token_probs_unsqueeze :164, run_prompts :171`.

### `display.py` (extra `[display]`)
`plot_topk_tokens :13` (plotly heatmap), `prompts_to_df :119`.

### `standardized_vllm.py`
`StandardizedVLLM(VLLM, StandardizationMixin) :13`: experimental gate `:81-97`, `tensor_parallel_size` default `:98-101`, prefix-caching refusal `:102-112`, `trace()` defaults `max_tokens=1` `:131`, `generate()` `:145`, HF→vLLM kwarg translation `:158-166`. In the mixin: `logits` row popped and `INTERFACE_ROWS` unavailable on vLLM (`standardized_transformer.py:193-203`).

### `utils.py`
`TraceTensor :14`, `ArchitectureNotFound` + guarded imports of 14 model classes `:22-87`, `detect_automodel :90` (ImageTextToText > CausalLM > Seq2Seq; `text_only` picks a text tower and proves it builds on meta), `is_notebook :182`, `display_markdown :194`, `DummyCache :204`, `dummy_inputs :209`, `try_with_scan :216` (scan, then trace fallback), `unpack_tuple :276`. Also `logging.py` and `__main__.py` (`python -m nnterp run_tests --model-names/--class-names`).

### The ignore / unavailable mechanisms
- `Address.unavailable`: a string, or `fn(layer_module) -> reason|None` per layer (DeepSeek dense vs MoE blocks) — `rename_utils.py:561-565`, `:1556-1573`.
- `LayerAccessor.unavailable_on(layer)`: declared reason, linear-attention layer, missing child — `:700-723`; any access raises `RenamingError(reason)` `:788-791`.
- `LayerAccessor.disable(reason)` rewrites the address so `status()` and the raise agree — `:780-786`.
- `Internals.status(layer)` — `internals.py:42-57`.
- `get_ignores`: an "ignore" for the renaming checks is only a declared `unavailable` on `attentions_output` / `mlps` / `mlps_output`, plus `RenameConfig.ignore_mlp/ignore_attn` — `rename_utils.py:1609-1643`.
- `_no_interface(cls)`: marks the four interface rows unavailable on families whose attention does its own arithmetic — `:1393-1404`.

### Model-family coverage
Two layers. (1) Name lists applied to every model (`:159-236`); no per-`model_type` registry. (2) Explicit family rows, `FAMILY_ADDRESSES` `:1422-1512`, 12 entries: `OPTForCausalLM` (no MLP module), predicate `post_sublayer_norm` for `gemma2, gemma3, gemma3_text, olmo2` (`:1410`), `BloomForCausalLM`, `bloom_slow_but_exact`, `MptForCausalLM`, `DbrxForCausalLM`, `FalconForCausalLM`, Falcon+`alibi`, `GPTJForCausalLM`, `GptOssForCausalLM`. `get_block_structure` `:1515-1532` names `gptj, phi, codegen, olmo2` and config flags `use_parallel_residual, parallel_attn, new_decoder_architecture`. Hybrid Gated DeltaNet (Qwen3-Next / Qwen3.5 / Qwen3.6) via the `linear_attn` name plus 9 `linear_attn` rows.

Tested coverage: 26 pinned invariant families (`tests/test_config.yaml:66-127`: Llama, GPT-2, Mistral, Qwen2, Qwen3, Gemma, Gemma-2, Gemma-3, Phi-3, OLMo, OLMo-2, Phi, GPT-NeoX, GPT-J, Falcon-7b, Falcon-40b, StableLM-2, BLOOM, MPT, DBRX, OPT, Qwen3-MoE, Qwen2-MoE, GPT-OSS, DeepSeek-V2, DeepSeek-V3), 3 hybrids, 16 `llama_like_models`, 6 `core_test_models`, and the `yujiepan/tiny-dummy-models` collection (~200 repos in `data/toy_models_cache.json`) minus `skip_patterns`. Last full sweep: transformers 5.15.0.dev0 / nnsight 0.8.0 — 34 classes fully available across 71 checkpoints (`data/test_logs/latest_status.json`).

### Attention-probability source-path handling
- Default row: `self_attn` → `.source.attention_interface_1.source.nn_functional_dropout_0.output` — the **dropout output**, deliberately not the softmax output: "the pattern the values are mixed with on every family (after the cast, and after an attention sink has been dropped)" `rename_utils.py:1234-1242`. `attention_scores` is the softmax's **input** `:1230-1233`.
- `_1` not `_0`: nnsight 0.8 counts the `attention_interface = ...` binding and the call on one counter (`:993-996`; CHANGELOG `:214-217`). transformers 5 renamed GPT-2's `module_attn_dropout_0` → `nn_functional_dropout_0` (CHANGELOG `:203-211`). `tests/test_source_ops.py:1-15`: "nnterp has been broken that way twice."
- Per-arch overrides: BLOOM `self_attention_dropout_0` `:1444-1446`; MPT bare `nn_functional_dropout_0` `:1461`; DBRX module `self_attn.attn` `:1466-1482`; Falcon `F_softmax_0`, alibi branch `self_attention_dropout_0` `:1487-1494`; GPT-J `self__attn_0.source.self_attn_dropout_0` `:1497`; GPT-OSS `tags={"sink"}` (rows sum to `<1`), scores row one key wider `:1500-1511`.
- Validation: `check_attention_probabilities` (shape, sum, and a write must move the logits — "an address can read a perfectly good pattern and be causally inert" `:916-919`); at load only if `enable_attention_probs=True`.
- `LayerAccessor.get_operation` raises naming the row, class, transformers version and every op at that level `:812-819`; `print_source()` `:875-902`.

### Tests (`nnterp/tests/`, 12 files + conftest/utils)
- `test_model_renaming.py` (1067): renamed vs raw activations equal (`:85-142`), logits equal (`:145-172`), every accessor equals direct module access (`:175-258`), input accessors, steer, skip, constructor options, properties, `ModuleAccessor`, residual-inside contribution identity on 6 families (`:762-805`), residual-arg detection + `RenameConfig` overrides, `Index`/`FirstIfTuple`, custom `Selection`, unrenamed `mlp` fails at load, row-name clash, `logits` vs `lm_head_output` on Gemma-2 softcap, `disable`, `IOType.INPUTS` query row.
- `test_block_invariants.py` (259): on 26 pinned families, `block_structure`, `status()` before any trace equals the YAML, whole-model rows, every accessor reads a tensor with the row's `seq_axis` on every layer, residual identities `layers_mid == layers_input + attentions_output`, `layers_output == layers_mid + mlps_output`, `mlps_norm_output == mlps_input`, sizes vs tensors, layouts, writes land. Reads cloned as reached because Falcon adds attention into the MLP output in place (`:47-50`).
- `test_source_ops.py` (425): every op row on every layer; pattern == softmax(scores); interface rows shapes; DeltaNet rows.
- `test_hybrid_models.py` (218), `test_construction.py` (106), `test_probabilities.py` (72), `test_interventions.py` (289), `test_nnsight_utils.py` (189), `test_prompt_utils.py` (204), `test_vllm.py` (269, GPU), `test_vlm.py` (107), `test_detect_automodel.py` (65).
- `conftest.py` (566): `--model-names`, `--class-names`, `--save-test-logs`; per-model failure categories; xdist merge; writes `data/test_logs/*_status.json`.

### Data / tooling / docs
`data/status.json`, `test_loading_status.json`, `toy_models_cache.json`, `test_logs/`. `pyproject.toml` extras `display`, `llms`, `vllm`; `Makefile`; `.pre-commit-config.yaml` (black, `docs/llms.txt`); `scripts/` (run_tests, coverage, all-transformers-versions sweeps, smoke); workflows `docs.yml` (gh-pages), `claude.yml` — no workflow runs the tests. Docs: 12 rst pages, `llms.txt`, `CHANGELOG.md`, `CLAUDE.md`, `design/internals-addressing.md`, demo, two `.claude/skills`.

## 2. Gap list: in nnterp, not in nnter

**absorbs** = fits family modules + Envoy subclasses + eproperties as they are; **new** = needs machinery nnter doesn't have.

### (a) Core standardization

| item | what it does | nnter |
|---|---|---|
| Name lists for `mlp`, `ln_final` (7 spellings), `lm_head`, `embed_tokens`, containers (`decoder`, `language_model`) — `rename_utils.py:199-236` | one rename dict works on any model whose names are in the lists | absorbs, one `RENAME` per family; but nnter has no "unknown model_type, try the lists" path, so Mistral/Qwen/Gemma/Phi/OLMo/StableLM each need a module before they load at all |
| `linear_attn` convention + `attention_layers`/`linear_attention_layers` — `:205-208`, `:428-439` | hybrids expose one of `self_attn`/`linear_attn` per block | absorbs for the name; the per-block kind list is a model property (new but trivial) |
| Block accessors `layers_input`, `layers_mid`, `attentions_input/_norm_output/_premix/_output`, `mlps_input/_norm_output/_activation/_neurons/_output` — `:1198-1256`, `:1319-1373` | every tensor of the block by name, with residual identities | absorbs as eproperties on `Layer` / `Attention` / a new `Mlp` envoy; the structural candidate lookup becomes a per-family constant |
| Contribution semantics: `attentions_output`/`mlps_output` on residual-inside (BLOOM, MPT, DBRX), post-sublayer-norm (Gemma-2/3, OLMo-2), no-MLP (OPT) — `:1407-1483` | the accessor is the additive contribution, never a residual state | absorbs via family overrides — but nnter's vocabulary currently promises `self_attn.output` as the standard thing, which is the residual-added state on those families (§4) |
| Attention interior rows `attention_queries/_keys/_scores/_head_outputs` — `:1219-1248` | read off the interface call's `inputs` / softmax input / `output[0]` | absorbs as `SourceEProperty`s with `attribute="inputs"`; `SourceEProperty` has no `select` step (`Index(0,1)`) yet |
| Whole-model rows `embeddings_input/_output`, `ln_final_output`, `lm_head_output`, `logits` (softcapped on Gemma-2) — `:1203-1210` | one per model | absorbs as eproperties on `StandardizedTransformer` (the root envoy) |
| `token_embeddings`, `next_token_probs`, `input_ids`, `input_size`, `attention_mask` — `standardized_transformer.py:377-419` | | absorbs |
| Published sizes with per-family config-key rules — `rename_utils.py:279-409` | `head_dim` is not `hidden//heads` on Qwen3/Gemma | absorbs (properties; family may override the key) |
| `block_structure` — `:1293`, `:1515-1532` | decides which places exist | absorbs as a family constant |
| Availability before any trace: `Address.unavailable`, `unavailable_on`, `Internals.status(layer)`, `disable(reason)` | a missing place says why, at load, per layer | **new**: an eproperty only answers inside `__get__`; nnter raises `SourceNotAvailable` inside the trace |
| Forward-order `Address.order` + `Internals.rank` | sort several reads into nnsight's required order | **new** |
| Layout facts `seq_axis, width, heads, keys, needs` | cut a tensor by position/head without knowing the family | **new** (eproperty has only `key`/`description`) |
| Load-time validation: `check_model_renaming`, `check_io`, `_check_attention_layers`, `_check_output_source`, heterogeneous-layer refusal, `check_attention_probabilities` with the causal write; `try_with_scan` | "if it loads it probably works" | **new** (nnter runs nothing at construction) |
| `RenameConfig` as user-side extension; `addresses=` adds/replaces a row | | partial: `rename=`/`envoys=` kwargs exist, but `lookup` refuses an unregistered `model_type` first — needs a registration hook |
| Hybrid Gated DeltaNet rows, `delta_rule_call`, `pin_linear_attention_kernels`, `delta_rule_scan`, `dispatch()` re-pin | | **new**: op chosen by decode step and globals pinned before `.source` are outside `SourceEProperty`'s shape |
| `skip_layer`/`skip_layers` with `DummyCache` | | absorbs (`Layer.skip_with(tensor)` rewrapping the tuple) |
| `steer(...)` incl. vLLM clone path | | absorbs |
| `project_on_vocab`, `get_topk_closest_tokens`, `probs_to_dict` | | absorbs |
| `add_prefix_false_tokenizer`, `tokenizer_kwargs` | | absorbs |
| `remote=True`: no dispatch, scan-only checks, server has the package — CHANGELOG `:160-178` | | **new**; nnter's README proposes by-value shipping via `nnsight.register`, which nnterp abandoned (RLock pickling, size) |
| `detect_automodel`, `text_only`, `StandardizedVLM`, `load_model` | VLMs and text towers | **new** (nnter hardcodes `task="text-generation"`) |
| `device_map="auto"` default | | trivial |

### (b) Conveniences

| item | where | nnter |
|---|---|---|
| Activation collection helpers | `nnsight_utils.py:260-450` | absorbs as a module against `model.layers[i].layer_output` |
| Raw-model helpers and `ModuleAccessor` | `nnsight_utils.py:19-256` | absorbs; needs the family lookup exposed as a function |
| `logit_lens`, `patchscope_lens`, `patchscope_generate`, `patch_object_attn_lens`, `TargetPrompt(Batch)`, `repeat_prompt`, `it_repeat_prompt` | `interventions.py` | absorbs |
| `get_first_tokens`, `Prompt`, `run_prompts` | `prompt_utils.py` | absorbs |
| `plot_topk_tokens`, `prompts_to_df` | `display.py` | absorbs |
| `StandardizedVLLM` | `standardized_vllm.py` | **new** class; vLLM's tree has no attention interface and computes logits outside the forward |

### (c) Tests / tooling / CI / docs

| item | where | nnter |
|---|---|---|
| Pinned per-family invariant suite with residual identities and layout checks | `test_block_invariants.py`, `test_config.yaml:66-127` | new; nnter's `CHECKPOINTS` table is the seed |
| Source-op regression suite across families and transformers versions | `test_source_ops.py` | new |
| Renamed-vs-raw equivalence tests | `test_model_renaming.py:85-172` | new; cheap |
| Hybrid, construction, VLM, vLLM, detect_automodel, gradient-shape tests | various | new |
| Whole-collection sweep + per-version status JSON, xdist-safe; `python -m nnterp run_tests` | `conftest.py`, `tests/utils.py`, `__main__.py`, `data/` | new |
| Cross-transformers-version scripts | `scripts/` | new |
| pre-commit, docs site + gh-pages workflow, `CHANGELOG.md`, `CLAUDE.md`, design doc, skills | | new; nnter has README only |
| Test CI | none in nnterp either | — |

## 3. Deliberate differences

1. **Accessor spelling.** nnterp: `model.layers_output[i]`, a `LayerAccessor` on the model per row. nnter: `model.layers[i].layer_output`, an eproperty on the block envoy. nnterp's answer `status`/`rank`/`width` outside a trace and pickle by value; nnter's follow nnsight's idiom, appear in the repr, and go through the interleaver like `.output`, but exist only inside a trace.
2. **Root vs `model.model`.** nnterp renames every container to `model` (`model.model.layers[i]`); nnter lifts `layers`/`norm`/`embed_tokens` to the root. nnterp's tree matches raw HF shape; nnter's is shorter but a VLM's `language_model` would need the same lift.
3. **Final-norm name.** nnterp `ln_final`; nnter keeps Llama's `norm`.
4. **Configuration model.** nnterp: global name lists + isinstance/predicate ladder + `RenameConfig`. nnter: a module per family by `model_type`, `UnsupportedFamily` otherwise. nnterp loads unknown Llama-likes with no code and needs `check_model_renaming` because a list can mis-bind; nnter never mis-binds silently but needs a module even for identical families.
5. **Eager by default vs opt-in.** nnterp keeps HF's default and `enable_attention_probs=True` forces eager; nnter defaults eager for every model (always works; always pays eager's cost).
6. **Which op the pattern reads.** nnterp: dropout output; nnter: softmax output (§4.1).
7. **When availability is answered.** nnterp at load per layer with a reason; nnter at first read inside the trace.
8. **Validation at load.** nnterp runs scan/trace checks on construction; nnter runs none.
9. **Tuple outputs.** nnterp `FirstIfTuple` selection + `returns_tuple(layer)`; nnter `Layer.layer_output` pre/postprocess. Same semantics, different home.
10. **Remote.** nnterp: nothing shipped, server has nnterp installed; nnter README: `nnsight.register(nnter)` by value. nnterp left that after `TypeError: cannot pickle '_thread.RLock'` on the module logger.
11. **Sizes.** nnterp publishes `num_layers/num_heads/...` with per-family key rules; nnter reads `model.config.*`.

## 4. Fragile spots in nnter, from nnterp's experience

1. **Softmax output instead of dropout output** (`base.py`). nnterp reads the dropout output: "the pattern the values are mixed with on every family (after the cast, and after an attention sink has been dropped)" (`rename_utils.py:1234-1242`). GPT-2's eager does `softmax` → `.type(value.dtype)` → `dropout`; Llama's `softmax(..., dtype=float32).to(query.dtype)`, so in a bf16 model nnter's value is the fp32 pre-cast tensor. On GPT-OSS the softmax spans keys **plus the sink** and the sink column is dropped afterwards; nnterp tags that row `sink` and checks rows sum to `<1`. nnter's shape and row-sum tests would both fail on GPT-OSS, and a read would carry `[b, h, q, k+1]` under a name documented as `[batch, heads, query, key]`.
2. **One hard-coded op string, no regression suite.** nnterp records the interface naming change under nnsight 0.8 (`_0` → `_1`) and the dropout rename under transformers 5; `tests/test_source_ops.py:1-15` exists because "nnterp has been broken that way twice". BLOOM, MPT, Falcon, GPT-J, DBRX all bypass the interface (`rename_utils.py:1439-1499`), so "Llama-named families need only the three container keys" is true for names, not for the op.
3. **`post_attention_layernorm` as a standard name.** nnterp keys the pre-MLP norm by block structure because the name lies: on Gemma-2/3 `post_attention_layernorm` is the attention's post-norm and the pre-MLP norm is `pre_feedforward_layernorm` (`:1296-1300`, `:1336-1348`); on GPT-NeoX (parallel) it normalizes the block input, not a mid-stream (`:1357-1360`). `test_block_invariants.py:12-15`: a wrong norm "reads a real tensor of the right shape from the wrong place, so nothing else would notice". nnter's gpt_neox family already inherits this.
4. **`self_attn.output` / `mlp.output` as standard values.** On BLOOM/MPT/DBRX the module adds the residual inside, so that is a residual-stream state (`:1417-1483`); on Gemma-2/3/OLMo-2 the tensor actually added is the post-sublayer norm's output; OPT has no `mlp`. nnterp refuses an unknown family whose sublayer forward takes a `residual` argument (`:1831-1889`). nnter has no guard and no `attention_output`/`mlp_output` eproperty to hang the override on yet.
5. **Module-level import and use of `Mediator.value`** (`base.py`). nnterp imports `Mediator` lazily: "nnsight's step pointer and not public API (a public accessor has been asked of nnsight)" (`rename_utils.py:1031-1034`). Likewise `HuggingFaceModel._config` is private and nnter reproduces its JSON key by hand; nnterp uses `AutoConfig.from_pretrained`. `tests/test_standardized.py` asserts on `model._aliases`.
6. **`.source` snapshots module globals at first drill.** nnsight builds the instrumented forward over a copy of the globals seen when first asked (`rename_utils.py:1052-1058`). nnter's `_drill` runs on every read, so the first access pins whatever kernel globals were bound; families with runtime kernel switches (Gated DeltaNet, `fla`/`causal_conv1d`) need a pin-before-source step nnter has no place for.
7. **Reads are not cloned.** nnterp clones each value as reached because Falcon adds the attention output into the MLP output in place (`tests/test_block_invariants.py:47-50`), and uses `Copied` for the DeltaNet buffer. nnter's `layer_output.save()` returns the live object: fine for the block output, a trap for an `attention_output` eproperty on Falcon.
8. **In-place per-head writes on GPT-2.** `attention_interface` output[0] on GPT-2 is a non-contiguous transpose view and torch refuses in-place edits, while Llama's ends with `.contiguous()` (`design/internals-addressing.md:333-339`). Safe for the softmax output, not for a head-output `SourceEProperty`.
9. **Class-keyed `ENVOYS` and heterogeneous layers.** Mllama's cross-attention blocks are a different class: plain `Envoy`s with no `layer_output`, and `model.layers[i].self_attn` would raise a bare `AttributeError`. nnterp refuses heterogeneous layer types unless `allow_multimodal=True` (`rename_utils.py:1902-1912`).
10. **No causal check at load.** nnterp validates the pattern by writing it: "an address can read a perfectly good pattern and be causally inert (the weights a mixer *returns* are one)" (`rename_utils.py:916-919`). nnter checks this in tests only.
11. **GPT-2 `reorder_and_upcast_attn`.** Documented as unreachable but not detected. nnterp does not detect it either.
