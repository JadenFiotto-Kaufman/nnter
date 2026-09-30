# nnter — Agent Guide

This file routes you to the right page under `docs/` for whatever the user is asking about. The
content lives in `docs/`; **read the matching page before writing code**. The pages are
recipe-style and every snippet in them has been run against the pinned checkpoints in `tests/`.

nnter is a thin layer on nnsight 0.8: `StandardizedTransformer` is an nnsight `TransformersModel`
whose envoy tree answers to one set of names on every transformer family, with standard values
(`layer_output`, `attention_output`, `attention_probabilities`, ...) that mean the same thing
everywhere. Everything nnsight does (`trace`, `generate`, `.save()`, `tracer.iter`, invokes,
`.source`, remote) works unchanged; nnsight's own guide is `~/wd/nnsight/CLAUDE.md` and its docs
`~/wd/nnsight/docs/`. This file covers only what nnter adds.

---

## How to use this file

1. Find the user's intent in **"By task"** and follow the link.
2. If the request is about one model family, check **[docs/reference/families.md](docs/reference/families.md)** for its quirks.
3. If a value is missing or raises `Unavailable`, read **[docs/usage/availability.md](docs/usage/availability.md)**.
4. The **inline cheat-sheet** at the bottom lists the mistakes agents make most; internalize it before writing nnter code.

---

## By task

### "Load a model and use the standard names"
- [docs/usage/loading.md](docs/usage/loading.md) — `StandardizedTransformer(repo_id, ...)`; pass `attn_implementation="eager"` for anything inside attention
- [docs/usage/vocabulary.md](docs/usage/vocabulary.md) — `embed_tokens`, `layers[i].self_attn`, `layers[i].mlp`, `norm`, `lm_head`; native names keep working
- [docs/reference/families.md](docs/reference/families.md) — the 50 families, their native names and quirks

### "Read or edit the residual stream / a sublayer's contribution"
- [docs/usage/residual-stream.md](docs/usage/residual-stream.md) — `layer_output`, `attention_output`, `mlp_output`; `input + attention_output + mlp_output == layer_output`

### "Read or edit attention: the pattern, queries, keys, values, scores, heads"
- [docs/usage/attention-interior.md](docs/usage/attention-interior.md) — `attention_probabilities`, `attention_queries/keys/values/scores/head_outputs`; needs eager; per-family caveats
- [docs/patterns/attention-patterns.md](docs/patterns/attention-patterns.md) — head metrics and pattern edits

### "Logits, embeddings, next-token probabilities, the input, the sizes"
- [docs/usage/root-values.md](docs/usage/root-values.md) — `logits`, `token_embeddings`, `next_token_probs`, `input_ids`, `attention_mask`, `input_size`, `num_layers`, `head_dim`, ... (each size a `StandardizedProperty`: the plain rule, or the family's spelling)

### "Does this checkpoint have that value?"
- [docs/usage/availability.md](docs/usage/availability.md) — `model.status()` before the trace; `nnter.Unavailable` at the read; the reasons you will see

### "Skip layers, steer, logit lens, top-k tokens"
- [docs/usage/methods.md](docs/usage/methods.md) — `skip_layers`, `steer`, `project_on_vocab`, `get_topk_closest_tokens`
- [docs/patterns/logit-lens.md](docs/patterns/logit-lens.md), [docs/patterns/steering.md](docs/patterns/steering.md)

### "Qwen3-Next / Qwen3.5 / OLMo-Hybrid: linear attention, the recurrent state"
- [docs/usage/delta-net.md](docs/usage/delta-net.md) — `linear_attn` values; `route_kernels(model.family, "torch")` for the per-token `state`/`states`
- [docs/patterns/delta-net-state.md](docs/patterns/delta-net-state.md) — patch and track the state

### "Mamba / Falcon-Mamba / Jamba: the selective scan, the state-space state"
- [docs/usage/selective-scan.md](docs/usage/selective-scan.md) — `linear_attn` on a Mamba-1 mixer (`SelectiveScan`): `C`/`B`/`x` as queries/keys/values, `betas` = `dt`, `decays` = `dt * A`; `route_kernels(model.family, "torch")` before the first trace

### "Mamba-2 / Nemotron-H / Bamba / Falcon-H1: the state-space mixer"
- [docs/usage/state-space.md](docs/usage/state-space.md) — `linear_attn` is a `StateSpace`: `C`/`B`/`x` as queries/keys/values, `dt` as `betas`; `route_kernels(model.family, "torch")` when `mamba_ssm` is installed; `nnter.chunk_per_token(model)` for the state after every token (`states`, `state_after`); `betas`/`decays` assignable

### "What shape is this value?"
- [docs/usage/layouts.md](docs/usage/layouts.md) — one layout per value on every family, named (`Residual`, `Pattern`, `Keys`, ... in `nnter.components`); `value.dims`, `value.layout is Pattern`

### "Generation, many prompts, activations datasets"
- [docs/usage/generation.md](docs/usage/generation.md) — the values under `model.generate`, `tracer.iter` picks the step
- [docs/usage/prompt-utils.md](docs/usage/prompt-utils.md) — `nnter.prompt_utils`: target-token mass over prompts
- [docs/usage/activations.md](docs/usage/activations.md) — `nnter.nnsight_utils`: `get_token_activations` and friends

### "Run a research pattern across families"
- [docs/patterns/index.md](docs/patterns/index.md) — logit lens, steering, attention patterns, ablation, activation patching, contribution decomposition, cross-family sweep, probing, DeltaNet state

### "Run remotely on NDIF"
- [docs/usage/remote.md](docs/usage/remote.md) — `remote=True`; nnter installed server-side, never shipped by value

### "Add a family, override a value, add my own value"
- [docs/extending/adding-a-family.md](docs/extending/adding-a-family.md) — one module named after `model_type`, one test file; `def <size>(model)` in the module where the config spells a root size its own way
- [docs/developing/recurrent-mixer-internals.md](docs/developing/recurrent-mixer-internals.md) — a mixer with a recurrent state read at a kernel call (DeltaNet, state-space): subclass `RecurrentMixer`, set `CHUNK_KERNEL` / `RECURRENT_KERNEL` / `STATE_OP`, declare the values
- [docs/extending/overriding-values.md](docs/extending/overriding-values.md) — an `EProperty` keyed on a path (`"../norm.output"`, `"source.<op>.inputs"` with `select`), `unavailable(...)`, `off_interface`, transforms
- [docs/extending/custom-values.md](docs/extending/custom-values.md) — a new `EProperty` (a path from the host: `"output"`, `"../ln_2.output"`, `"source.<op>.output"`) through `envoys=`; annotate `-> Residual` / `-> Pattern` from `nnter.components`
- [docs/extending/finding-source-ops.md](docs/extending/finding-source-ops.md) — `print(envoy.source)` and how ops are named
- [docs/extending/registering.md](docs/extending/registering.md) — `nnter.families.register(module)`

### "Change nnter itself"
- [docs/developing/index.md](docs/developing/index.md) — architecture, descriptor internals, the recurrent mixer (`RecurrentMixer`, DeltaNet) and its occurrence arithmetic, tests, transformers compatibility, gotchas, contributing
- **Run `HF_HUB_OFFLINE=1 pytest` (1200 tests, ~70 s) before and after.**

### "Every symbol / every term"
- [docs/reference/api-quick-reference.md](docs/reference/api-quick-reference.md), [docs/reference/glossary.md](docs/reference/glossary.md)

---

## Folders

| Folder | What it holds | Start at |
|---|---|---|
| `docs/usage/` | one page per feature: loading, names, every standard value, methods, hybrids, helpers, remote | [docs/usage/index.md](docs/usage/index.md) |
| `docs/patterns/` | interpretability recipes written once against the standard values, so they run on every family | [docs/patterns/index.md](docs/patterns/index.md) |
| `docs/extending/` | adding a family, overriding a value, adding your own values, registering from outside nnter | [docs/extending/index.md](docs/extending/index.md) |
| `docs/developing/` | internals: architecture, the descriptors, the recurrent mixer and its occurrence arithmetic, tests, compatibility, gotchas | [docs/developing/index.md](docs/developing/index.md) |
| `docs/reference/` | API quick reference, the families table, glossary | [docs/reference/api-quick-reference.md](docs/reference/api-quick-reference.md) |

---

## Inline cheat-sheet (read before writing nnter code)

- **Everything nnsight's cheat-sheet says still holds**: `.save()` and bind the name, reads in forward order within an invoke, nothing assigned in a trace body survives it without a save.
- **Pass `attn_implementation="eager"` at load** if you will touch anything inside attention (`attention_probabilities`, queries, keys, values, scores, head outputs). The default is the checkpoint's, usually `sdpa`, and the values are then unavailable.
- **Check `model.status()` outside the trace, not `hasattr` inside it.** `hasattr(envoy, "attention_probabilities")` never answers `False`: it raises `nnter.Unavailable` when the value is unavailable, and outside a trace raises nnsight's "Cannot access ... outside of interleaving" for an available one.
- **`layer_output`, `attention_output`, `mlp_output` are tensors on every family**; never index `[0]`. The native `.output` may be a tuple (GPT-J, BLOOM, MPT, Falcon).
- **`attention_output` is what the block adds to the stream**, not necessarily the module's return: on Gemma-2/3, OLMo-2/3 and OLMo-Hybrid's attention blocks it is the post-norm's output, on BLOOM/MPT/DBRX the pre-residual value. The identity `layers[i].input + attention_output + mlp_output == layer_output` is what you can rely on.
- **`model.logits` is the model's output logits (softcap applied); `lm_head.output` is the raw projection.** `next_token_probs`, `input_size` and `states` are read-only.
- **Decide which blocks have `self_attn` vs `linear_attn` outside the trace** on a hybrid; `getattr(envoy, name, None)` inside a trace can trip served values, and `if envoy:` falls through to the module's `__len__`.
- **Read order traps**: on Falcon without alibi read `attention_values` before `attention_queries`/`attention_keys` (with alibi: queries, then keys, then values); on DeltaNet read `states` before any state write; `skip_layers` consumes `layers[start].input`, so read it first; a block's interior values come before its `attention_output`.
- **An out-of-order read of a source-located value does not raise**: the block is cut short with a `UserWarning` and later names are unbound. If a saved name is missing, look for that warning.
- **GPT-2 and MPT queries/keys/values are split views**: assign, do not edit in place. Falcon's `mlp_output` is a copy carried back by a transform; both in-place and assignment reach the model.
- **DeltaNet per-token state needs `nnter.route_kernels(model.family, "torch")` (or `route_delta_rule(model.family, "recurrent")`) before the first trace of a linear block**; write the state by assignment from a tensor you already hold (a token's `state` is served once, so reading it and then assigning it in the same `tracer.iter` step cuts the block), never in place.
- **Mamba-1 families (Mamba, Falcon-Mamba, Jamba) need `nnter.route_kernels(model.family, "torch")` before the first trace** when `mamba_ssm` is installed: its CUDA kernels have no source and do not run on CPU. In a decode step the state (`state_output`) comes before `attention_head_outputs`; in the prompt's scan, after.
- **`envoys=` keys match by module type or native path, never by alias**; to displace a family's envoy, key yours on the type.
- **Import nnter (or nnsight) before any `transformers.models...` module**; the reverse order segfaults on this stack.
- **Every snippet in `docs/` ran against the tiny checkpoints in `tests/families/`**; when a page and the code disagree, the suite is the arbiter: `HF_HUB_OFFLINE=1 pytest tests/families/test_<family>.py`.
