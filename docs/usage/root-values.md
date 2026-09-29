---
title: Root values and sizes
one_liner: "The model answers for the whole run — `logits`, `token_embeddings`, `next_token_probs`, `input_ids`, `attention_mask`, `input_size` — and for its sizes from the config."
tags: [usage, logits, token_embeddings, next_token_probs, input_ids, sizes, config]
related: [docs/usage/residual-stream.md, docs/usage/methods.md, docs/usage/layouts.md, docs/usage/availability.md]
sources: [nnter/standardized.py, nnter/components/eproperty.py]
---

# Root values and sizes

## What this is for

Inside a trace the root envoy carries the values that belong to the whole model rather
than to one block: the final logits, the embeddings entering block 0, the next-token
distribution, and the ids and mask the model was called with. Outside a trace it answers
for the sizes an experiment needs (`num_layers`, `hidden_size`, `num_heads`, ...), read off
the config with the fallbacks older configs need. Both are the same on every family.

## Canonical pattern

```python
from nnter import StandardizedTransformer

model = StandardizedTransformer("google/gemma-2-2b", dispatch=True)

with model.trace("The Eiffel Tower is in"):
    ids = model.input_ids.save()               # [batch, seq]
    emb = model.token_embeddings.save()        # [batch, seq, hidden]
    raw = model.lm_head.output.save()          # the raw projection
    logits = model.logits.save()               # [batch, seq, vocab], softcapping applied
    probs = model.next_token_probs.save()      # [batch, vocab]

model.num_layers, model.hidden_size, model.vocab_size, model.num_heads, model.num_kv_heads
```

Read them in forward order: `input_ids` / `attention_mask` / `input_size` first (they are
the model's input), `token_embeddings` next, `lm_head.output`, `logits` and
`next_token_probs` last.

## `logits`

The `.logits` of the model's output, `[batch, seq, vocab]`. On a family that softcaps after
`lm_head` (Gemma-2, `config.final_logit_softcapping`) it differs from `lm_head.output`:

```python
cap = model.config.final_logit_softcapping        # 30.0 on Gemma-2
torch.equal(logits, raw)                          # False on Gemma-2, True on GPT-2
torch.allclose(logits, cap * torch.tanh(raw / cap))   # True
```

Assigning replaces the logits in the model's output, so `tracer.result.logits` is what you
set:

```python
with model.trace(prompt) as tracer:
    model.logits = model.logits * 0
    result = tracer.result.logits.save()          # zeros
```

## `token_embeddings`

The embedding module's output, `[batch, seq, hidden]`: `model.embed_tokens.output` under
its standard name, before anything the family applies afterwards (GPT-2's positional
`wpe`, BLOOM's `word_embeddings_layernorm`, Gemma's scaling). Assign to replace what enters
the first block:

```python
with model.trace(prompt):
    model.token_embeddings = model.token_embeddings * 0
    logits = model.logits.save()                  # changed
```

## `next_token_probs`

`logits[:, -1].softmax(-1)`, `[batch, vocab]`, derived from the output. Read-only: there is
no inverse, and assigning raises

```
AttributeError: next_token_probs is derived from the logits and cannot be assigned; assign model.logits instead
```

Position `-1` is the last token of every row only under left padding. With the default
right padding a shorter prompt's last position is a pad token:

```python
model = StandardizedTransformer("openai-community/gpt2", dispatch=True,
                                tokenizer_kwargs={"padding_side": "left", "pad_token": "<|endoftext|>"})
with model.trace(["Hi", "The Eiffel Tower is in"]):
    probs = model.next_token_probs.save()         # row 0 is the distribution after "Hi"
```

`nnter.nnsight_utils.compute_next_token_probs(model, prompts)` is this read over a list of
prompts.

## `input_ids`, `attention_mask`, `input_size`

What the model was called with, `[batch, seq]` each; `input_size` is the ids' `torch.Size`.
`input_ids` and `attention_mask` are assignable, and the model then runs on what you set:

```python
with model.trace("Paris is the capital of"):
    other_ids = model.input_ids.save()
    other_logits = model.logits.save()

with model.trace("The Eiffel Tower is in"):
    model.input_ids = other_ids.clone()
    model.attention_mask = torch.ones_like(other_ids)
    logits = model.logits.save()

torch.allclose(logits, other_logits)              # True: the second trace ran on the first prompt's ids
```

`input_size` is read-only (`AttributeError: input_size is the ids' shape and cannot be
assigned; assign input_ids`). A `torch.Size` bound inside the block does not survive the
trace; save `torch.tensor(model.input_size)` if you need it outside.

The three print with the model:

```
  (logits): The model's final logits, [batch, seq, vocab], softcapping applied
  (token_embeddings): The token embeddings entering the first block, [batch, seq, hidden]
  (next_token_probs): The next-token distribution at the last position, [batch, vocab]; derived, read-only
  (input_ids): The token ids the model was called with, [batch, seq]
  (attention_mask): The attention mask the model was called with, [batch, seq]; zeros are padding
  (input_size): [batch, seq] of the current call; read-only
```

## Sizes

Plain properties, readable before any trace and without `dispatch`:

| size | what it reads |
| --- | --- |
| `num_layers` | `len(model.layers)` |
| `hidden_size` | `config.hidden_size` |
| `vocab_size` | `config.vocab_size` |
| `num_heads` | `config.num_attention_heads` |
| `num_kv_heads` | `config.num_key_value_heads`; else `config.num_kv_heads` on Falcon's 40B layout (`new_decoder_architecture`); else `1` under `multi_query`, else `num_heads` |
| `head_dim` | `config.v_head_dim` under multi-head latent attention (DeepSeek); else `config.head_dim` when the config says (Qwen3, Gemma); else `hidden_size // num_heads` |
| `qk_head_dim` | `qk_nope_head_dim + qk_rope_head_dim` under latent attention; else `head_dim` |
| `intermediate_size` | `config.n_inner` on a GPT-2-style config (`None` meaning `4 * hidden_size`; its `intermediate_size` is never read by the model); else `config.intermediate_size` or `config.ffn_dim` (OPT); else `hidden_size * expansion_ratio` (MPT) or `4 * hidden_size` (BLOOM, Falcon) |

`head_dim` is the config's on Qwen3 and Gemma, not `hidden // heads`: a Qwen3 checkpoint
with `hidden_size=8`, `num_heads=4` and `head_dim=128` has 128-wide heads. On DeepSeek-V3
`head_dim` (values, 128) and `qk_head_dim` (queries and keys, 192) differ; see
[layouts](layouts.md). A mixture of experts' experts are `config.moe_intermediate_size`
wide, not `intermediate_size`.

## Gotchas

- **`logits` is not `lm_head.output` on Gemma-2.** Use `logits` for the model's prediction
  and `lm_head.output` only when you want the raw projection.
- **`next_token_probs` and `input_size` are read-only.** Assign `logits` or `input_ids`.
- **`next_token_probs` assumes the last position is the last token.** Left-pad a batch.
- **Forward order.** `input_ids` and `token_embeddings` come before any block's value in
  the same trace; `logits` and `next_token_probs` after them all.
- **Assigning `input_ids` does not resize the mask.** Assign `attention_mask` to match when
  the new ids have another length.
- **`intermediate_size` is the dense MLP's width.** For an all-MoE family read
  `config.moe_intermediate_size`.

## Related

- [residual-stream](residual-stream.md): the per-block values.
- [methods](methods.md): `project_on_vocab`, which reproduces `logits` from a block's stream.
- [layouts](layouts.md): the axes of each value against these sizes.
- [availability](availability.md): the root values in `status()`.
