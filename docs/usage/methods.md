---
title: Methods over the values
one_liner: "`skip_layers`, `steer`, `project_on_vocab`, `get_topk_closest_tokens` and `probs_to_dict`: the common operations written once against the standard values."
tags: [usage, skip_layers, steer, project_on_vocab, logit-lens, topk]
related: [docs/usage/residual-stream.md, docs/usage/root-values.md, docs/usage/vocabulary.md]
sources: [nnter/standardized.py, nnter/components/layer.py]
---

# Methods over the values

## What this is for

Five methods on `StandardizedTransformer` do the things an experiment does over and over,
written against the standard names so they run unchanged on every family: skip a range of
blocks, add a vector to the residual stream, decode a hidden state through the model's own
head (the logit lens), and turn a distribution into readable tokens. Three run inside a
trace; two work on saved tensors outside.

| method | where it runs | what it does |
| --- | --- | --- |
| `skip_layers(start, end, skip_with=None)` | inside a trace | blocks `start..end` inclusive do not run |
| `steer(layers, vector, factor=1.0, token_positions=None, batch_index=None)` | inside a trace | adds `factor * vector` to `layer_output` in place |
| `project_on_vocab(hidden)` | inside on a live value, or outside on a saved one | final norm, `lm_head`, softcapping |
| `get_topk_closest_tokens(hidden, k=5)` | either | `project_on_vocab` then softmax then top-k per position |
| `probs_to_dict(probs, k=5)` | either | one `[vocab]` distribution to `{token: probability}` |

## Canonical pattern

```python
import torch
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in"

with model.trace(prompt):
    model.skip_layers(4, 7)                                        # blocks 4..7 do not run
    model.steer(10, direction, factor=3.0, token_positions=-1)     # add to the stream leaving block 10
    lens = model.project_on_vocab(model.layers[5].layer_output).save()   # logit lens at block 5
    logits = model.logits.save()

model.get_topk_closest_tokens(lens[0, -1], k=5)   # [{token: probability, ...}] for that position
```

## `skip_layers(start, end, skip_with=None)`

The residual stream entering block `start` (or `skip_with`) is handed straight to block
`end + 1`; the skipped blocks do not run. Indices are inclusive, and negative ones count
from the end:

```python
with model.trace(prompt):
    before = model.layers[1].layer_output.save()
    model.skip_layers(2, 3)
    after = model.layers[3].layer_output.save()     # equals `before`
    logits = model.logits.save()

with model.trace(prompt):
    model.skip_layers(-2, -1)                        # the last two blocks
    logits = model.logits.save()                     # == project_on_vocab(layers[-3].layer_output)
```

`skip_with` replaces the stream entirely: `model.skip_layers(0, 0, skip_with=torch.zeros_like(x))`
makes block 1 start from zeros.

Each skipped block gets `Layer.skip_with(hidden)`, which calls nnsight's `.skip()` with
`hidden` packed the way that family's block returns it: the tensor alone, or `(hidden, None)`
on a family whose `Layer.returns_tuple` is `True` (GPT-J, BLOOM, MPT, Falcon), since the
second element is the attention weights nothing downstream reads. Call `skip_with` directly
to skip one block with a stream of your own:

```python
with model.trace(prompt):
    model.layers[0].skip_with(model.layers[0].input)   # a pass-through
```

Skipping needs the block's `input` to be read, so `skip_layers` reads it; a second read of
`layers[start].input` after the call is out of order. Read `layers[start - 1].layer_output`
before the call instead, as above.

## `steer(layers, vector, factor=1.0, token_positions=None, batch_index=None)`

Adds `factor * vector` to `layer_output` of each block in `layers`, in place, so it reaches
the model. `vector` is `[hidden]` or anything broadcastable to the selected slice;
`token_positions` (an int, a list, a slice) restricts the positions and `batch_index` the
row; both default to all. Given a list, `layers` must be ascending: the adds happen in
forward order.

```python
torch.manual_seed(0)
direction = torch.randn(model.hidden_size)

with model.trace(prompt):
    model.steer(1, direction, factor=3.0, token_positions=-1)
    out = model.layers[1].layer_output.save()
    logits = model.logits.save()
# out[:, -1] == clean[:, -1] + 3 * direction; out[:, :-1] unchanged

with model.trace(prompt):
    model.steer([0, 2], direction, factor=1.0, token_positions=[0, 2], batch_index=0)
```

`vector` is moved to the stream's dtype and device (`vector.to(out)`), so a CPU float32
direction works on a bf16 GPU model. Scale the factor against the stream's norm at that
block; nnsight docs/patterns/steering.md is the pattern.

## `project_on_vocab(hidden)`

Logits for a residual-stream tensor: `lm_head(norm(hidden))`, then the model's softcapping
when `config.final_logit_softcapping` is set (Gemma-2). Applied to the last block's
`layer_output` it equals `logits` exactly:

```python
with model.trace(prompt):
    last = model.layers[-1].layer_output.save()
    logits = model.logits.save()

torch.equal(model.project_on_vocab(last), logits)    # True (max abs diff 0.0 on GPT-2, BLOOM, GPT-NeoX)
```

Inside a trace it takes a live value, so a logit lens over every block is a loop:

```python
with model.trace(prompt):
    lens = [model.project_on_vocab(layer.layer_output)[0, -1].save() for layer in model.layers]
```

Outside a trace it runs the norm and head modules as plain `nn.Module`s on a saved tensor.

## `get_topk_closest_tokens(hidden, k=5)` and `probs_to_dict(probs, k=5)`

`get_topk_closest_tokens` is `project_on_vocab`, softmax, then the `k` most likely tokens
per position, one `{token: probability}` dict per row of `hidden.reshape(-1, hidden_size)`:

```python
model.get_topk_closest_tokens(last[0, -1], k=3)   # one position: a list of one dict
model.get_topk_closest_tokens(last[0], k=3)       # every position: seq dicts
```

`probs_to_dict` does the last step alone on a `[vocab]` distribution:

```python
with model.trace(prompt):
    probs = model.next_token_probs.save()
model.probs_to_dict(probs[0], k=5)
```

Both call `.item()` and `tokenizer.decode`, so their result is plain Python. Inside a
trace they run on live values too, but the list they return has to be kept with
`nnsight.save(...)` like any other Python object bound in the block. Calling them outside,
on saved tensors, is the plain form.

## Gotchas

- **`skip_layers` and `steer` only work inside an active trace**, before the blocks they
  touch have run. `project_on_vocab` and the top-k helpers work in both places.
- **Read `layers[start].input` before `skip_layers`, or not at all.** The method reads it;
  a later read of the same location is out of order.
- **A skipped block's inner modules never run**: nothing under `layers[i].self_attn` or
  `layers[i].mlp` is readable for a skipped `i` (nnsight docs/usage/skip.md).
- **`steer` layers must be ascending**, and the add happens on `layer_output`: to steer
  the stream *entering* block `i`, steer block `i - 1`.
- **`project_on_vocab` uses the model's `norm` and `lm_head`**, so on a checkpoint whose
  head is tied or softcapped it is still exact; a hand-rolled `lm_head(hidden)` without the
  norm and cap is not the model's prediction.
- **Top-k on a tiny random checkpoint is noise**; the shapes are what to check there.

## Related

- [residual-stream](residual-stream.md): `layer_output`, what these methods move.
- [root-values](root-values.md): `logits`, `next_token_probs`, the sizes.
- nnsight docs/usage/skip.md: what `.skip()` replaces and why the shape must match.
- nnsight docs/patterns/steering.md and docs/patterns/logit-lens.md: the experiments.
