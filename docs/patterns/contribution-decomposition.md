---
title: Contribution Decomposition
one_liner: "Direct logit attribution over the standard contributions: `model.lm_head(attention_output)` and `model.lm_head(mlp_output)` per block sum to the final projection, `project_on_vocab` gives the normed lens on each, and `attention_head_outputs` with the output projection's weight splits attention per head."
tags: [patterns, attribution, residual-stream, heads, logit-lens]
related: [docs/usage/residual-stream.md, docs/usage/root-values.md, docs/patterns/logit-lens.md, docs/patterns/ablation.md, docs/patterns/attention-patterns.md]
sources: [nnter/standardized.py, nnter/components/layer.py, nnter/components/attention.py, nnter/components/mlp.py, nnter/families/gpt2.py]
---

# Contribution Decomposition

## What this is for

The residual stream is a sum: the block input, plus what each attention sublayer
adds, plus what each MLP adds. Direct logit attribution (DLA) pushes each term
through the unembedding on its own and asks how much it moved a target logit.

nnter defines the terms so the sum holds on every family:
`layers[i].input + attention_output + mlp_output == layer_output`, checked per family
by the test suite, whether the block is sequential, parallel, sandwich-normed or a
DeltaNet hybrid. This page verifies the sum end to end, then decomposes it: by
block, by sublayer and by head. That identity is the one piece of normalization it
rests on.

## Canonical pattern

Read the base and every contribution in forward order; their running sum is the
last block's output:

```python
import torch
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer.encode(" Paris")[0]

parts = {}                                          # made outside the trace
with model.trace(prompt):
    base = model.layers[0].input.save()             # the stream entering block 0
    for i, layer in enumerate(model.layers):        # block i's attention, then its MLP: forward order
        parts["attn", i] = layer.self_attn.attention_output.save()
        parts["mlp", i] = layer.mlp.mlp_output.save()
    final = model.layers[-1].layer_output.save()

total = base + sum(parts.values())
torch.testing.assert_close(total, final, rtol=1e-4, atol=1e-3)   # the stream is the sum of its contributions
# a float32 sum of 2 * num_layers terms lands within ~1e-4 of the stream on a real model; the tiny
# test checkpoints are exact, so a tolerance that passes there can fail on GPT-2
```

Then the linear DLA: `model.lm_head(x)` inside a trace runs the unembedding on any
`[..., hidden]` tensor, stood down from the model's own call, so each term's logits
come from the same trace:

```python
dla = {}
with model.trace(prompt):
    dla["base"] = model.lm_head(model.layers[0].input)[0, -1].save()             # [vocab]
    for i, layer in enumerate(model.layers):
        dla["attn", i] = model.lm_head(layer.self_attn.attention_output)[0, -1].save()
        dla["mlp", i] = model.lm_head(layer.mlp.mlp_output)[0, -1].save()
    head_final = model.lm_head(model.layers[-1].layer_output)[0, -1].save()

assert torch.allclose(sum(dla.values()), head_final, atol=1e-4)                  # linear: the attributions sum

for i in range(model.num_layers):
    print(f"block {i:2d}   attn {dla['attn', i][paris]:+.3f}   mlp {dla['mlp', i][paris]:+.3f}")
```

`lm_head` has no bias on tied-embedding checkpoints (GPT-2, Llama); on one that has
a bias, it is added once per call, so subtract it `2 * num_layers` times from the
sum before comparing.

## The norm is nonlinear

`head_final` above is `lm_head` applied to the *unnormed* final stream, which is
not the model's logits: the model applies `model.norm` first (LayerNorm on GPT-2,
RMSNorm on Llama), then `lm_head`, then any softcap. A norm rescales each position
by its own statistics, so it does not distribute over the sum, and there are two
honest ways to read a contribution through it:

- **Linear DLA**, `model.lm_head(contribution)`: attributions sum exactly, but on
  the unnormed scale. To put them on the model's scale, divide by the final norm's
  per-position scale (`final.norm(dim=-1)` for RMSNorm, up to the weight) and
  multiply by the norm weight; the centering of a LayerNorm is a further term.
- **The normed lens**, `model.project_on_vocab(contribution)`: the full final norm,
  `lm_head` and softcap applied to the contribution *alone*, as in
  [logit-lens](logit-lens.md). It answers "what would this contribution predict by
  itself" on the model's scale, and it does not sum to the logits:

```python
lens = {}
with model.trace(prompt):
    for i, layer in enumerate(model.layers):
        lens["attn", i] = model.project_on_vocab(layer.self_attn.attention_output)[0, -1].save()
        lens["mlp", i] = model.project_on_vocab(layer.mlp.mlp_output)[0, -1].save()
    logits = model.logits[0, -1].save()

sum(lens.values())      # not close to `logits`: the norm was applied per term
```

Use the linear form for attribution shares, the normed lens for "what does this
term say"; say which you used.

## The base: `layers[0].input`, not `token_embeddings`

`model.token_embeddings` is the embedding module's output, before positional
embeddings or embedding norms. On GPT-2 the block input is `wte + wpe`
(after the embedding dropout), so `token_embeddings` is *not* the base of the sum
and the running sum from it misses the positional term. On Llama the two are equal.
`layers[0].input` is what enters block 0 on every family, so it is the base to use;
`token_embeddings` is the value to read when you want the token's own vector.

## Per head

`attention_head_outputs` is each head's output before concatenation and the output
projection, `[batch, seq, heads, head_dim]`. The projection is linear, so head `h`'s
contribution to `attention_output` is its slice times the projection weight's
matching rows. The projection module keeps its native name (`o_proj` on Llama-style
families, `c_proj` on GPT-2, `dense` on GPT-NeoX) and its weight layout differs:
`torch.nn.Linear` stores `[out, in]`, GPT-2's `Conv1D` stores `[in, out]`.

```python
LAYER = model.num_layers // 2
attention = model.layers[LAYER].self_attn

projection = None                          # `or` over envoys calls __len__ on the module; test `is not None`
for name in ("o_proj", "c_proj", "dense"):
    candidate = getattr(attention, name, None)
    if candidate is not None:
        projection = candidate
        break

with model.trace(prompt):
    heads = attention.attention_head_outputs.save()          # [batch, seq, heads, head_dim]
    contribution = attention.attention_output.save()         # [batch, seq, hidden]

W = projection._module.weight
W_in_out = W.t() if isinstance(projection._module, torch.nn.Linear) else W     # [heads * head_dim, hidden]
H, D = heads.shape[2], heads.shape[3]
per_head = torch.einsum("bshd,hdo->bsho", heads, W_in_out.reshape(H, D, -1))    # [batch, seq, heads, hidden]

bias = projection._module.bias
total = per_head.sum(2) + (bias if bias is not None else 0)
torch.testing.assert_close(total, contribution, rtol=1e-4, atol=1e-3)          # the heads sum to the contribution

head_dla = per_head[0, -1] @ model.lm_head._module.weight.t()                    # [heads, vocab], linear
print(head_dla[:, paris])
```

The same numbers come from the model itself: zero every head but `h` in
`attention_head_outputs` inside a trace and read `attention_output`; that equals
`per_head[..., h, :]` plus the bias. Under grouped-query attention the head axis
of `attention_head_outputs` is still `num_heads` (query heads), so the slicing is
unchanged.

## Gotchas

- Read in forward order: block `i`'s `attention_output` then its `mlp_output`, then
  block `i + 1`. Two list comprehensions, one over all attentions and one over all
  MLPs, are out of order (`OutOfOrderError`).
- `model.token_embeddings` must be read before `model.layers[0].input` in one
  trace, and it is not the base of the sum on a family with positional
  embeddings added after it (GPT-2's `wpe`) or an embedding norm (BLOOM).
- `model.lm_head(x)` inside a trace is a stood-down call on your tensor;
  `model.lm_head.output` is the model's own projection of the *normed* final
  stream, and `model.logits` adds the softcap on Gemma-2. Three different things.
- Gemma-4 is not a plain sum. Each block adds a third term on checkpoints with
  per-layer embeddings (`layers[i].per_layer_output`), then multiplies its stream by
  `layers[i]._module.layer_scalar` (0.005 to 0.99 on the released weights). A term
  added in block `i` reaches the last block's output times the product of the
  scalars of blocks `i` through the last, and the base times all of them; weight
  each term by that product before summing or attributing.
- A softcapped model's logits are not a sum of anything; `project_on_vocab` applies
  the cap, the linear DLA does not.
- `getattr(a, "x", None) or getattr(b, ...)` on envoys raises on a module without
  `__len__`; chain with `is not None`.
- The heads' sum matches `attention_output` because that value is the projection's
  output on these families. On a family whose contribution is a post-attention norm's
  output (Gemma-2/3, OLMo-2) the per-head sum is the *pre-norm* attention output;
  compare against `projection.output` there (`o_proj.output` on Gemma-2), and the
  per-head split is of that, not of the contribution.
- The match holds to the checkpoint's dtype: within ~1e-4 relative in float32, and
  a few percent on a bf16 load (a sum of 60 bf16 terms on SmolLM2 lands about 2% off
  logits in the hundreds). Use a relative tolerance, or load with
  `dtype=torch.float32` for the check.
- A name bound inside the trace does not survive it; `parts`, `dla`, `lens` are
  made outside.

## Related

- [logit-lens](logit-lens.md): `project_on_vocab` on the stream itself.
- [ablation](ablation.md): the causal counterpart to an attribution share.
- [attention-patterns](attention-patterns.md): what the heads whose contributions
  you just ranked attend to.
- [../usage/residual-stream.md](../usage/residual-stream.md): the identity and
  where each family binds its contributions.
- [../usage/root-values.md](../usage/root-values.md): `logits`, `token_embeddings`.
- nnsight `docs/usage/access-and-modify.md`: calling modules inside a trace.
- Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits".
