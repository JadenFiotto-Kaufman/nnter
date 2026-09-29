---
title: Overriding Values
one_liner: How a family redefines a standard value when the base does not hold — `RelativeEProperty`, `SourceEProperty` with `attribute`/`select`, `unavailable` markers and predicates, `off_interface`, `seq_first`, a clone with a transform, `postprocess` for writes; and a root size, which is a plain function in the family module, not a descriptor.
tags: [extending, families, eproperty, source, availability]
related: [docs/extending/adding-a-family.md, docs/extending/custom-values.md, docs/extending/finding-source-ops.md]
sources: [nnter/components/eproperty.py, nnter/components/attention.py, nnter/components/layer.py, nnter/components/mlp.py, nnter/components/standard.py, nnter/families/gemma2.py, nnter/families/olmo2.py, nnter/families/bloom.py, nnter/families/mpt.py, nnter/families/gptj.py, nnter/families/falcon.py, nnter/families/gpt2.py, nnter/families/gpt_oss.py, nnter/families/deepseek_v2.py, nnter/families/deepseek_v3.py, nnter/standardized.py]
---

# Overriding Values

## What this is for

A standard value means the same thing on every family: `attention_output` is what the
attention adds to the residual stream, `attention_probabilities` the pattern the values
are mixed with. The base `Layer`, `Attention` and `Mlp` locate those values where
Llama's forward puts them. When a family's forward puts a value somewhere else, its
subclass redefines the descriptor under the same name, pointing at the right place,
and the name keeps its meaning. This page lists every shape of override the shipped
families use, with the real snippet and the reason. The rule for all of them: keep the
name, keep the layout name in the annotation (`-> Residual`, the base's, imported from
`..components` with the envoys), keep the description, change only the location.

## Canonical pattern

Gemma-2's block is a sandwich: `x + post_attention_layernorm(attn(input_layernorm(x)))`.
What the block adds is the sibling norm's output, not the attention module's, so the
value points there:

```python
# nnter/families/gemma2.py
from ..components import Attention, Layer, Mlp, RelativeEProperty, Residual


class Attention(Attention):
    """Gemma-2's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @RelativeEProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    @RelativeEProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value
```

Every read, write and in-place edit of `layers[i].self_attn.attention_output` now goes to
`layers[i].post_attention_layernorm.output`, and the contribution identity the suite checks
holds. Gemma-3 (text), OLMo-2 and OLMo-3 are the same override; OLMo-2/3 have only the
post-norms, so `self_attn.input` is the block input there.

## `RelativeEProperty`: a value another module produces

`key` is `"<path>.<attribute>"`. The path resolves from the host envoy the way attribute
access does, aliases included; a leading `../` steps to the parent first, by native name.
`"../post_attention_layernorm.output"` on an attention envoy reaches its sibling norm;
`"embed_tokens.output"` on the root is `token_embeddings`. The value is served at that
location, so in-place edits reach the model without anything more.

An `.input` location serves the raw `(args, kwargs)` pair, the same as nnsight's
`.inputs`, so a `RelativeEProperty` on one destructures it:

```python
class Layer(gpt2.Layer):
    @RelativeEProperty("post_attention_layernorm.input", description="The residual stream after the attention sublayer")
    def mid_stream(self, value) -> Residual:
        (hidden,), _ = value
        return hidden
```

On GPT-2 tiny this reads `layer.input + attention_output` exactly.

The location can be an operation in the parent's forward. Llama 4's mixture of experts
returns its output flattened to `[batch * seq, hidden]` and the block views it back
(`residual + hidden_states.view(residual.shape)`), so its `mlp_output` is that view:

```python
# nnter/families/llama4_text.py
class Mlp(Mlp):
    @RelativeEProperty("../source.hidden_states_view_0.output", description="...", unavailable=_not_a_block_feed_forward)
    def mlp_output(self, value) -> Residual:
        return value
```

An operation is only served on a call whose forward was source-instrumented before the
call began, and this one is read after the block has started (after its attention), so
the family's `Layer` builds its `.source` in `__init__` and again in `_update` (when
real weights replace the meta ones a lazy load starts from). Without that, the first
trace that reads `attention_output` and then `mlp_output` raises nnsight's
`OutOfOrderError`, and later traces work.

## `SourceEProperty`: a value at an operation inside the forward

BLOOM's sublayers take the residual as an argument and add it inside the module
(`dropout_add(x, residual, ...)`), so the module's output is a residual-stream state, not
a contribution. The contribution is the first argument of that call:

```python
# nnter/families/bloom.py
class Attention(Attention):
    @SourceEProperty(
        "dropout_add_0",
        attribute="input",
        description="What the attention adds to the residual stream: the tensor entering dropout_add",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    @SourceEProperty("dropout_add_0", attribute="input", description="What the MLP adds to the residual stream: the tensor entering dropout_add")
    def mlp_output(self, value) -> Residual:
        return value
```

MPT's MLP adds the residual inside too; its contribution is the dropout's output, the
tensor just before the add: `SourceEProperty("F_dropout_0", description=...)`
(`attribute` defaults to `"output"`).

`op` is the operation's path under the module's `.source`, with `.source.` between a
call and an operation inside it; [finding-source-ops.md](finding-source-ops.md) is how
to find the name. The descriptor walks `.source` before every read or write, so the
operation is instrumented for the current run, then reads or writes the operation's own
`.output` / `.input` / `.inputs`. An operation the run does not have raises
`SourceNotAvailable` naming what exists. `op` may also be a function of the envoy
returning the path, decided on the instance: Falcon's `by_alibi(without, with_alibi)`
returns one that reads `config.alibi`, and `branched(...)` one that reads a binding the
forward makes ([finding-source-ops.md](finding-source-ops.md)).

### `attribute` and `select`

- `attribute="input"` is the call's first argument; assigning replaces the first
  argument and keeps the rest (BLOOM above).
- `attribute="inputs", select=n` is the n-th positional argument, `select="name"` a
  keyword argument. A write repacks that one element into the call's `(args, kwargs)`,
  so assigning replaces just that argument. The base `Attention` reads the queries, keys
  and values this way off the interface call:

  ```python
  # nnter/components/attention.py
  @SourceEProperty(INTERFACE, attribute="inputs", select=1, description="The queries entering attention, [batch, heads, seq, head_dim]", unavailable=interface_reason)
  def attention_queries(self, value: torch.Tensor) -> Queries:
      return value
  ```

  GPT-J does its arithmetic in its own `_attn` method, so its family reads the same
  three values off that call: `SourceEProperty("self__attn_0", attribute="inputs", select=0, ...)`
  for the queries, `select=1` keys, `select=2` values.
- `attribute="output", select=i` is one element of a returned tuple. BLOOM's
  `_reshape` returns `(query, key, value)`, so its family reads
  `SourceEProperty("self__reshape_0", attribute="output", select=0)` for the queries and
  `select=1`, `select=2` for the rest; Falcon's queries and keys are
  `apply_rotary_pos_emb_0`'s two returns (`select=0`, `select=1`), and its values the
  `value_layer_0` binding just before it.

Because the descriptor reads through the operation's own descriptors, the element it
hands back is the object the call holds: in-place edits reach the model, and no
`transform` exists or is needed on a `SourceEProperty`.

### Reuse the base description and the base layout name

A redefined value keeps its meaning, so it keeps its description and its layout name:

```python
from ..components import Queries


@SourceEProperty("self__reshape_0", attribute="output", select=0, description=Attention.attention_queries.description)
def attention_queries(self, value) -> Queries:
    return value
```

`Attention.attention_queries` on the class is the descriptor itself (`__get__` with no
instance returns it), so `.description` is the base's text; `-> Queries` is the base's
annotation, so `layout` is the same alias on both (`bloom.Attention.attention_queries.layout
is Attention.attention_queries.layout`) and the redefinition cannot drift from it.

### The pattern: the dropout after the softmax

The base reads `attention_probabilities` at `attention_interface_1.source.nn_functional_dropout_0`
and `attention_scores` at the softmax's input. A family whose attention does its own
arithmetic points both at its own operations: GPT-J at `self__attn_0.source.self_attn_dropout_0`,
MPT at its `nn_functional_dropout_0`, BLOOM at `self_attention_dropout_0`. Falcon without
alibi has no dropout after its softmax, so its pattern is `F_softmax_0`'s output; with
alibi it is `self_attention_dropout_0`, after the second softmax, and
`by_alibi("F_softmax_0", "self_attention_dropout_0")` is the op that picks per checkpoint. Read
the pattern after the dropout wherever one exists: that is the tensor the values are
mixed with, in the model's dtype and, on a sink model, with the sink column dropped.

GPT-OSS is on the shared interface but its softmax takes one extra column (the sink), so
its family reads `attention_scores` one step earlier, at the masked scores bound just
before the sink joins them, and flags the sink for the suite:

```python
# nnter/families/gpt_oss.py
class Attention(Attention):
    SINK = True

    @SourceEProperty(f"{INTERFACE}.source.attn_weights_1", description=Attention.attention_scores.description, unavailable=interface_reason)
    def attention_scores(self, value) -> Pattern:
        return value
```

## Availability

### `unavailable(...)`: a value the family does not have

A marker in the class body replaces the inherited descriptor, keeps the name in the tree
and the repr (`(attention_scores): Unavailable: <reason>`), makes `status()` report the
reason, and makes any access raise `nnter.Unavailable` with it before the model runs:

```python
from nnter.components import NOT_ON_INTERFACE, unavailable


class Attention(Attention):
    attention_scores = unavailable(NOT_ON_INTERFACE)
```

`NOT_ON_INTERFACE` is the reason a family gives for an interface value it has not mapped
onto its own arithmetic. A value that cannot exist takes its own reason
(`unavailable("no softmax: the attention is linear")`).

### `off_interface()`: one decision for every interface value

The base `Attention`'s interface values all take `unavailable=interface_reason`, which
calls `self.off_interface()`; the default is `needs_eager` (the model runs `sdpa` or
another fused kernel). A family with another reason overrides the method rather than
each value:

```python
# nnter/families/gpt2.py
class Attention(Attention):
    def off_interface(self):
        if self._module.config.reorder_and_upcast_attn:
            return "this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path"
        return super().off_interface()
```

### A per-instance `unavailable=` callable

`unavailable=` takes a function of the envoy returning a reason or `None`, evaluated on
the instance so the checkpoint's config decides. `needs_eager` is the one every value read
inside the eager attention forward uses, and it is nothing more than such a function:

```python
# nnter/components/attention.py
def needs_eager(envoy: Envoy) -> str | None:
    implementation = envoy._module.config._attn_implementation
    if implementation != "eager":
        return f"read inside the eager attention forward, but this model runs {implementation!r}; load with attn_implementation='eager'"
    return None
```

Falcon's six interior values pass it as is (`unavailable=needs_eager`), on both of its
attention branches. A family with a config flag of its own writes a predicate of the same
shape; one that combines two reasons reads `needs_eager(self) or <its own check>`, so the
eager reason wins when both apply; and when the flag decides for every interface value at
once, `off_interface()` above is the place.

Families whose attention ignores `attn_implementation` (BLOOM, MPT) pass no
`unavailable=` on their own operations: the pattern needs no eager load there.

## Layout: `seq_first` views

`attention_head_outputs` is `[batch, seq, heads, head_dim]`, what the shared interface
returns. GPT-J, MPT and Falcon keep heads first at their own operation, so the family
serves a transposed view on read and transposes back on write. `seq_first` is its own
inverse, so both callbacks are the same function:

```python
# nnter/families/mpt.py
@SourceEProperty("torch_matmul_1", description=Attention.attention_head_outputs.description)
def attention_head_outputs(self, value) -> HeadOutputs:
    return seq_first(value)

@attention_head_outputs.postprocess
def attention_head_outputs(self, value):
    return seq_first(value)
```

A view keeps in-place edits landing on the model's tensor. BLOOM's `bmm` result is
`[batch * heads, seq, head_dim]`, so its family views and transposes on read and
reverses both on write in the `postprocess`.

## A clone carried back by a transform

Falcon's parallel block adds the attention output *into the MLP's output tensor in
place*. A plain `mlp_output` would be a live tensor the block later mutates, so a saved
read would silently become `mlp + attn`. The value reads a clone; a clone is invisible
to the model, so in-place edits to it would be lost, and an `eproperty` transform hands
the edited copy back to be swapped in once the block is done with the read:

```python
# nnter/families/falcon.py
class Mlp(Mlp):
    @EProperty(key="output", description="What the MLP adds to the residual stream (a copy, since the block adds the attention into the live tensor in place)")
    def mlp_output(self, value) -> Residual:
        return first_tensor(value).clone()

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value)

    @mlp_output.transform
    def mlp_output(self, value, raw):
        # Fires on the model side, after the read. The module returns a bare
        # tensor, so ``raw`` needs no rebuilding around the edited copy.
        return value.clone()
```

The transform's second clone keeps the user's tensor clean when the block then adds into
the swapped-in one; `test_falcon.py` checks both that `mlp.output == mlp_output +
attention_output` and that `mlp_output[:] = 0` moves the logits while the saved copy stays
zero. A transform is needed only when the preprocess returns something other than the
served object (a clone, a reshaped copy) *and* in-place edits must still reach the model.
GPT-2's MLP output is not mutated later, so the base holds and `mlp_output[:] = 0` reaches
the model with no clone and no transform. `transform(self, view, raw)` receives the raw
served value so a module that returns a tuple can be rebuilt around the edited element
(`(edited.clone(), *raw[1:])`); see nnsight docs/developing/extending-envoy.md.

## `postprocess`: writes on a `key="output"` value

The base boundary values are `EProperty(key="output")` with `first_tensor` as the
preprocess and `rewrap` as the postprocess, so a tuple module (GPT-J's block, GPT-2's
attention returning `(attn_output, attn_weights)`) reads as a tensor and an assignment
puts the tensor back in its tuple with the other elements unchanged:

```python
# nnter/components/layer.py
@EProperty(key="output", description="The residual stream leaving the block, a tensor even when the block returns a tuple")
def layer_output(self, value: Any) -> Residual:
    return first_tensor(value)

@layer_output.postprocess
def layer_output(self, value: torch.Tensor) -> Any:
    return rewrap(self, value)
```

A family that redefines a value on `key="output"` (Falcon's `Mlp` above) keeps both
halves. A value redefined as a `RelativeEProperty` or `SourceEProperty` at a location
that serves a bare tensor needs neither.

## A size: a function in the family module

The root's sizes are not value descriptors. Each is a `StandardizedProperty` on
`StandardizedTransformer` (`num_layers`, `hidden_size`, `vocab_size`, `num_heads`,
`num_kv_heads`, `head_dim`, `qk_head_dim`, `intermediate_size`): no location, nothing
served inside a trace, no `status()` entry, only a plain rule over the config, and
read-only (an assignment raises `AttributeError` pointing at `def <name>(model)`). A family
whose config spells a size its own way overrides it with a module-level function of the
same name taking the model, which the descriptor calls instead of its rule. DeepSeek-V2's
latent attention gives values and queries different widths, and the config's own
`head_dim` key is the latent width that no served value has:

```python
# nnter/families/deepseek_v2.py

def head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's values and outputs: ``v_head_dim`` (the config's ``head_dim`` is the latent width, which no served value has)."""
    return model.config.v_head_dim


def qk_head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's queries and keys: the non-rotary part plus the rotary part."""
    return model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
```

DeepSeek-V3 has the same attention and imports both (`from .deepseek_v2 import head_dim,
qk_head_dim`). The other shipped overrides are `intermediate_size` on GPT-2, GPT-J, OPT,
MPT, BLOOM and Falcon, and `num_kv_heads` on Falcon; the full list with what each reads
is in [../usage/root-values.md](../usage/root-values.md#sizes), and the recipe in
[adding-a-family.md](adding-a-family.md#sizes). Keep the docstring in the same voice as
the root's: what the width is, and which config key says so.

## Gotchas

- **Keep the name.** An override under another name adds a value instead of replacing one;
  the inherited descriptor stays, and `status()` keeps reporting it.
- **Keep the layout name.** Annotate the redefinition with the base's name from
  `..components` (`-> Keys`, `-> Residual`), never an inline
  `Float[Tensor, "..."]`: `layout` and `dims` are read off the annotation, the name
  keeps them identical to the base's, and the suite checks every value's tensor against
  it ([custom-values.md](custom-values.md), [../usage/layouts.md](../usage/layouts.md)).
- **An `.input` location serves `(args, kwargs)`.** A `RelativeEProperty` or `EProperty`
  keyed there destructures on read and repacks on write; `SourceEProperty(...,
  attribute="input")` already means the first argument.
- **Reads in one trace follow forward order.** Falcon's values bind before the rotary that
  produces its queries and keys, so `attention_values` must be read first; the suite
  reads interior values one per trace for this reason.
- **An `unavailable=` predicate must not raise `AttributeError`.** A descriptor's
  `AttributeError` falls through to `Envoy.__getattr__` as "no attribute"; `EProperty`
  re-raises it as a `RuntimeError` naming the check, so a wrong config attribute in a
  predicate reports itself on a read. `status()` calls the predicate directly and lets
  the raw `AttributeError` through (`'GPT2MLP' object has no attribute 'config'`): an
  MLP module carries no `config`, so a per-checkpoint predicate on an `Mlp` reaches it
  another way.
- **No `transform` on a `SourceEProperty`.** It reads through the operation's own
  descriptors, so the element it returns is the model's object; in-place edits land
  without one, and `preprocess`/`postprocess` are the only callbacks.
- **A class attribute like `SINK` is for tooling**, not availability; `status()` reports
  only descriptors.
- **A size override goes on the module, not on a class.** `StandardizedProperty` looks
  for `model.family.<name>`; a `head_dim` on the family's `Attention` class is an
  ordinary attribute the root never reads.

## Related

- [adding-a-family.md](adding-a-family.md): where these subclasses go.
- [finding-source-ops.md](finding-source-ops.md): the operation names a `SourceEProperty` takes.
- [custom-values.md](custom-values.md): adding a value rather than redefining one.
- nnsight docs/developing/extending-envoy.md: `eproperty`, `postprocess` and `transform` in full.
