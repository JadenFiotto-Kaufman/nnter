"""`StandardizedTransformer`: a `TransformersModel` renamed to the standard vocabulary."""

from __future__ import annotations

from types import ModuleType
from typing import Any, Sequence

import torch
from jaxtyping import Float, Int
from nnsight.intervention.envoy import Envoy
from nnsight.modeling.transformers import TransformersModel
from torch import Tensor

from . import families
from .components import EProperty, Layer, RelativeEProperty, Standard


class StandardizedTransformer(TransformersModel):
    """A causal language model whose modules answer to one set of names.

    The checkpoint's config is read first (its ``model_type``), the matching
    family toolkit is looked up in `nnter.families.REGISTRY`, and the family's
    ``RENAME`` is handed to nnsight's ``rename`` so every envoy in the tree also
    answers to the standard name. The original names keep working: an alias is
    an extra attribute on the same envoy, not a replacement.

    The family's ``ENVOYS`` wraps its decoder blocks in its `Layer`
    (``layer_output``), its attention modules in its `Attention`
    (``attention_output``, ``attention_probabilities``) and its feed-forwards
    in its `Mlp` (``mlp_output``). The pattern is read inside the eager
    attention forward, so it is unavailable unless the model is loaded with
    ``attn_implementation="eager"``; `status` says which values this
    checkpoint has. See `nnter.components`.

    Args:
        repo_id: A HuggingFace repo id, or an already-loaded ``torch.nn.Module``
            (its own ``config`` is read then).
        rename: Extra aliases, merged over the family's; a key given here wins.
        envoys: Extra ``envoys=`` entries, merged over the family's ``ENVOYS``
            (and nnsight's tensor-parallel envoys when the load shards); a key
            given here wins.
        tokenizer_kwargs: Attributes to set on the loaded tokenizer, such as
            ``padding_side="left"`` or a ``pad_token``.
        **kwargs: Passed through to `TransformersModel`. ``task`` defaults to
            ``"text-generation"``.

    Over the values, `skip_layers`, `steer`, `project_on_vocab` and
    `get_topk_closest_tokens` do the common things (see each). Inside a
    trace `input_ids`, `input_size` and `attention_mask` are what the model
    was called with.

    The root envoy also answers for the whole model, inside a trace: ``logits``
    (the model's final logits, with any softcapping applied), ``token_embeddings``
    (the embedding's output) and ``next_token_probs``; and outside one: the
    sizes ``num_layers``, ``num_heads``, ``num_kv_heads``, ``head_dim``,
    ``qk_head_dim``, ``hidden_size``, ``intermediate_size`` and ``vocab_size``,
    read off the config with the fallbacks older configs need.

    Attributes:
        family: The toolkit module the checkpoint resolved to.
        layers: The decoder blocks, each a `Layer` (the family's subclass).
        embed_tokens, norm, lm_head: The embedding, the final norm, the unembedding.

    Raises:
        UnsupportedFamily: when no family covers the checkpoint's ``model_type``.
    """

    family: ModuleType
    layers: Sequence[Layer]
    embed_tokens: Envoy
    norm: Envoy
    lm_head: Envoy

    def __init__(
        self,
        repo_id: Any,
        *args: Any,
        rename: dict[str, str | list[str]] | None = None,
        envoys: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("task", "text-generation")
        self._add_prefix_false_tokenizer = None
        config = self._read_config(repo_id, kwargs)
        # A multimodal checkpoint's config nests the language model's; the
        # text-generation task builds that model, so its family is the one.
        self.family = families.lookup(getattr(config, "text_config", config).model_type)
        super().__init__(
            repo_id,
            *args,
            rename={**self.family.RENAME, **(rename or {})},
            envoys={
                **self._base_envoys(repo_id, kwargs),
                **self.family.ENVOYS,
                **(envoys or {}),
            },
            **kwargs,
        )
        for key, value in (tokenizer_kwargs or {}).items():
            setattr(self.tokenizer, key, value)

    @staticmethod
    def _base_envoys(repo_id: Any, kwargs: dict) -> dict:
        """What `TransformersModel` would have installed had we passed no ``envoys``.

        It only sets its tensor-parallel envoys as a default, so passing our own
        map would silently drop them on a sharded load; start from them instead.
        """
        from nnsight.modeling.tp.envoys import tp_envoys, wants_tensor_parallel

        return tp_envoys() if wants_tensor_parallel(repo_id, kwargs) else {}

    # -- whole-model values (inside a trace) ---------------------------------

    @EProperty(key="output", description="The model's final logits, [batch, seq, vocab], softcapping applied")
    def logits(self, value: Any) -> Float[Tensor, "batch seq vocab"]:
        """The logits the model returns, ``[batch, seq, vocab]``.

        Read off the model's output, so a family that softcaps after
        ``lm_head`` (Gemma-2) is already accounted for; ``lm_head.output`` is
        the raw projection. Assigning replaces the logits in the model's output.
        """
        return value.logits

    @logits.postprocess
    def logits(self, value: torch.Tensor) -> Any:
        output = self.output
        output.logits = value
        return output

    @RelativeEProperty("embed_tokens.output", description="The token embeddings entering the first block, [batch, seq, hidden]")
    def token_embeddings(self, value: torch.Tensor) -> Float[Tensor, "batch seq hidden"]:
        """The embedding module's output, ``[batch, seq, hidden]``.

        Positional embeddings and embedding norms (GPT-2's ``wpe``, BLOOM's
        ``word_embeddings_layernorm``) are applied after this by the families
        that have them. Assign to replace it.
        """
        return value

    @EProperty(key="output", description="The next-token distribution at the last position, [batch, vocab]; derived, read-only")
    def next_token_probs(self, value: Any) -> Float[Tensor, "batch vocab"]:
        """The next-token distribution at the last position, ``[batch, vocab]``.

        ``logits[:, -1].softmax(-1)``, derived from the model's output; the
        last position is the last token of every row only under left padding.
        Read-only: there is no inverse, so assign ``logits`` instead.
        """
        return value.logits[:, -1].softmax(-1)

    @next_token_probs.postprocess
    def next_token_probs(self, value: Any) -> Any:
        raise AttributeError(
            "next_token_probs is derived from the logits and cannot be assigned; "
            "assign model.logits instead"
        )

    # -- methods over the values (inside a trace unless said otherwise) ----------

    def skip_layers(self, start: int, end: int, skip_with: torch.Tensor | None = None) -> None:
        """Skip blocks ``start`` through ``end`` inclusive.

        The residual stream entering block ``start`` (or ``skip_with``) is
        handed straight to block ``end + 1``; the skipped blocks do not run.
        Negative indices count from the end. Inside a trace::

            with model.trace(prompt):
                model.skip_layers(4, 7)
                logits = model.logits.save()
        """
        start, end = range(self.num_layers)[start], range(self.num_layers)[end]
        hidden = self.layers[start].input if skip_with is None else skip_with
        for i in range(start, end + 1):
            self.layers[i].skip_with(hidden)

    def steer(
        self,
        layers: int | list[int],
        vector: torch.Tensor,
        factor: float = 1.0,
        token_positions: int | list[int] | slice | None = None,
        batch_index: int | None = None,
    ) -> None:
        """Add ``factor * vector`` to the residual stream leaving the given blocks.

        ``vector`` is ``[hidden]`` (or broadcastable to the selected slice).
        ``token_positions`` restricts the positions and ``batch_index`` the
        row; both default to all. The add is in place on ``layer_output``, so
        it reaches the model. Inside a trace, with ``layers`` ascending.
        """
        rows = slice(None) if batch_index is None else batch_index
        cols = slice(None) if token_positions is None else token_positions
        for i in [layers] if isinstance(layers, int) else layers:
            out = self.layers[i].layer_output
            out[rows, cols] += factor * vector.to(out)

    def project_on_vocab(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits for a residual-stream tensor: the final norm, ``lm_head``, and the model's softcapping if any.

        The logit lens: applied to a block's ``layer_output`` it reads that
        layer's prediction; applied to the last block's, it is `logits`.
        Works inside a trace on a live value and outside on a saved one.
        """
        logits = self.lm_head(self.norm(hidden))
        cap = getattr(self.config, "final_logit_softcapping", None)
        return cap * torch.tanh(logits / cap) if cap else logits

    def probs_to_dict(self, probs: torch.Tensor, k: int = 5) -> dict[str, float]:
        """The ``k`` most likely tokens of one ``[vocab]`` distribution, as ``{token: probability}``."""
        values, indices = probs.topk(k)
        return {self.tokenizer.decode(index): value.item() for value, index in zip(values, indices)}

    def get_topk_closest_tokens(self, hidden: torch.Tensor, k: int = 5) -> list[dict[str, float]]:
        """The ``k`` most likely next tokens for each position of ``hidden``, ``[..., hidden]``.

        `project_on_vocab` then softmax, one ``{token: probability}`` per
        position, in row-major order over the leading dimensions.
        """
        probs = self.project_on_vocab(hidden).softmax(-1).reshape(-1, self.vocab_size)
        return [self.probs_to_dict(row, k) for row in probs]

    # -- availability ----------------------------------------------------------

    def status(self, layer: int | None = None) -> dict[str, Any]:
        """Which standard values this checkpoint has, without running anything.

        With ``layer``, that block's values by dotted name (``"layer_output"``,
        ``"self_attn.attention_probabilities"``, ``"mlp.mlp_output"``): ``None``
        when available, else the reason, including ``"no <module> module"``
        when the block has no such module at all (a hybrid's blocks have either
        ``self_attn`` or ``linear_attn``). Without, the root's values
        plus every block value: ``None`` when available on every block, else
        ``{layer: reason}`` for the blocks where it is not, so a hybrid reads
        as a short dict.

        The tree decides what is listed: every child of a block that carries
        standard values (a `Standard` envoy) is walked under its standard
        name, so a value added through ``envoys=`` or a registered family
        appears here as it does in the envoy's own `Standard.status`, and a
        module no block has (OPT's ``mlp``) has no entry.
        """
        hosts = self._hosts()
        if layer is not None:
            return self._layer_status(self.layers[layer], hosts)
        status: dict[str, Any] = {name: value.reason(self) for name, value in Standard.values.__func__(type(self)).items()}
        per_layer = [self._layer_status(block, hosts) for block in self.layers]
        for name in per_layer[0]:
            missing = {i: reasons[name] for i, reasons in enumerate(per_layer) if reasons[name]}
            status[name] = missing or None
        return status

    @staticmethod
    def _standard_children(block: Envoy) -> dict[str, Standard]:
        """The block's children that carry standard values, by standard name (the alias where one is bound)."""
        aliases = {native: alias for alias, native in block._aliases.items()}
        found = {aliases.get(name, name): child for name, child in block._named_children() if isinstance(child, Standard)}
        for alias, path in block._aliases.items():
            if "." in path:  # a mount: the module sits deeper (DBRX's ``norm_attn_norm.attn`` as ``self_attn``)
                child = block.get(path)
                if isinstance(child, Standard):
                    found[alias] = child
        return found

    def _hosts(self) -> dict[str, list[str]]:
        """Standard-value hosts across every block: module name -> value names, in first-seen order.

        The union over the blocks, so a hybrid lists both ``self_attn`` and
        ``linear_attn`` and a block lacking one reports it as missing; a
        module no block has (OPT's ``mlp``) is not listed.
        """
        hosts: dict[str, dict[str, None]] = {}
        for block in self.layers:
            for module, child in self._standard_children(block).items():
                hosts.setdefault(module, {}).update(dict.fromkeys(child.values()))
        return {module: list(names) for module, names in hosts.items()}

    def _layer_status(self, block: Any, hosts: dict[str, list[str]]) -> dict[str, str | None]:
        status: dict[str, str | None] = dict(block.status())
        present = self._standard_children(block)
        for module, names in hosts.items():
            envoy = present.get(module)
            reasons = envoy.status() if envoy is not None else {}
            for name in names:
                if envoy is None:
                    status[f"{module}.{name}"] = f"no {module} module on this block"
                else:
                    status[f"{module}.{name}"] = reasons.get(name, f"no {name} value on this block's {module}")
        return status

    # -- the input (inside a trace) ----------------------------------------------

    @EProperty(key="input", description="The token ids the model was called with, [batch, seq]")
    def input_ids(self, value: Any) -> Int[Tensor, "batch seq"]:
        """The token ids the model was called with, ``[batch, seq]``. Assign to run the model on other ids."""
        return value[1]["input_ids"]

    @input_ids.postprocess
    def input_ids(self, value: Tensor) -> Any:
        args, kwargs = self.inputs
        return args, {**kwargs, "input_ids": value}

    @EProperty(key="input", description="The attention mask the model was called with, [batch, seq]; zeros are padding")
    def attention_mask(self, value: Any) -> Int[Tensor, "batch seq"]:
        """The attention mask the model was called with, ``[batch, seq]``; zeros are padding. Assignable."""
        return value[1]["attention_mask"]

    @attention_mask.postprocess
    def attention_mask(self, value: Tensor) -> Any:
        args, kwargs = self.inputs
        return args, {**kwargs, "attention_mask": value}

    @EProperty(key="input", description="[batch, seq] of the current call; read-only")
    def input_size(self, value: Any) -> torch.Size:
        """``[batch, seq]`` of the current call, from the ids; read-only."""
        return value[1]["input_ids"].shape

    @input_size.postprocess
    def input_size(self, value: Any) -> Any:
        raise AttributeError("input_size is the ids' shape and cannot be assigned; assign input_ids")

    # -- tokenizers ---------------------------------------------------------------

    @property
    def add_prefix_false_tokenizer(self) -> Any:
        """The checkpoint's tokenizer loaded with ``add_prefix_space=False``, so ``"word"`` and ``" word"`` differ.

        What `nnter.prompt_utils.get_first_tokens` uses. Loaded once, on first use.
        """
        if self._add_prefix_false_tokenizer is None:
            from transformers import AutoTokenizer

            self._add_prefix_false_tokenizer = AutoTokenizer.from_pretrained(self.repo_id, add_prefix_space=False)
        return self._add_prefix_false_tokenizer

    # -- sizes (from the config) ----------------------------------------------

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def num_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        """Key/value heads: fewer than ``num_heads`` under grouped-query attention."""
        config = self.config
        heads = getattr(config, "num_key_value_heads", None)
        if heads is not None:
            return heads
        if getattr(config, "new_decoder_architecture", False):  # Falcon's 40B layout
            return config.num_kv_heads
        return 1 if getattr(config, "multi_query", False) else self.num_heads

    @property
    def head_dim(self) -> int:
        """Width of one attention head's values and outputs.

        ``v_head_dim`` on multi-head latent attention (DeepSeek), else the
        config's ``head_dim`` when it says (Qwen3, Gemma), else
        ``hidden_size // num_heads``.
        """
        config = self.config
        return getattr(config, "v_head_dim", None) or getattr(config, "head_dim", None) or self.hidden_size // self.num_heads

    @property
    def qk_head_dim(self) -> int:
        """Width of one head's queries and keys: `head_dim`, except under multi-head latent attention, where it is ``qk_nope_head_dim + qk_rope_head_dim``."""
        config = self.config
        if getattr(config, "qk_nope_head_dim", None) is not None:
            return config.qk_nope_head_dim + config.qk_rope_head_dim
        return self.head_dim

    @property
    def intermediate_size(self) -> int:
        """Width of the MLP's hidden layer.

        The dense MLP's width; a mixture of experts' experts are
        ``config.moe_intermediate_size`` wide instead. A GPT-2-style config
        says it as ``n_inner`` (``None`` meaning 4 times the hidden size) and
        may carry an ``intermediate_size`` the model never reads, so
        ``n_inner`` is consulted first. Otherwise the config's
        ``intermediate_size`` (``ffn_dim`` on OPT), else ``expansion_ratio``
        (MPT) or 4 times the hidden size (BLOOM, Falcon).
        """
        if hasattr(self.config, "n_inner"):
            return self.config.n_inner or 4 * self.hidden_size
        size = getattr(self.config, "intermediate_size", None) or getattr(self.config, "ffn_dim", None)
        if size is not None:
            return size
        return int(self.hidden_size * getattr(self.config, "expansion_ratio", 4))

    # -- remote ------------------------------------------------------------------

    def _remoteable_class(self) -> type:
        """The class in this model's remote key: `TransformersModel`, what a server deploys.

        A remote trace re-runs the block against the client's envoy tree, so
        the aliases and the family's envoy classes travel with the request
        (by reference: the server needs nnter installed). The deployed model
        itself is a plain `TransformersModel`, so the key says so; a
        subclass-specific key would match nothing on the server.
        """
        return TransformersModel

    # -- loading ---------------------------------------------------------------

    @staticmethod
    def _read_config(repo_id: Any, kwargs: dict) -> Any:
        """The checkpoint's config, before any model is built.

        A ready module carries its own; a repo id is read with ``AutoConfig``,
        so a config transformers cannot parse fails here with its own error.
        The Hub caches the file, and the meta build reads it again.
        """
        if isinstance(repo_id, torch.nn.Module):
            return repo_id.config
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            repo_id,
            revision=kwargs.get("revision"),
            trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
        )
