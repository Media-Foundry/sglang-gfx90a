"""Single home for the chat-encoding dispatch.

Which encoder turns chat messages into prompt tokens is a property of the
model, so the serving path and offline tools (benchmarks, evals) must resolve
it here instead of re-deriving it from model architectures themselves.
"""

from __future__ import annotations

import ast
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from sglang.srt.entrypoints.openai import encoding_dsv4

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def resolve_dsv41_effort_profile(model_path: str):
    """Read alias budgets from a local checkpoint without executing its code.

    V4.1 snapshots differ (e.g. low/high=50/75 vs 25/50). Keep the bundled
    encoder default when no checked local reference is available.
    """
    from sglang.srt.entrypoints.openai import encoding_dsv41

    for filename in ("encoding/encoding.py", "encoding/encoding_dsv41.py"):
        path = Path(model_path) / filename
        try:
            if not path.is_file() or path.stat().st_size > 1 << 20:
                continue
            values = {}
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif isinstance(node, ast.AnnAssign):
                    targets, value = [node.target], node.value
                else:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in (
                        "REASONING_EFFORT_MAPPINGS",
                        "DEFAULT_REASONING_EFFORT",
                    ):
                        values[target.id] = ast.literal_eval(value)
            mapping = values.get("REASONING_EFFORT_MAPPINGS")
            default = values.get("DEFAULT_REASONING_EFFORT")
            if (
                isinstance(mapping, dict)
                and mapping
                and all(
                    isinstance(k, str) and type(v) is int and 1 <= v <= 100
                    for k, v in mapping.items()
                )
                and isinstance(default, str)
                and default in mapping
            ):
                return mapping, default
        except (OSError, SyntaxError, ValueError, TypeError, UnicodeError):
            logger.debug("Could not read V4.1 effort profile from %s", path)
    return (
        dict(encoding_dsv41.REASONING_EFFORT_MAPPINGS),
        encoding_dsv41.DEFAULT_REASONING_EFFORT,
    )


def dsv41_effort_budget(effort, profile):
    mapping, default = profile
    if effort is None:
        effort = default
    if isinstance(effort, str):
        if effort not in mapping:
            raise ValueError(
                f"Unsupported V4.1 reasoning effort {effort!r}; "
                f"expected {list(mapping)} or integer 1..100"
            )
        return mapping[effort]
    return effort  # The encoder strictly validates numeric budgets.


DSV4_REASONING_EFFORT_PROFILE_OVERRIDE = "dsv4_reasoning_effort_profile"
_DSV4_REASONING_EFFORT_ENCODER = "encoding/encoding_dsv4.py"
_MAX_DSV4_ENCODER_BYTES = 1 << 20


def _detect_dsv4_reasoning_effort_profile(
    model_path: str, revision: Optional[str] = None
) -> Optional[str]:
    encoder_path = Path(model_path) / _DSV4_REASONING_EFFORT_ENCODER
    try:
        if not encoder_path.is_file():
            from huggingface_hub import hf_hub_download

            encoder_path = Path(
                hf_hub_download(
                    model_path,
                    _DSV4_REASONING_EFFORT_ENCODER,
                    revision=revision,
                )
            )
        if encoder_path.stat().st_size > _MAX_DSV4_ENCODER_BYTES:
            return None
        tree = ast.parse(encoder_path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.debug(
            "Could not inspect DeepSeek-V4 checkpoint encoder at %s: %s",
            encoder_path,
            error,
        )
        return None

    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                assignments[target.id] = ast.literal_eval(value)
            except (TypeError, ValueError):
                continue

    prompts = assignments.get("REASONING_EFFORT_PROMPTS")
    if (
        assignments.get("DEFAULT_REASONING_EFFORT") == "low"
        and isinstance(prompts, dict)
        and {"low", "high", "max"} <= prompts.keys()
    ):
        return "official"
    if "REASONING_EFFORT_MAX" in assignments:
        return "preview"
    return None


def _validate_dsv4_reasoning_effort_profile(profile: str) -> str:
    if profile not in encoding_dsv4.REASONING_EFFORT_PROFILES:
        raise ValueError(
            f"Invalid {DSV4_REASONING_EFFORT_PROFILE_OVERRIDE}: {profile!r}; "
            f"expected one of {list(encoding_dsv4.REASONING_EFFORT_PROFILES)}"
        )
    return profile


def resolve_dsv4_reasoning_effort_profile(
    *,
    model_path: str,
    revision: Optional[str] = None,
    override: Optional[str] = None,
) -> str:
    if override is not None:
        return _validate_dsv4_reasoning_effort_profile(override)

    return (
        _detect_dsv4_reasoning_effort_profile(
            model_path=model_path,
            revision=revision,
        )
        or "preview"
    )


def resolve_chat_encoding_spec(
    *,
    hf_config: Any,
    tokenizer: Any,
    tool_call_parser: Optional[str] = None,
) -> Optional[str]:
    """Return the chat encoding spec for a model.

    None means the default path (HF chat template); any non-None spec also owns
    reasoning-history rendering (:func:`spec_owns_reasoning_history`).
    """
    if tool_call_parser == "deepseekv41":
        return "dsv41"
    if tool_call_parser == "deepseekv4":
        return "dsv4"
    if tool_call_parser == "deepseekv32":
        return "dsv32"
    if tool_call_parser == "kimi_k3":
        return "kimi_k3"

    architectures = hf_config.architectures
    arch = architectures[0] if architectures else ""

    # V4.1 has spaced DSML tags and a numeric effort budget. Check it before
    # the V4 substring, otherwise an apparently working plain-chat endpoint
    # silently uses the wrong tool/thinking conversation format.
    if "DeepseekV41" in arch:
        return "dsv41"
    if "DeepseekV4" in arch:
        return "dsv4"
    if "KimiK3" in arch:
        return "kimi_k3"

    # Inkling has no Jinja chat_template and uses a tiktoken base + a special-token
    # overlay + negative MM placeholders, so it can't go through apply_chat_template;
    # render input_ids directly via the Inkling renderer (serving_chat._encode_messages).
    if "InklingForConditionalGeneration" in arch:
        return "inkling"

    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    if "DeepseekV3" in arch and not has_chat_template:
        return "dsv32"
    return None


def spec_owns_reasoning_history(spec: Optional[str]) -> bool:
    """Whether the encoder for ``spec`` renders assistant reasoning history itself.

    Custom encoders frame the reasoning and content channels, so history must be
    passed as assistant ``reasoning_content``. Splicing a detector's markers into
    content instead nests a reasoning block inside the content channel and leaves
    the real one empty, teaching the model to emit raw markers as visible text.

    Answered for the whole family rather than a list of specs, so a new spec gets
    the safe default: worst case is dropped history, not a leak.
    """
    return spec is not None


def encode_simple_chat(
    *,
    tokenizer: Any,
    spec: Optional[str],
    messages: List[Dict[str, Any]],
    thinking_mode: str = "chat",
) -> List[int]:
    """Encode a plain-text chat conversation into prompt token ids.

    Minimal encode for offline tools: no tools, no multimodal content, no
    continue_final_message; the serving path keeps its full request-level
    pipeline in ``serving_chat``. Like
    ``serving_chat``, an empty system message is prepended when the
    conversation does not start with one (for the dsv4/dsv32 encoders this
    currently renders to zero tokens, but keeping the insertion explicit ties
    this helper to the serving semantics rather than to that coincidence).
    """
    if spec == "inkling":
        from sglang.srt.parser.inkling_renderer import render_inkling_messages
        from sglang.srt.parser.inkling_tokenizer import InklingTokenizer

        return render_inkling_messages(
            messages,
            InklingTokenizer(tokenizer=tokenizer),
            add_generation_prompt=False,
        )

    if spec in ("dsv41", "dsv4", "dsv32"):
        if spec != "dsv41" and messages and messages[0]["role"] != "system":
            messages = [{"role": "system", "content": ""}] + list(messages)
        if spec == "dsv41":
            from sglang.srt.entrypoints.openai import encoding_dsv41

            real_input = encoding_dsv41.encode_messages(
                messages,
                thinking_mode=thinking_mode,
                reasoning_effort=dsv41_effort_budget(
                    None,
                    resolve_dsv41_effort_profile(
                        getattr(tokenizer, "name_or_path", "") or ""
                    ),
                ),
            )
        elif spec == "dsv4":
            from sglang.srt.entrypoints.openai import encoding_dsv4

            real_input = encoding_dsv4.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        else:
            from sglang.srt.entrypoints.openai import encoding_dsv32

            real_input = encoding_dsv32.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        return tokenizer.encode(real_input)

    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "This model has no HF chat template and no custom chat encoder; "
            f"cannot encode chat messages with {getattr(tokenizer, 'name_or_path', tokenizer)!r}."
        )
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
