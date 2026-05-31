import asyncio
import os
from typing import List, Dict, Optional, Tuple

import tiktoken
from groq import AsyncGroq, RateLimitError

from prompts import (
    STAGE1_SYSTEM, STAGE1_USER_TEMPLATE,
    CODE_EXTRACTOR_SYSTEM, DECISION_EXTRACTOR_SYSTEM,
    DEBUG_EXTRACTOR_SYSTEM, INTENT_EXTRACTOR_SYSTEM,
    EXTRACTOR_USER_TEMPLATE,
    SYNTHESIS_SYSTEM, SYNTHESIS_USER_TEMPLATE,
)

STAGE1_MODEL = "llama-3.1-8b-instant"
STAGE2_MODEL = "llama-3.3-70b-versatile"
STAGE3_MODEL_ANTHROPIC = "claude-sonnet-4-5"
STAGE3_MODEL_GROQ = "llama-3.3-70b-versatile"

# Groq free tier: ~6000 TPM per model. We budget 4500 tokens for the
# conversation portion of each request, leaving room for the system
# prompt (~500 tokens) and output tokens within the per-minute window.
MAX_CONVERSATION_TOKENS = 4500
MAX_SEGMENTS_TOKENS = 3500

_USE_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

_enc = tiktoken.get_encoding("cl100k_base")


def _get_groq() -> AsyncGroq:
    return AsyncGroq(api_key=os.environ["GROQ_API_KEY"])


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _truncate_text(text: str, max_tokens: int) -> Tuple[str, bool]:
    """Keep the first 25 % and last 75 % of the token budget.

    The beginning establishes the project; the end contains the most
    recent work — both matter more than the middle for continuity.
    """
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text, False

    head_n = max_tokens // 4
    tail_n = max_tokens - head_n
    dropped = len(tokens) - max_tokens

    head = _enc.decode(tokens[:head_n])
    tail = _enc.decode(tokens[-tail_n:])
    marker = f"\n\n[... {dropped} tokens from the middle omitted — keeping start and most recent context ...]\n\n"
    return head + marker + tail, True


def _format_messages(messages: List[Dict[str, str]]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def _compress_output(text: str, target: int) -> str:
    if _count_tokens(text) <= target:
        return text

    sections = text.split("\n\n")
    if len(sections) <= 6:
        return text

    keep_start = sections[:3]
    keep_end = sections[-3:]
    middle = sections[3:-3]

    while middle and _count_tokens("\n\n".join(keep_start + middle + keep_end)) > target:
        middle.pop(len(middle) // 2)

    return "\n\n".join(keep_start + middle + keep_end)


async def _groq_call(
    groq: AsyncGroq,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    retries: int = 3,
) -> str:
    """Calls Groq with exponential-backoff retry on rate-limit errors."""
    for attempt in range(retries):
        try:
            resp = await groq.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except RateLimitError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # 1 s, 2 s, 4 s


async def _stage1_segment(messages: List[Dict[str, str]]) -> str:
    conversation = _format_messages(messages)
    conversation, _ = _truncate_text(conversation, MAX_CONVERSATION_TOKENS)
    user_content = STAGE1_USER_TEMPLATE.format(conversation=conversation)

    return await _groq_call(
        _get_groq(), STAGE1_MODEL,
        [
            {"role": "system", "content": STAGE1_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        max_tokens=3000,
        temperature=0.2,
    )


async def _stage2_extract(segments: str, system_prompt: str) -> str:
    segments, _ = _truncate_text(segments, MAX_SEGMENTS_TOKENS)
    user_content = EXTRACTOR_USER_TEMPLATE.format(segments=segments)

    return await _groq_call(
        _get_groq(), STAGE2_MODEL,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=2000,
        temperature=0.2,
    )


async def _stage3_synthesize(
    code_extraction: str,
    decision_extraction: str,
    debug_extraction: str,
    intent_extraction: str,
    mode: str,
) -> str:
    user_content = SYNTHESIS_USER_TEMPLATE.format(
        code_extraction=code_extraction,
        decision_extraction=decision_extraction,
        debug_extraction=debug_extraction,
        intent_extraction=intent_extraction,
        mode=mode,
    )

    if _USE_ANTHROPIC:
        from anthropic import AsyncAnthropic  # type: ignore[import]
        client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = await client.messages.create(
            model=STAGE3_MODEL_ANTHROPIC,
            max_tokens=4096,
            system=SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text

    return await _groq_call(
        _get_groq(), STAGE3_MODEL_GROQ,
        [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        max_tokens=4096,
        temperature=0.3,
    )


async def run_pipeline(
    messages: List[Dict[str, str]],
    mode: str,
    token_target: Optional[int],
) -> Dict:
    segments = await _stage1_segment(messages)

    # Stage 2: run sequentially to avoid saturating the free-tier TPM
    # bucket with 4 simultaneous calls. Each call is ~3500 input tokens;
    # four in parallel would exceed the 6000 TPM window.
    code_result = await _stage2_extract(segments, CODE_EXTRACTOR_SYSTEM)
    decision_result = await _stage2_extract(segments, DECISION_EXTRACTOR_SYSTEM)
    debug_result = await _stage2_extract(segments, DEBUG_EXTRACTOR_SYSTEM)
    intent_result = await _stage2_extract(segments, INTENT_EXTRACTOR_SYSTEM)

    output = await _stage3_synthesize(
        code_result, decision_result, debug_result, intent_result, mode
    )

    if token_target:
        output = _compress_output(output, token_target)

    return {
        "output": output,
        "token_count": _count_tokens(output),
        "stages_used": [STAGE1_MODEL, STAGE2_MODEL, STAGE3_MODEL_ANTHROPIC if _USE_ANTHROPIC else STAGE3_MODEL_GROQ],
    }
