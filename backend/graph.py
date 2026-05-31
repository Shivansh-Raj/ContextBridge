"""
LangGraph stateful memory pipeline — ContextBridge.
Memory layers are structured JSON dicts, not free-text strings.
Merges are deterministic; the LLM only provides patches/updates.
"""

import asyncio
import json
import os
import re
from typing import TypedDict, Optional, Annotated
import operator

import tiktoken
from groq import AsyncGroq, RateLimitError
from langgraph.graph import StateGraph, END

# ── Models ────────────────────────────────────────────────────────────────────

CLASSIFIER_MODEL = "llama-3.1-8b-instant"
MEMORY_MODEL     = "llama-3.3-70b-versatile"
OUTPUT_MODEL     = "llama-3.3-70b-versatile"

CHUNK_TOKEN_TARGET = 600
MAX_MEMORY_TOKENS  = 2000   # max tokens when serialising a memory dict for an API call
MAX_CHUNK_TOKENS   = 1000
MAX_OUTPUT_TOKENS  = 2500

# ── Token utilities ───────────────────────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _truncate(text: str, max_tokens: int) -> str:
    """Keep first 25 % + last 75 % of the token budget."""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    head_n  = max_tokens // 4
    tail_n  = max_tokens - head_n
    dropped = len(tokens) - max_tokens
    return (
        _enc.decode(tokens[:head_n])
        + f"\n\n[...{dropped} tokens omitted...]\n\n"
        + _enc.decode(tokens[-tail_n:])
    )


def _mem_str(memory: dict) -> str:
    return json.dumps(memory, separators=(",", ":"), ensure_ascii=False)


def _mem_str_t(memory: dict, max_tokens: int = MAX_MEMORY_TOKENS) -> str:
    s = _mem_str(memory)
    return _truncate(s, max_tokens) if _count_tokens(s) > max_tokens else s

# ── JSON extraction ───────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict | None:
    """Robustly pull the first JSON object out of a model response."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None

# ── Empty memory schemas ──────────────────────────────────────────────────────


def _empty_intent() -> dict:
    return {
        "project_objective": "",
        "what_is_being_built": "",
        "tech_stack": [],
        "architecture_decisions": [],   # [{"decision":"","reason":"","rejected":""}]
        "implementation_reasoning": [],
        "assistant_recommendations": [], # [{"advice":"","adopted":null}]
        "debugging_insights": [],        # [{"issue":"","cause":"","fix":"","resolved":false}]
        "current_progress": {"completed": [], "in_progress": "", "next_steps": []},
        "unresolved_issues": [],
    }

# code_memory      → {"path/file.ext": {"responsibility":"","functions":[{"name":"","description":""}],
#                      "classes":[],"dependencies":[],"called_by":[],"recent_changes":""}}
# structure_memory → {"path/": "description",  "path/file.ext": "description"}
# code_detail_memory → {"path/file.ext": {"code":"","language":"","is_partial":false}}

# ── Merge helpers ─────────────────────────────────────────────────────────────


def _merge_intent(current: dict, updated: dict | None) -> dict:
    """Full-replacement merge.  The model received the full current dict and
    returned the complete updated version — just normalise types."""
    if not updated or not isinstance(updated, dict):
        return current
    base = _empty_intent()
    for key in base:
        if key not in updated:
            continue
        val = updated[key]
        expected = base[key]
        if isinstance(expected, list) and isinstance(val, str):
            base[key] = [v.strip() for v in val.split(",") if v.strip()]
        elif isinstance(expected, dict) and isinstance(val, dict):
            base[key] = {**expected, **val}
        else:
            base[key] = val
    return base


def _apply_patch(current: dict, patch: dict | None) -> dict:
    """Apply {"upsert": {...}, "delete": [...]} onto current dict.
    Guarantees no duplicates — same key means update-in-place."""
    if not patch or not isinstance(patch, dict):
        return current
    result = {**current}
    for k, v in patch.get("upsert", {}).items():
        result[k] = v
    for k in patch.get("delete", []):
        result.pop(k, None)
    return result

# ── Memory summarization ──────────────────────────────────────────────────────


async def _summarize_if_needed(memory: dict, kind: str) -> dict:
    """Compress a memory dict when it grows past MAX_MEMORY_TOKENS.
    Called just before sending memory to an updater so the updater
    always receives a token-budget-safe snapshot."""
    if _count_tokens(_mem_str(memory)) <= MAX_MEMORY_TOKENS:
        return memory

    if kind == "intent":
        system = (
            "Compress this intent memory JSON. "
            "Shorten all string values to essentials (≤15 words each). "
            "Truncate each list to its 3 most recent/important items. "
            "Keep every key. Return ONLY valid JSON, no explanation."
        )
        raw    = await _groq_call(MEMORY_MODEL, system, _mem_str_t(memory, 3500), max_tokens=2000)
        parsed = _extract_json(raw)
        return _merge_intent(memory, parsed)

    if kind == "code":
        system = (
            "Compress this code memory JSON. "
            "For each file entry, shorten 'responsibility' and 'recent_changes' to one sentence. "
            "Truncate 'functions' and 'classes' to 3 items each. "
            "Keep all file keys. Return ONLY valid JSON."
        )
        raw    = await _groq_call(MEMORY_MODEL, system, _mem_str_t(memory, 3500), max_tokens=2000)
        parsed = _extract_json(raw)
        return parsed if isinstance(parsed, dict) else memory

    if kind == "structure":
        # Pure Python — keep most recent MAX_ENTRIES paths, no API call needed
        MAX_ENTRIES = 80
        items = list(memory.items())
        return dict(items[-MAX_ENTRIES:]) if len(items) > MAX_ENTRIES else memory

    if kind == "code_detail":
        # Pure Python — truncate each file's code to first 40 lines
        result: dict = {}
        for fname, entry in memory.items():
            if not isinstance(entry, dict):
                continue
            code  = entry.get("code", "")
            lines = code.splitlines()
            if len(lines) > 40:
                code  = "\n".join(lines[:40]) + "\n# ... (truncated)"
                entry = {**entry, "code": code, "is_partial": True}
            result[fname] = entry
        return result

    return memory

# ── State ─────────────────────────────────────────────────────────────────────


class ContextBridgeState(TypedDict):
    raw_messages:       list[dict]
    mode:               str

    conversation_chunks: list[dict]
    current_index:      int
    current_categories: list[str]

    # Structured JSON memory — never free-text strings
    intent_memory:      dict
    code_memory:        dict   # keyed by file path → no duplicates possible
    structure_memory:   dict   # flat {path: description}
    code_detail_memory: dict   # keyed by file path → always latest version

    processing_log: Annotated[list[str], operator.add]
    is_complete:    bool
    output_files:   dict

# ── Groq helper ───────────────────────────────────────────────────────────────


def _get_groq() -> AsyncGroq:
    return AsyncGroq(api_key=os.environ["GROQ_API_KEY"])


async def _groq_call(
    model: str, system: str, user: str,
    max_tokens: int = 2000, retries: int = 3,
) -> str:
    groq = _get_groq()
    for attempt in range(retries):
        try:
            resp = await groq.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except RateLimitError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

# ── Node 1: Chunker ───────────────────────────────────────────────────────────


def chunker_node(state: ContextBridgeState) -> dict:
    messages = state["raw_messages"]
    chunks:        list[dict] = []
    current_lines: list[str]  = []
    current_tokens = 0

    for msg in messages:
        line = f"{msg['role'].upper()}: {msg['content']}"
        tok  = _count_tokens(line)
        if current_tokens + tok > CHUNK_TOKEN_TARGET and current_lines:
            chunks.append({"index": len(chunks), "content": "\n\n".join(current_lines)})
            current_lines, current_tokens = [], 0
        current_lines.append(line)
        current_tokens += tok

    if current_lines:
        chunks.append({"index": len(chunks), "content": "\n\n".join(current_lines)})

    return {
        "conversation_chunks": chunks,
        "current_index": 0,
        "processing_log": [f"Chunked {len(messages)} messages → {len(chunks)} chunks"],
    }

# ── Node 2: Classifier ────────────────────────────────────────────────────────


_CLASSIFIER_SYS = (
    "Classify this conversation chunk into one or more categories.\n"
    "Valid: code_update, architecture, goal_statement, debugging, folder_structure,\n"
    "       assistant_recommendation, progress_update, unresolved_issue, filler\n"
    "filler = greetings/thanks/one-word replies with no technical content.\n"
    'Return ONLY valid JSON: {"categories": ["cat1", "cat2"]}'
)


async def classifier_node(state: ContextBridgeState) -> dict:
    idx        = state["current_index"]
    total      = len(state["conversation_chunks"])
    chunk_text = _truncate(state["conversation_chunks"][idx]["content"], 800)

    raw = await _groq_call(CLASSIFIER_MODEL, _CLASSIFIER_SYS, chunk_text, max_tokens=60)
    try:
        m          = re.search(r"\{.*?\}", raw, re.DOTALL)
        categories = json.loads(m.group())["categories"] if m else ["filler"]
    except (json.JSONDecodeError, KeyError, AttributeError):
        categories = ["filler"]

    return {
        "current_categories": categories,
        "processing_log": [f"[{idx+1}/{total}] {categories}"],
    }

# ── Category trigger sets ─────────────────────────────────────────────────────

_INTENT_CATS    = frozenset({"architecture", "goal_statement", "debugging",
                              "assistant_recommendation", "progress_update", "unresolved_issue"})
_CODE_CATS      = frozenset({"code_update"})
_STRUCT_CATS    = frozenset({"folder_structure", "code_update"})
_DETAIL_CATS    = frozenset({"code_update"})

# ── Updater prompts ───────────────────────────────────────────────────────────

_INTENT_SCHEMA = (
    '{"project_objective":"","what_is_being_built":"","tech_stack":[],'
    '"architecture_decisions":[{"decision":"","reason":"","rejected":""}],'
    '"implementation_reasoning":[],'
    '"assistant_recommendations":[{"advice":"","adopted":null}],'
    '"debugging_insights":[{"issue":"","cause":"","fix":"","resolved":false}],'
    '"current_progress":{"completed":[],"in_progress":"","next_steps":[]},'
    '"unresolved_issues":[]}'
)

_INTENT_SYS = (
    "You update a structured intent memory JSON for a software project.\n"
    "Return the COMPLETE updated JSON matching this schema exactly:\n"
    f"{_INTENT_SCHEMA}\n"
    "Rules: preserve all existing entries; add new info; set resolved:true when a bug is fixed; "
    "never duplicate list entries. Return ONLY valid JSON, no explanation."
)
_INTENT_USR = "CURRENT MEMORY:\n{memory}\n\nNEW CHUNK:\n{chunk}\n\nReturn the updated intent JSON."

_CODE_SYS = (
    "You update a code memory JSON for a software project.\n"
    "Return a JSON patch — only files that need to be added or updated:\n"
    '{"upsert":{"path/file.ext":{"responsibility":"","functions":[{"name":"","description":""}],'
    '"classes":[],"dependencies":[],"called_by":[],"recent_changes":""}},"delete":[]}\n'
    "Never include files that have not changed. Return ONLY valid JSON."
)
_CODE_USR = "CURRENT CODE MEMORY:\n{memory}\n\nNEW CHUNK:\n{chunk}\n\nReturn the patch JSON."

_STRUCT_SYS = (
    "You update a project structure memory — a flat JSON dict of {path: description}.\n"
    "Return a patch: "
    '{"upsert":{"backend/auth/":"JWT module","backend/auth/auth.py":"token validation"},"delete":[]}\n'
    "Only include paths that are new or changed. Return ONLY valid JSON."
)
_STRUCT_USR = "CURRENT STRUCTURE:\n{memory}\n\nNEW CHUNK:\n{chunk}\n\nReturn the patch JSON."

_DETAIL_SYS = (
    "Extract ACTUAL CODE from this conversation chunk.\n"
    "Return a JSON patch:\n"
    '{"upsert":{"path/file.ext":{"code":"...","language":"python","is_partial":false}},"delete":[]}\n'
    "- If a file already in memory has a newer version shown, include it in upsert to replace it.\n"
    "- Set is_partial:true if only a snippet was shown.\n"
    "- If no code is visible, return {\"upsert\":{},\"delete\":[]}.\n"
    "Return ONLY valid JSON."
)
_DETAIL_USR = "CURRENT CODE STATE:\n{memory}\n\nNEW CHUNK:\n{chunk}\n\nReturn the patch JSON."

# ── Node 3: Memory Dispatcher ─────────────────────────────────────────────────


async def memory_dispatcher_node(state: ContextBridgeState) -> dict:
    categories = frozenset(state.get("current_categories", []))
    idx        = state["current_index"]

    if "filler" in categories or not categories:
        return {"processing_log": [f"[{idx}] skipped (filler)"]}

    chunk   = _truncate(state["conversation_chunks"][idx]["content"], MAX_CHUNK_TOKENS)
    updates: dict = {}

    if categories & _INTENT_CATS:
        current = await _summarize_if_needed(state["intent_memory"], "intent")
        raw     = await _groq_call(
            MEMORY_MODEL, _INTENT_SYS,
            _INTENT_USR.format(memory=_mem_str_t(current), chunk=chunk),
            max_tokens=2000,
        )
        updates["intent_memory"] = _merge_intent(current, _extract_json(raw))

    if categories & _CODE_CATS:
        current = await _summarize_if_needed(state["code_memory"], "code")
        raw     = await _groq_call(
            MEMORY_MODEL, _CODE_SYS,
            _CODE_USR.format(memory=_mem_str_t(current), chunk=chunk),
            max_tokens=2000,
        )
        updates["code_memory"] = _apply_patch(current, _extract_json(raw))

    if categories & _STRUCT_CATS:
        current = await _summarize_if_needed(state["structure_memory"], "structure")
        raw     = await _groq_call(
            CLASSIFIER_MODEL, _STRUCT_SYS,
            _STRUCT_USR.format(memory=_mem_str_t(current), chunk=chunk),
            max_tokens=800,
        )
        updates["structure_memory"] = _apply_patch(current, _extract_json(raw))

    if state.get("mode") == "detailed" and categories & _DETAIL_CATS:
        current = await _summarize_if_needed(state["code_detail_memory"], "code_detail")
        raw     = await _groq_call(
            MEMORY_MODEL, _DETAIL_SYS,
            _DETAIL_USR.format(memory=_mem_str_t(current), chunk=chunk),
            max_tokens=2500,
        )
        updates["code_detail_memory"] = _apply_patch(current, _extract_json(raw))

    return updates

# ── Node 4: Loop Controller ───────────────────────────────────────────────────


def loop_controller_node(state: ContextBridgeState) -> dict:
    next_idx = state["current_index"] + 1
    done     = next_idx >= len(state["conversation_chunks"])
    return {
        "current_index": next_idx,
        "is_complete":   done,
        "processing_log": (
            ["All chunks processed — generating output"] if done
            else [f"→ chunk {next_idx}"]
        ),
    }


def _route(state: ContextBridgeState) -> str:
    return "output_generator" if state["is_complete"] else "classifier"

# ── Python renderers (no AI call) ─────────────────────────────────────────────


def _render_structure(structure: dict) -> str:
    if not structure:
        return "No project structure captured."
    lines = []
    for path in sorted(structure.keys()):
        depth  = max(0, path.rstrip("/").count("/"))
        name   = path.rstrip("/").split("/")[-1] + ("/" if path.endswith("/") else "")
        desc   = structure[path]
        indent = "│   " * depth
        lines.append(f"{indent}├── {name:<32} {desc}")
    return "```\n" + "\n".join(lines) + "\n```"


def _render_code_detail(code_detail: dict) -> str:
    if not code_detail:
        return ""
    parts = []
    for filepath in sorted(code_detail.keys()):
        entry = code_detail[filepath]
        if not isinstance(entry, dict):
            continue
        lang    = entry.get("language", "")
        code    = entry.get("code", "").rstrip()
        partial = entry.get("is_partial", False)
        parts.append(f"## {filepath}")
        if partial:
            parts.append("> **Note:** Only the portion shown in the conversation was captured.\n")
        parts.append(f"```{lang}\n{code}\n```\n")
    return "\n".join(parts)

# ── Output Generator prompts ──────────────────────────────────────────────────

_INTENT_FMT_SYS = (
    "Convert this intent memory JSON into clean technical markdown. "
    "Write in clear prose for an AI assistant to read. Skip empty sections."
)
_INTENT_FMT_USR = (
    "{memory}\n\n"
    "Format under these headings (omit any with no content):\n"
    "## Project Objective\n## What Is Being Built\n## Tech Stack\n"
    "## Architecture and Technology Decisions\n## Implementation Reasoning\n"
    "## Assistant Recommendations\n## Debugging Insights\n"
    "## Current Progress\n## Unresolved Issues"
)

_CODE_FMT_SYS = (
    "Convert this code memory JSON into clean technical markdown. "
    "Write for an AI assistant. One section per file."
)
_CODE_FMT_USR = (
    "{memory}\n\n"
    "For each file:\n"
    "### path/to/file.ext\n"
    "**Responsibility:** one line\n"
    "**Key behavior:** bullet list\n"
    "**Dependencies:** list\n"
    "**Recent changes:** if any"
)

# ── Node 5: Output Generator ──────────────────────────────────────────────────


async def output_generator_node(state: ContextBridgeState) -> dict:
    intent_doc = await _groq_call(
        OUTPUT_MODEL, _INTENT_FMT_SYS,
        _INTENT_FMT_USR.format(memory=_mem_str_t(state["intent_memory"], MAX_OUTPUT_TOKENS)),
        max_tokens=2000,
    )
    code_doc = await _groq_call(
        OUTPUT_MODEL, _CODE_FMT_SYS,
        _CODE_FMT_USR.format(memory=_mem_str_t(state["code_memory"], MAX_OUTPUT_TOKENS)),
        max_tokens=2000,
    )

    # Structure and code-detail are rendered in pure Python — no extra API calls
    structure_doc   = _render_structure(state["structure_memory"])
    code_detail_doc = (
        _render_code_detail(state["code_detail_memory"])
        if state.get("mode") == "detailed" and state.get("code_detail_memory")
        else ""
    )

    files: dict = {"intent": intent_doc, "code": code_doc, "structure": structure_doc}
    if code_detail_doc:
        files["code_detail"] = code_detail_doc

    return {"output_files": files, "processing_log": ["Output generated"]}

# ── Graph assembly ────────────────────────────────────────────────────────────


def _build_graph():
    g = StateGraph(ContextBridgeState)
    g.add_node("chunker",           chunker_node)
    g.add_node("classifier",        classifier_node)
    g.add_node("memory_dispatcher", memory_dispatcher_node)
    g.add_node("loop_controller",   loop_controller_node)
    g.add_node("output_generator",  output_generator_node)
    g.set_entry_point("chunker")
    g.add_edge("chunker",           "classifier")
    g.add_edge("classifier",        "memory_dispatcher")
    g.add_edge("memory_dispatcher", "loop_controller")
    g.add_conditional_edges(
        "loop_controller", _route,
        {"classifier": "classifier", "output_generator": "output_generator"},
    )
    g.add_edge("output_generator", END)
    return g.compile()


_graph = _build_graph()

# ── Public entry point ────────────────────────────────────────────────────────


async def run_graph(
    messages: list[dict], mode: str, token_target: Optional[int],
) -> dict:
    initial: ContextBridgeState = {
        "raw_messages":       messages,
        "mode":               mode,
        "conversation_chunks": [],
        "current_index":      0,
        "current_categories": [],
        "intent_memory":      _empty_intent(),
        "code_memory":        {},
        "structure_memory":   {},
        "code_detail_memory": {},
        "processing_log":     [],
        "is_complete":        False,
        "output_files":       {},
    }

    final = await _graph.ainvoke(initial)
    files = final.get("output_files", {})

    sections = [
        ("--- FILE 1: PROJECT INTENT & CONTEXT ---", files.get("intent",      "")),
        ("--- FILE 2: CODE UNDERSTANDING ---",       files.get("code",        "")),
        ("--- FILE 3: PROJECT STRUCTURE ---",        files.get("structure",   "")),
        ("--- FILE 4: DETAILED CODE ---",            files.get("code_detail", "")),
    ]
    combined = "\n\n".join(f"{h}\n\n{b}" for h, b in sections if b.strip())

    if token_target and _count_tokens(combined) > token_target:
        toks     = _enc.encode(combined)
        combined = _enc.decode(toks[:token_target]) + "\n\n[... truncated to token target ...]"

    return {
        "output":           combined,
        "output_files":     files,
        "token_count":      _count_tokens(combined),
        "stages_used":      [CLASSIFIER_MODEL, MEMORY_MODEL, OUTPUT_MODEL],
        "chunks_processed": len(final.get("conversation_chunks", [])),
        "processing_log":   final.get("processing_log", []),
    }
