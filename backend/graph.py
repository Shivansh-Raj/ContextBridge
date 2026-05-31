"""
LangGraph stateful memory pipeline — ContextBridge.

Key design decisions:
  - All memory layers are typed JSON dicts (not strings) — duplicates impossible by key
  - Updaters receive patches (small output), never full replacements — no mid-JSON truncation
  - Summarisation triggers late (4500 tokens) and compresses lightly
  - Output generation is batched — large code_memory produces complete docs, not truncated ones
  - Chunk size adapts to conversation length — fewer, richer chunks for long inputs
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

# ── Token budgets ─────────────────────────────────────────────────────────────
# Groq free tier: ~6000 TPM per model.
# Each call budget: system (~400) + memory (MAX_MEMORY_TOKENS) + chunk (MAX_CHUNK_TOKENS)
# + response (max_tokens) must stay under ~5800 tokens.

MAX_MEMORY_TOKENS  = 3000   # existing memory sent to updaters
MAX_CHUNK_TOKENS   = 1000   # chunk content sent to updaters
MAX_OUTPUT_BATCH   = 3500   # memory per output-generation batch call
SUMMARIZE_AT       = 4500   # trigger compression when memory exceeds this

# Chunk level settings: base token target + max chunk count cap
# max_chunks prevents runaway API call counts on very large conversations.
_CHUNK_LEVEL = {
    "low":    {"base": 1800, "max_chunks": 12},   # fast, coarse
    "medium": {"base": 800,  "max_chunks": 25},   # balanced (default)
    "high":   {"base": 350,  "max_chunks": 50},   # thorough, slowest
}


def _get_chunk_target(chunk_level: str, total_tokens: int) -> int:
    cfg        = _CHUNK_LEVEL.get(chunk_level, _CHUNK_LEVEL["medium"])
    base       = cfg["base"]
    max_chunks = cfg["max_chunks"]
    # Enforce the cap: if the conversation is large, raise the target so we
    # don't exceed max_chunks chunks regardless of the chosen level.
    return max(base, total_tokens // max_chunks)

# ── Utilities ─────────────────────────────────────────────────────────────────

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


def _batch_dict(d: dict, max_tokens: int = MAX_OUTPUT_BATCH) -> list[dict]:
    """Split a dict into batches where each batch serialises to ≤ max_tokens.
    This lets output_generator produce complete docs without truncation."""
    batches:  list[dict] = []
    current:  dict       = {}
    cur_tok              = 0
    for key, val in d.items():
        entry_tok = _count_tokens(json.dumps({key: val}, separators=(",", ":")))
        if cur_tok + entry_tok > max_tokens and current:
            batches.append(current)
            current, cur_tok = {}, 0
        current[key] = val
        cur_tok += entry_tok
    if current:
        batches.append(current)
    return batches or [{}]

# ── JSON extraction ───────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict | None:
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

# ── Empty memory schema ───────────────────────────────────────────────────────


def _empty_intent() -> dict:
    return {
        "project_objective":      "",
        "what_is_being_built":    "",
        "tech_stack":             [],
        "architecture_decisions": [],   # [{"decision":"","reason":"","rejected":""}]
        "implementation_reasoning": [],
        "assistant_recommendations": [], # [{"advice":"","adopted":null}]
        "debugging_insights":     [],   # [{"issue":"","cause":"","fix":"","resolved":false}]
        "current_progress": {"completed": [], "in_progress": "", "next_steps": []},
        "unresolved_issues":      [],
    }

# code_memory      → {"path/file.ext": {"responsibility":"","functions":[...],"classes":[],
#                      "dependencies":[],"called_by":[],"recent_changes":""}}
# structure_memory → {"path/": "description", "path/file.ext": "description"}
# code_detail_memory → {"path/file.ext": {"code":"","language":"","is_partial":false}}

# ── Merge helpers ─────────────────────────────────────────────────────────────


def _merge_intent_patch(current: dict, patch: dict | None) -> dict:
    """Apply an intent PATCH onto the current intent dict.
    Patch schema: {set:{...}, append:{list_key:[...]}, resolve:[], current_progress:{...}}
    This never truncates — the LLM only returns small deltas, not the full replacement."""
    if not patch or not isinstance(patch, dict):
        return current
    result = {**current}

    # Scalar field updates
    for key, val in patch.get("set", {}).items():
        if key in result and val:
            result[key] = val

    # List appends — only add genuinely new entries
    for key, items in patch.get("append", {}).items():
        if key in result and isinstance(result[key], list) and isinstance(items, list):
            result[key] = result[key] + items

    # Mark issues as resolved
    for issue_text in patch.get("resolve", []):
        for insight in result.get("debugging_insights", []):
            if insight.get("issue") == issue_text:
                insight["resolved"] = True
        ui = result.get("unresolved_issues", [])
        if issue_text in ui:
            ui.remove(issue_text)

    # Progress update (merge sub-keys)
    if "current_progress" in patch and isinstance(patch["current_progress"], dict):
        cp_patch = patch["current_progress"]
        cp       = result.setdefault("current_progress",
                                     {"completed": [], "in_progress": "", "next_steps": []})
        if cp_patch.get("in_progress"):
            cp["in_progress"] = cp_patch["in_progress"]
        cp["completed"]  = cp.get("completed",  []) + cp_patch.get("completed",  [])
        cp["next_steps"] = cp.get("next_steps", []) + cp_patch.get("next_steps", [])

    return result


def _apply_patch(current: dict, patch: dict | None) -> dict:
    """Apply {"upsert": {...}, "delete": [...]} onto current dict.
    File-path keys guarantee no duplicate entries."""
    if not patch or not isinstance(patch, dict):
        return current
    result = {**current}
    for k, v in patch.get("upsert", {}).items():
        result[k] = v
    for k in patch.get("delete", []):
        result.pop(k, None)
    return result

# ── Memory summarisation ──────────────────────────────────────────────────────


async def _summarize_if_needed(memory: dict, kind: str) -> dict:
    """Compress memory only when it exceeds SUMMARIZE_AT (4500 tokens).
    Compression is lighter than before — more items kept, more context preserved."""
    if _count_tokens(_mem_str(memory)) <= SUMMARIZE_AT:
        return memory

    if kind == "intent":
        system = (
            "Compress this intent memory JSON. "
            "Shorten string values to ≤25 words each. "
            "Truncate each list to its 8 most recent/important items. "
            "Keep EVERY key — never delete a field. "
            "Return ONLY valid JSON, no explanation."
        )
        raw    = await _groq_call(MEMORY_MODEL, system, _mem_str_t(memory, 4000), max_tokens=2500)
        parsed = _extract_json(raw)
        return _merge_intent_patch(memory, {"set": parsed}) if parsed else memory

    if kind == "code":
        system = (
            "Compress this code memory JSON. "
            "For each file entry shorten 'responsibility' to 1 sentence and "
            "'recent_changes' to 1 sentence. "
            "Truncate 'functions' and 'classes' lists to 6 items each. "
            "KEEP ALL FILE ENTRIES — do not remove any files. "
            "Return ONLY valid JSON with the same structure."
        )
        raw    = await _groq_call(MEMORY_MODEL, system, _mem_str_t(memory, 4000), max_tokens=2500)
        parsed = _extract_json(raw)
        return parsed if isinstance(parsed, dict) else memory

    if kind == "structure":
        # Pure Python — keep most recent 150 entries
        items = list(memory.items())
        return dict(items[-150:]) if len(items) > 150 else memory

    if kind == "code_detail":
        # Pure Python — truncate each file to first 80 lines
        result: dict = {}
        for fname, entry in memory.items():
            if not isinstance(entry, dict):
                continue
            code  = entry.get("code", "")
            lines = code.splitlines()
            if len(lines) > 80:
                code  = "\n".join(lines[:80]) + "\n# ... (truncated beyond 80 lines)"
                entry = {**entry, "code": code, "is_partial": True}
            result[fname] = entry
        return result

    return memory

# ── State ─────────────────────────────────────────────────────────────────────


class ContextBridgeState(TypedDict):
    raw_messages:        list[dict]
    mode:                str
    chunk_level:         str   # "low" | "medium" | "high"

    conversation_chunks: list[dict]
    current_index:       int
    current_categories:  list[str]

    intent_memory:       dict   # patched via _merge_intent_patch
    code_memory:         dict   # patched via _apply_patch, keyed by file path
    structure_memory:    dict   # patched via _apply_patch, flat {path: desc}
    code_detail_memory:  dict   # patched via _apply_patch, keyed by file path

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
    messages     = state["raw_messages"]
    chunk_level  = state.get("chunk_level", "medium")

    total_tokens = sum(
        _count_tokens(f"{m['role'].upper()}: {m['content']}") for m in messages
    )
    chunk_target = _get_chunk_target(chunk_level, total_tokens)

    chunks:        list[dict] = []
    current_lines: list[str]  = []
    current_tokens = 0

    for msg in messages:
        line = f"{msg['role'].upper()}: {msg['content']}"
        tok  = _count_tokens(line)
        if current_tokens + tok > chunk_target and current_lines:
            chunks.append({"index": len(chunks), "content": "\n\n".join(current_lines)})
            current_lines, current_tokens = [], 0
        current_lines.append(line)
        current_tokens += tok

    if current_lines:
        chunks.append({"index": len(chunks), "content": "\n\n".join(current_lines)})

    return {
        "conversation_chunks": chunks,
        "current_index": 0,
        "processing_log": [
            f"Chunked {len(messages)} msgs → {len(chunks)} chunks "
            f"(level={chunk_level}, ~{chunk_target} tok/chunk, conv={total_tokens} tok)"
        ],
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

_INTENT_CATS = frozenset({
    "architecture", "goal_statement", "debugging",
    "assistant_recommendation", "progress_update", "unresolved_issue",
})
_CODE_CATS    = frozenset({"code_update"})
_STRUCT_CATS  = frozenset({"folder_structure", "code_update"})
_DETAIL_CATS  = frozenset({"code_update"})

# ── Updater prompts ───────────────────────────────────────────────────────────

# Intent uses PATCH approach — model returns only the delta, never the full JSON.
# This means the output is always small (~300–800 tokens) regardless of memory size,
# so we never risk the model truncating a large JSON response mid-object.
_INTENT_PATCH_SCHEMA = (
    '{"set":{"project_objective":"only if changed","what_is_being_built":"only if changed"},'
    '"append":{"architecture_decisions":[{"decision":"","reason":"","rejected":""}],'
    '"implementation_reasoning":[""],'
    '"assistant_recommendations":[{"advice":"","adopted":null}],'
    '"debugging_insights":[{"issue":"","cause":"","fix":"","resolved":false}],'
    '"unresolved_issues":[""],"tech_stack":["new tech only"]},'
    '"resolve":["issue text that was fixed in this chunk"],'
    '"current_progress":{"in_progress":"","completed":[""],"next_steps":[""]}}'
)

_INTENT_SYS = (
    "You update an intent memory for a software project using PATCH semantics.\n"
    "Return ONLY the delta — what is NEW or CHANGED, not everything already in memory.\n"
    f"Patch schema:\n{_INTENT_PATCH_SCHEMA}\n"
    "Rules:\n"
    "  'set'    — only if the field value actually changed\n"
    "  'append' — only items not already present in memory\n"
    "  'resolve'— list issue texts that were fixed in this chunk\n"
    "  'current_progress' — only if progress changed\n"
    "  If nothing relevant, return {}\n"
    "Return ONLY valid JSON, no explanation."
)
_INTENT_USR = (
    "CURRENT INTENT MEMORY:\n{memory}\n\n"
    "NEW CHUNK:\n{chunk}\n\n"
    "Return the patch JSON."
)

_CODE_SYS = (
    "You update a code memory JSON. Return ONLY files that are new or changed.\n"
    '{"upsert":{"path/file.ext":{"responsibility":"","functions":[{"name":"","description":""}],'
    '"classes":[],"dependencies":[],"called_by":[],"recent_changes":""}},"delete":[]}\n'
    "Never repeat files that have not changed. Return ONLY valid JSON."
)
_CODE_USR = "CURRENT CODE MEMORY:\n{memory}\n\nNEW CHUNK:\n{chunk}\n\nReturn the patch JSON."

_STRUCT_SYS = (
    "You update a project structure memory — a flat JSON dict {path: description}.\n"
    'Return a patch: {"upsert":{"backend/auth/":"desc","backend/auth/auth.py":"desc"},"delete":[]}\n'
    "Only include paths that are new or changed. Return ONLY valid JSON."
)
_STRUCT_USR = "CURRENT STRUCTURE:\n{memory}\n\nNEW CHUNK:\n{chunk}\n\nReturn the patch JSON."

_DETAIL_SYS = (
    "Extract ACTUAL CODE from this chunk.\n"
    'Return: {"upsert":{"path/file.ext":{"code":"full code here","language":"python","is_partial":false}},"delete":[]}\n'
    "- Replace a file's entry if a newer version is shown.\n"
    "- is_partial:true if only a snippet was shown.\n"
    '- Return {"upsert":{},"delete":[]} if no code is present.\n'
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
            max_tokens=1000,   # patch is small — no risk of mid-JSON truncation
        )
        updates["intent_memory"] = _merge_intent_patch(current, _extract_json(raw))

    if categories & _CODE_CATS:
        current = await _summarize_if_needed(state["code_memory"], "code")
        raw     = await _groq_call(
            MEMORY_MODEL, _CODE_SYS,
            _CODE_USR.format(memory=_mem_str_t(current), chunk=chunk),
            max_tokens=1500,   # patch with 2–5 changed files
        )
        updates["code_memory"] = _apply_patch(current, _extract_json(raw))

    if categories & _STRUCT_CATS:
        current = await _summarize_if_needed(state["structure_memory"], "structure")
        raw     = await _groq_call(
            CLASSIFIER_MODEL, _STRUCT_SYS,
            _STRUCT_USR.format(memory=_mem_str_t(current), chunk=chunk),
            max_tokens=600,
        )
        updates["structure_memory"] = _apply_patch(current, _extract_json(raw))

    if state.get("mode") == "detailed" and categories & _DETAIL_CATS:
        current = await _summarize_if_needed(state["code_detail_memory"], "code_detail")
        raw     = await _groq_call(
            MEMORY_MODEL, _DETAIL_SYS,
            _DETAIL_USR.format(memory=_mem_str_t(current), chunk=chunk),
            max_tokens=2500,   # code can be long
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

# ── Python renderers ──────────────────────────────────────────────────────────


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

# ── Output Generator helpers ──────────────────────────────────────────────────

_INTENT_FMT_SYS = (
    "Convert this intent memory JSON into clean technical markdown. "
    "Include EVERY item — do not skip or summarise entries. "
    "Write in clear prose for an AI assistant to read."
)
_INTENT_FMT_USR = (
    "{memory}\n\n"
    "Format under these headings (omit only truly empty sections):\n"
    "## Project Objective\n## What Is Being Built\n## Tech Stack\n"
    "## Architecture and Technology Decisions\n## Implementation Reasoning\n"
    "## Assistant Recommendations\n## Debugging Insights\n"
    "## Current Progress\n## Unresolved Issues"
)

_CODE_FMT_SYS = (
    "Convert this code memory JSON into clean technical markdown. "
    "Include EVERY file entry — do not skip any. "
    "Write for an AI assistant. One section per file."
)
_CODE_FMT_USR = (
    "{memory}\n\n"
    "For each file:\n"
    "### path/to/file.ext\n"
    "**Responsibility:** one line\n"
    "**Key behavior:** bullet list (include all functions/classes)\n"
    "**Dependencies:** list\n"
    "**Recent changes:** if any"
)


async def _generate_batched_doc(
    memory: dict, system: str, user_template: str, label: str,
) -> str:
    """Generate an output document in batches so large memories are never truncated.
    Each batch processes a subset of the dict keys and the results are concatenated."""
    if not memory:
        return f"No {label} information captured."

    batches = _batch_dict(memory, MAX_OUTPUT_BATCH)

    if len(batches) == 1:
        return await _groq_call(
            OUTPUT_MODEL, system,
            user_template.format(memory=_mem_str(batches[0])),
            max_tokens=2000,
        )

    parts: list[str] = []
    for i, batch in enumerate(batches, 1):
        suffix = f"\n\n(This is batch {i} of {len(batches)}. Document all entries in this batch.)"
        part   = await _groq_call(
            OUTPUT_MODEL, system,
            user_template.format(memory=_mem_str(batch)) + suffix,
            max_tokens=2000,
        )
        parts.append(part)
    return "\n\n".join(parts)

# ── Node 5: Output Generator ──────────────────────────────────────────────────


async def output_generator_node(state: ContextBridgeState) -> dict:
    # Intent — pass full memory (no truncation); format as prose
    intent_doc = await _groq_call(
        OUTPUT_MODEL, _INTENT_FMT_SYS,
        _INTENT_FMT_USR.format(memory=_mem_str(state["intent_memory"])),
        max_tokens=2000,
    )

    # Code — batched so ALL files appear in the output regardless of count
    code_doc = await _generate_batched_doc(
        state["code_memory"], _CODE_FMT_SYS, _CODE_FMT_USR, "code",
    )

    # Structure and code-detail — pure Python, zero API calls, deterministic
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
    chunk_level: str = "medium",
) -> dict:
    initial: ContextBridgeState = {
        "raw_messages":       messages,
        "mode":               mode,
        "chunk_level":        chunk_level,
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
