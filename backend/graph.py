"""
LangGraph-based stateful memory pipeline for ContextBridge.

Architecture:
  chunker → classifier → memory_dispatcher → loop_controller
                ↑___________________________________|
  (loop until all chunks processed)
  loop_controller → output_generator → END
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

# ── Models ───────────────────────────────────────────────────────────────────

CLASSIFIER_MODEL = "llama-3.1-8b-instant"    # fast + cheap for classification
MEMORY_MODEL = "llama-3.3-70b-versatile"      # higher reasoning for memory updates
OUTPUT_MODEL = "llama-3.3-70b-versatile"      # final document formatting

# Token budgets — Groq free tier is ~6000 TPM per model.
# Keep each request well under that including system prompt overhead.
CHUNK_TOKEN_TARGET = 600     # messages grouped into ~this many tokens per chunk
MAX_MEMORY_TOKENS = 2500     # existing memory sent to each updater call
MAX_CHUNK_TOKENS = 1000      # chunk content sent to each updater call
MAX_OUTPUT_TOKENS = 3000     # memory sent to output formatter calls

# ── Utilities ────────────────────────────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _truncate(text: str, max_tokens: int) -> str:
    """Keeps first 25 % + last 75 % of the token budget."""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    head_n = max_tokens // 4
    tail_n = max_tokens - head_n
    dropped = len(tokens) - max_tokens
    return (
        _enc.decode(tokens[:head_n])
        + f"\n\n[...{dropped} tokens from the middle omitted...]\n\n"
        + _enc.decode(tokens[-tail_n:])
    )

# ── State ────────────────────────────────────────────────────────────────────


class ContextBridgeState(TypedDict):
    # Set once at the start, never changed
    raw_messages: list[dict]
    mode: str

    # Set by chunker, read by classifier/dispatcher
    conversation_chunks: list[dict]   # [{"index": int, "content": str}]
    current_index: int

    # Set by classifier, read by dispatcher — ephemeral per iteration
    current_categories: list[str]

    # Memory layers — incrementally patched, never reset
    intent_memory: str
    code_memory: str
    structure_memory: str
    code_detail_memory: str   # actual code blocks per file (detailed mode only)

    # Use operator.add so each node appends without overwriting others' logs
    processing_log: Annotated[list[str], operator.add]

    is_complete: bool
    output_files: dict   # {"intent": str, "code": str, "structure": str}

# ── Groq helper ──────────────────────────────────────────────────────────────


def _get_groq() -> AsyncGroq:
    return AsyncGroq(api_key=os.environ["GROQ_API_KEY"])


async def _groq_call(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2000,
    retries: int = 3,
) -> str:
    groq = _get_groq()
    for attempt in range(retries):
        try:
            resp = await groq.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except RateLimitError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)   # 1 s, 2 s, 4 s

# ── Node 1: Chunker ───────────────────────────────────────────────────────────


def chunker_node(state: ContextBridgeState) -> dict:
    """Groups consecutive messages into token-bounded chunks."""
    messages = state["raw_messages"]
    chunks: list[dict] = []
    current_lines: list[str] = []
    current_tokens = 0

    for msg in messages:
        line = f"{msg['role'].upper()}: {msg['content']}"
        msg_tokens = _count_tokens(line)

        if current_tokens + msg_tokens > CHUNK_TOKEN_TARGET and current_lines:
            chunks.append({"index": len(chunks), "content": "\n\n".join(current_lines)})
            current_lines = []
            current_tokens = 0

        current_lines.append(line)
        current_tokens += msg_tokens

    if current_lines:
        chunks.append({"index": len(chunks), "content": "\n\n".join(current_lines)})

    return {
        "conversation_chunks": chunks,
        "current_index": 0,
        "processing_log": [f"Chunked {len(messages)} messages → {len(chunks)} chunks"],
    }

# ── Node 2: Classifier ────────────────────────────────────────────────────────


_CLASSIFIER_SYS = (
    'Classify the conversation chunk into one or more categories.\n'
    'Valid categories: code_update, architecture, goal_statement, debugging,\n'
    'folder_structure, assistant_recommendation, progress_update,\n'
    'unresolved_issue, filler\n\n'
    'filler = greetings / thanks / one-word replies with no technical content.\n'
    'Return ONLY valid JSON: {"categories": ["cat1", "cat2"]}'
)


async def classifier_node(state: ContextBridgeState) -> dict:
    idx = state["current_index"]
    chunk_text = _truncate(state["conversation_chunks"][idx]["content"], 800)
    total = len(state["conversation_chunks"])

    raw = await _groq_call(
        CLASSIFIER_MODEL,
        _CLASSIFIER_SYS,
        chunk_text,
        max_tokens=60,
    )

    try:
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        categories: list[str] = json.loads(m.group())["categories"] if m else ["filler"]
    except (json.JSONDecodeError, KeyError, AttributeError):
        categories = ["filler"]

    return {
        "current_categories": categories,
        "processing_log": [f"[{idx+1}/{total}] {categories}"],
    }

# ── Memory category sets ──────────────────────────────────────────────────────

_INTENT_CATS = frozenset({
    "architecture", "goal_statement", "debugging",
    "assistant_recommendation", "progress_update", "unresolved_issue",
})
_CODE_CATS = frozenset({"code_update"})
_STRUCTURE_CATS = frozenset({"folder_structure", "code_update"})
_CODE_DETAIL_CATS = frozenset({"code_update"})   # same trigger, different memory layer

# ── Memory updater prompts ────────────────────────────────────────────────────

_INTENT_SYS = (
    "You maintain a living intent/context memory document for a software project.\n"
    "Patch the existing memory based on the new chunk.\n"
    "Rules: add new information; update changed information; mark open issues "
    "[UNRESOLVED] and remove the marker when resolved; never duplicate.\n"
    "Return ONLY the updated memory text, no preamble."
)
_INTENT_USR = (
    "CURRENT INTENT MEMORY:\n{memory}\n\n"
    "NEW CHUNK:\n{chunk}\n\n"
    "Sections to maintain (skip empty ones):\n"
    "Project Objective | Architecture & Technology Decisions | "
    "Implementation Reasoning | Debugging Insights | Current Progress | Unresolved Issues"
)

_CODE_SYS = (
    "You maintain a living code understanding document.\n"
    "Rules: add entries for new files; UPDATE existing entries for known files — "
    "never create duplicate entries for the same file; describe behavior "
    "semantically, never store raw code.\n"
    "Return ONLY the updated memory text, no preamble."
)
_CODE_USR = (
    "CURRENT CODE MEMORY:\n{memory}\n\n"
    "NEW CHUNK:\n{chunk}\n\n"
    "Per-file format:\n"
    "  filename.ext — primary responsibility\n"
    "  - key functions/classes\n"
    "  - dependencies\n"
    "  - recent changes (if any in this chunk)"
)

_STRUCTURE_SYS = (
    "You maintain a project folder/file layout document.\n"
    "Rules: add new folders and files; move files that were relocated; "
    "keep one-line descriptions per entry.\n"
    "Return ONLY the updated structure as a folder tree, no preamble."
)
_STRUCTURE_USR = "CURRENT STRUCTURE MEMORY:\n{memory}\n\nNEW CHUNK:\n{chunk}\n\nUpdate the folder tree."

_CODE_DETAIL_SYS = (
    "You maintain a document of ACTUAL CODE for each file in a software project.\n"
    "Rules:\n"
    "- If a file's complete or partial code appears in the chunk, capture it\n"
    "- If the file already exists in memory and a NEWER version is shown, REPLACE the old version — never duplicate\n"
    "- If only a snippet or diff is shown, keep the best known version and note the change inline\n"
    "- Never add commentary outside code blocks\n"
    "Return ONLY the updated code document, no preamble."
)
_CODE_DETAIL_USR = (
    "CURRENT CODE STATE:\n{memory}\n\n"
    "NEW CHUNK:\n{chunk}\n\n"
    "Extract any code shown. For each file use:\n"
    "### path/to/filename.ext\n"
    "```language\n[code here]\n```\n"
    "If the file already exists in the current code state, replace that entry with the latest version shown."
)

# ── Node 3: Memory Dispatcher ─────────────────────────────────────────────────


async def memory_dispatcher_node(state: ContextBridgeState) -> dict:
    """Routes the current chunk to all relevant memory updaters (sequentially)."""
    categories = frozenset(state.get("current_categories", []))
    idx = state["current_index"]

    if "filler" in categories or not categories:
        return {"processing_log": [f"[{idx}] skipped (filler)"]}

    chunk = _truncate(state["conversation_chunks"][idx]["content"], MAX_CHUNK_TOKENS)
    updates: dict = {}

    if categories & _INTENT_CATS:
        mem = _truncate(state["intent_memory"] or "(empty)", MAX_MEMORY_TOKENS)
        updates["intent_memory"] = await _groq_call(
            MEMORY_MODEL, _INTENT_SYS,
            _INTENT_USR.format(memory=mem, chunk=chunk),
            max_tokens=2000,
        )

    if categories & _CODE_CATS:
        mem = _truncate(state["code_memory"] or "(empty)", MAX_MEMORY_TOKENS)
        updates["code_memory"] = await _groq_call(
            MEMORY_MODEL, _CODE_SYS,
            _CODE_USR.format(memory=mem, chunk=chunk),
            max_tokens=2000,
        )

    if categories & _STRUCTURE_CATS:
        mem = _truncate(state["structure_memory"] or "(empty)", MAX_MEMORY_TOKENS)
        updates["structure_memory"] = await _groq_call(
            CLASSIFIER_MODEL, _STRUCTURE_SYS,   # lighter model — simpler task
            _STRUCTURE_USR.format(memory=mem, chunk=chunk),
            max_tokens=1500,
        )

    # Code detail: only in detailed mode — stores actual code, not descriptions
    if state.get("mode") == "detailed" and categories & _CODE_DETAIL_CATS:
        mem = _truncate(state["code_detail_memory"] or "(empty)", MAX_MEMORY_TOKENS)
        updates["code_detail_memory"] = await _groq_call(
            MEMORY_MODEL, _CODE_DETAIL_SYS,
            _CODE_DETAIL_USR.format(memory=mem, chunk=chunk),
            max_tokens=2500,
        )

    return updates

# ── Node 4: Loop Controller ───────────────────────────────────────────────────


def loop_controller_node(state: ContextBridgeState) -> dict:
    next_idx = state["current_index"] + 1
    done = next_idx >= len(state["conversation_chunks"])
    return {
        "current_index": next_idx,
        "is_complete": done,
        "processing_log": (
            ["All chunks processed — generating output"] if done
            else [f"Advancing to chunk {next_idx}"]
        ),
    }


def _route(state: ContextBridgeState) -> str:
    return "output_generator" if state["is_complete"] else "classifier"

# ── Node 5: Output Generator ──────────────────────────────────────────────────

_INTENT_FMT_SYS = (
    "Format the provided memory as clean markdown. "
    "This will be read by an AI assistant to orient itself before continuing the project. "
    "Write in clear technical prose. Skip sections that have no content."
)
_INTENT_FMT_USR = (
    "{memory}\n\n"
    "Format under these sections:\n"
    "## Project Objective\n## What Is Being Built\n"
    "## Architecture and Technology Decisions\n## Implementation Reasoning\n"
    "## Assistant Recommendations\n## Debugging Insights\n"
    "## Current Progress\n## Unresolved Issues"
)

_CODE_FMT_SYS = (
    "Format the provided code memory as clean markdown. "
    "This will be read by an AI assistant. Write in clear technical language."
)
_CODE_FMT_USR = (
    "{memory}\n\n"
    "For each file:\n"
    "### filename.ext\n"
    "**Responsibility:** one-line summary\n"
    "**Key behavior:** bullet list\n"
    "**Dependencies:** what imports or calls this\n"
    "**Recent changes:** if any"
)

_STRUCTURE_FMT_SYS = (
    "Format the provided structure memory as clean markdown. "
    "This will be read by an AI assistant."
)
_STRUCTURE_FMT_USR = (
    "{memory}\n\n"
    "Produce:\n"
    "1. An ASCII folder tree with one-line descriptions per entry\n"
    "2. A brief architectural summary paragraph"
)


async def output_generator_node(state: ContextBridgeState) -> dict:
    intent_mem = _truncate(state["intent_memory"] or "No intent information captured.", MAX_OUTPUT_TOKENS)
    code_mem = _truncate(state["code_memory"] or "No code files discussed.", MAX_OUTPUT_TOKENS)
    structure_mem = _truncate(state["structure_memory"] or "No project structure defined.", MAX_OUTPUT_TOKENS)

    # Sequential calls — each needs the full TPM budget
    intent_doc = await _groq_call(
        OUTPUT_MODEL, _INTENT_FMT_SYS,
        _INTENT_FMT_USR.format(memory=intent_mem),
        max_tokens=2000,
    )
    code_doc = await _groq_call(
        OUTPUT_MODEL, _CODE_FMT_SYS,
        _CODE_FMT_USR.format(memory=code_mem),
        max_tokens=2000,
    )
    structure_doc = await _groq_call(
        OUTPUT_MODEL, _STRUCTURE_FMT_SYS,
        _STRUCTURE_FMT_USR.format(memory=structure_mem),
        max_tokens=1500,
    )

    files = {
        "intent": intent_doc,
        "code": code_doc,
        "structure": structure_doc,
    }

    # Generate the detailed code file only when mode == "detailed"
    if state.get("mode") == "detailed" and state.get("code_detail_memory", "").strip():
        code_detail_mem = _truncate(state["code_detail_memory"], MAX_OUTPUT_TOKENS)
        files["code_detail"] = await _groq_call(
            OUTPUT_MODEL,
            "Format the following as clean markdown. Each file must be in its own fenced code block. "
            "Use the correct language identifier (python, javascript, etc.). "
            "Do not add any commentary outside the headings and code blocks.",
            (
                f"{code_detail_mem}\n\n"
                "Format each file as:\n"
                "## path/to/filename.ext\n"
                "```language\n[full code]\n```\n\n"
                "If a file's code was only partially shown in the conversation, add a comment at the top: "
                "# NOTE: partial — only the shown portion is captured"
            ),
            max_tokens=3000,
        )

    return {
        "output_files": files,
        "processing_log": ["Output files generated"],
    }

# ── Graph assembly ────────────────────────────────────────────────────────────


def _build_graph():
    g = StateGraph(ContextBridgeState)

    g.add_node("chunker", chunker_node)
    g.add_node("classifier", classifier_node)
    g.add_node("memory_dispatcher", memory_dispatcher_node)
    g.add_node("loop_controller", loop_controller_node)
    g.add_node("output_generator", output_generator_node)

    g.set_entry_point("chunker")
    g.add_edge("chunker", "classifier")
    g.add_edge("classifier", "memory_dispatcher")
    g.add_edge("memory_dispatcher", "loop_controller")
    g.add_conditional_edges(
        "loop_controller",
        _route,
        {"classifier": "classifier", "output_generator": "output_generator"},
    )
    g.add_edge("output_generator", END)

    return g.compile()


_graph = _build_graph()

# ── Public entry point ────────────────────────────────────────────────────────


async def run_graph(
    messages: list[dict],
    mode: str,
    token_target: Optional[int],
) -> dict:
    initial: ContextBridgeState = {
        "raw_messages": messages,
        "mode": mode,
        "conversation_chunks": [],
        "current_index": 0,
        "current_categories": [],
        "intent_memory": "",
        "code_memory": "",
        "structure_memory": "",
        "code_detail_memory": "",
        "processing_log": [],
        "is_complete": False,
        "output_files": {},
    }

    final = await _graph.ainvoke(initial)
    files: dict = final.get("output_files", {})

    sections = [
        ("--- FILE 1: PROJECT INTENT & CONTEXT ---", files.get("intent", "")),
        ("--- FILE 2: CODE UNDERSTANDING ---", files.get("code", "")),
        ("--- FILE 3: PROJECT STRUCTURE ---", files.get("structure", "")),
        ("--- FILE 4: DETAILED CODE ---", files.get("code_detail", "")),
    ]
    combined = "\n\n".join(
        f"{header}\n\n{body}" for header, body in sections if body.strip()
    )

    if token_target and _count_tokens(combined) > token_target:
        # Simple trim: keep the first token_target tokens
        toks = _enc.encode(combined)
        combined = _enc.decode(toks[:token_target]) + "\n\n[... truncated to token target ...]"

    return {
        "output": combined,
        "output_files": files,
        "token_count": _count_tokens(combined),
        "stages_used": [CLASSIFIER_MODEL, MEMORY_MODEL, OUTPUT_MODEL],
        "chunks_processed": len(final.get("conversation_chunks", [])),
        "processing_log": final.get("processing_log", []),
    }
