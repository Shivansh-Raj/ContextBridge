import time
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Search: backend/.env → project-root/.env → project-root/.env.example
_here = Path(__file__).parent
for _p in [_here / ".env", _here.parent / ".env", _here.parent / ".env.example"]:
    if _p.exists():
        load_dotenv(_p)
        break

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parsers import parse_conversation
from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ContextBridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    conversation_text: str
    file_content: Optional[str] = None
    file_type: Optional[str] = "txt"
    mode: str = "detailed"
    token_target: Optional[int] = None


class ProcessResponse(BaseModel):
    output: str
    token_count: int
    mode: str
    stages_used: list[str]
    processing_time_ms: int


@app.post("/api/process", response_model=ProcessResponse)
async def process_conversation(req: ProcessRequest):
    start = time.time()

    raw_text = req.file_content if req.file_content else req.conversation_text
    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="No conversation text provided")

    messages = parse_conversation(raw_text, req.file_type or "txt")
    if not messages:
        raise HTTPException(status_code=400, detail="Could not parse any messages from input")

    try:
        result = await run_pipeline(messages, req.mode, req.token_target)
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = int((time.time() - start) * 1000)

    return ProcessResponse(
        output=result["output"],
        token_count=result["token_count"],
        mode=req.mode,
        stages_used=result["stages_used"],
        processing_time_ms=elapsed_ms,
    )


@app.get("/health")
def health():
    import os
    return {
        "status": "ok",
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY")),
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
    }
