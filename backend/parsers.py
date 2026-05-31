import json
import re
from typing import List, Dict


def parse_conversation(raw: str, file_type: str) -> List[Dict[str, str]]:
    file_type = file_type.lower().strip(".")
    if file_type == "json":
        return _parse_json(raw)
    elif file_type == "html":
        return _parse_html(raw)
    else:
        return _parse_text(raw)


def _parse_json(raw: str) -> List[Dict[str, str]]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "messages" in data:
            msgs = data["messages"]
        elif isinstance(data, list):
            msgs = data
        else:
            return _parse_text(raw)

        result = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if role in ("user", "assistant") and content.strip():
                result.append({"role": role, "content": content.strip()})
        return result
    except (json.JSONDecodeError, KeyError):
        return _parse_text(raw)


def _parse_html(raw: str) -> List[Dict[str, str]]:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        messages = []
        for div in soup.find_all("div", class_=re.compile(r"human|user|assistant|bot", re.I)):
            classes = " ".join(div.get("class", []))
            role = "user" if re.search(r"human|user", classes, re.I) else "assistant"
            text = div.get_text(separator=" ", strip=True)
            if text:
                messages.append({"role": role, "content": text})
        if messages:
            return messages
    except ImportError:
        pass

    clean = re.sub(r"<[^>]+>", "\n", raw)
    return _parse_text(clean)


def _parse_text(raw: str) -> List[Dict[str, str]]:
    lines = raw.split("\n")
    messages = []
    current_role = None
    current_lines: List[str] = []

    user_pattern = re.compile(r"^(you|user|human)\s*[:>]", re.I)
    assistant_pattern = re.compile(r"^(assistant|ai|claude|gpt|bot|chatgpt|gemini)\s*[:>]", re.I)

    for line in lines:
        stripped = line.strip()
        if user_pattern.match(stripped):
            if current_role and current_lines:
                messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})
            current_role = "user"
            content = user_pattern.sub("", stripped).strip()
            current_lines = [content] if content else []
        elif assistant_pattern.match(stripped):
            if current_role and current_lines:
                messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})
            current_role = "assistant"
            content = assistant_pattern.sub("", stripped).strip()
            current_lines = [content] if content else []
        else:
            if current_role is not None:
                current_lines.append(line)

    if current_role and current_lines:
        messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})

    # Fallback: split on double newlines and alternate roles
    if not messages:
        chunks = [c.strip() for c in re.split(r"\n{2,}", raw) if c.strip()]
        for i, chunk in enumerate(chunks):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": chunk})

    return [m for m in messages if m["content"]]
