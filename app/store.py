
from __future__ import annotations
import json, os, time
from typing import List, Dict, Any

STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
os.makedirs(STORE_DIR, exist_ok=True)

def _chat_path(user_id: str) -> str:
    return os.path.join(STORE_DIR, f"chats_{user_id}.json")

def load_chats(user_id: str) -> List[Dict[str, Any]]:
    p = _chat_path(user_id)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return []

def save_chat(user_id: str, role: str, text: str, meta: Dict[str, Any] | None = None):
    chats = load_chats(user_id)
    chats.append({"ts": time.time(), "role": role, "text": text, "meta": meta or {}})
    with open(_chat_path(user_id), "w") as f:
        json.dump(chats, f, indent=2)
