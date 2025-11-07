# main.py — homeroom-api v1.2.2
from __future__ import annotations
import os, time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===== Optional OpenAI import =====
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="homeroom-api", version="1.2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 本番は必要に応じて絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
class Teacher(BaseModel):
    id: str
    displayName: str
    subject: str
    color: str = "#4f46e5"
    imageName: Optional[str] = None

TEACHERS: List[Teacher] = [
    Teacher(id="hazuki",  displayName="水瀬 葉月",   subject="国語", color="#ef4444", imageName="teacher_hazuki"),
    Teacher(id="toru",    displayName="五十嵐 トオル", subject="理科", color="#22c55e", imageName="teacher_toru"),
    Teacher(id="rika",    displayName="小町 リカ",   subject="数学", color="#06b6d4", imageName="teacher_rika"),
    Teacher(id="rei",     displayName="進藤 怜",     subject="英語", color="#8b5cf6", imageName="teacher_rei"),
    Teacher(id="natsuki", displayName="小林 夏樹",   subject="社会", color="#f59e0b", imageName="teacher_natsuki"),
]
TEACHER_MAP: Dict[str, Teacher] = {t.id: t for t in TEACHERS}

class HistoryMsg(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ConsultResponse(BaseModel):
    reply: str

# -----------------------------------------------------------------------------
# Health / Teachers
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "service": "homeroom-api", "version": app.version, "time": int(time.time())}

@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/teachers", response_model=List[Teacher])
def get_teachers() -> List[Teacher]:
    return TEACHERS

# -----------------------------------------------------------------------------
# LLM settings
# -----------------------------------------------------------------------------
SYSTEM_BASE = (
    "You are a kind, concise study and life advisor. "
    "Answer in Japanese. Keep answers supportive and clear."
)
TEACHER_STYLES = {
    "hazuki": "語尾はやわらかく。時々軽いジョークを挟む。優先: 励まし→要点→次の一歩。",
    "toru":   "落ち着いた敬語。観察→仮説→提案の順で論理的に。",
    "rika":   "明るく素直に要点を箇条書き。テンポよく短文で。",
    "rei":    "穏やかな口調。共感→整理→小さな達成の提案。",
    "natsuki":"ぶっきらぼうだが優しい。結論先出し→理由→具体策の三段構成。",
}

def build_system_prompt(teacher_id: str) -> str:
    return f"{SYSTEM_BASE}\n教師スタイル: {TEACHER_STYLES.get(teacher_id, '')}"

def openai_client_or_none():
    if not OPENAI_AVAILABLE:
        return None, None
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None, None
    try:
        return OpenAI(api_key=key), os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    except Exception:
        return None, None

def call_openai_sync(messages: List[Dict[str, str]]) -> Optional[str]:
    client, model = openai_client_or_none()
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

def teacher_fallback_reply(teacher_id: str, user_text: str) -> str:
    t = TEACHER_MAP.get(teacher_id)
    name = t.displayName if t else "担任の先生"
    if teacher_id == "hazuki":
        return f"うん、よく来たね。まずは深呼吸しよ。『{user_text}』って悩み、ちゃんと向き合えてる時点でえらい。10分だけ手をつけてみよう。"
    if teacher_id == "toru":
        return f"{name}です。現状:『{user_text}』。仮説を1つだけ立てて、小さな実験（5分着手）をやってみましょう。"
    if teacher_id == "rika":
        return f"了解！\n- お悩み: {user_text}\n- いまやること: “10分だけスタート”\n- コツ: 机を30秒で整える\nまずは小さく！"
    if teacher_id == "rei":
        return f"話してくれてありがとう。『{user_text}』で落ち着かないね。3呼吸→“いちばん軽い作業”を5分、一緒にやろう。"
    if teacher_id == "natsuki":
        return f"結論：まず動け。理由：考えすぎるほど重くなる。具体策：1) 5分やる 2) 立って伸びる 3) もう5分。『{user_text}』は分割すれば大丈夫。"
    return f"{name}だよ。『{user_text}』は小さく始めて成功体験を作ろう。"

# -----------------------------------------------------------------------------
# Consult endpoint — accepts both key styles
# -----------------------------------------------------------------------------
@app.post("/consult", response_model=ConsultResponse)
def consult(payload: Dict[str, Any] = Body(...)) -> ConsultResponse:
    """
    Accepts either:
      {\"teacher\":\"hazuki\", \"text\":\"...\", \"history\":[...]}
    or
      {\"teacher_id\":\"hazuki\", \"message\":\"...\", \"history\":[...]}
    """
    teacher = (payload.get("teacher") or payload.get("teacher_id") or "").strip()
    text    = (payload.get("text")    or payload.get("message")     or "").strip()
    raw_hist = payload.get("history") or []

    if not teacher:
        raise HTTPException(status_code=400, detail="missing 'teacher' (or 'teacher_id')")
    if not text:
        raise HTTPException(status_code=400, detail="missing 'text' (or 'message')")

    # Coerce history safely
    history: List[HistoryMsg] = []
    if isinstance(raw_hist, list):
        for h in raw_hist:
            try:
                role = (h.get("role") if isinstance(h, dict) else None) or "user"
                content = (h.get("content") if isinstance(h, dict) else None) or ""
                if role not in ("system", "user", "assistant"):
                    role = "user"
                if content:
                    history.append(HistoryMsg(role=role, content=content))
            except Exception:
                continue

    # Compose messages
    messages: List[Dict[str, str]] = [{"role": "system", "content": build_system_prompt(teacher)}]
    for h in history:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": text})

    # LLM → fallback
    reply = call_openai_sync(messages) or teacher_fallback_reply(teacher, text)
    return ConsultResponse(reply=reply)
