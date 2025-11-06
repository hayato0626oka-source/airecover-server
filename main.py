# main.py
from __future__ import annotations

import os
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===== Optional OpenAI import (not required in requirements.txt) =====
OPENAI_AVAILABLE = True
try:
    # Works with openai>=1.0.0 style client; if the package is missing, we fallback safely.
    from openai import OpenAI  # type: ignore
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore


# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="homeroom-api", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Domain: Teachers (IDs must match client-side expectations)
# -----------------------------------------------------------------------------
class Teacher(BaseModel):
    id: str
    displayName: str
    subject: str
    color: str = "#4f46e5"  # Tailwind indigo-600
    imageName: Optional[str] = None  # for clients that use asset names


TEACHERS: List[Teacher] = [
    Teacher(id="hazuki",  displayName="水瀬 葉月", subject="国語", color="#ef4444", imageName="teacher_hazuki"),
    Teacher(id="toru",    displayName="五十嵐 トオル", subject="理科", color="#22c55e", imageName="teacher_toru"),
    Teacher(id="rika",    displayName="小町 リカ", subject="数学", color="#06b6d4", imageName="teacher_rika"),
    Teacher(id="rei",     displayName="進藤 怜", subject="英語", color="#8b5cf6", imageName="teacher_rei"),
    Teacher(id="natsuki", displayName="小林 夏樹", subject="社会", color="#f59e0b", imageName="teacher_natsuki"),
]

TEACHER_MAP: Dict[str, Teacher] = {t.id: t for t in TEACHERS}

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class HistoryMsg(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ConsultRequest(BaseModel):
    teacher: str
    text: str
    history: Optional[List[HistoryMsg]] = None

class ConsultResponse(BaseModel):
    reply: str

# -----------------------------------------------------------------------------
# Health & Root
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    """
    Renderのヘルスチェック対策。200でサービス稼働を返す。
    """
    return {
        "status": "ok",
        "service": "homeroom-api",
        "version": app.version,
        "time": int(time.time()),
    }

@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# Teachers
# -----------------------------------------------------------------------------
@app.get("/teachers", response_model=List[Teacher])
def get_teachers() -> List[Teacher]:
    return TEACHERS

# -----------------------------------------------------------------------------
# Consult (LLM or Fallback)
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
    style = TEACHER_STYLES.get(teacher_id, "")
    return f"{SYSTEM_BASE}\n教師スタイル: {style}"

def openai_client_or_none():
    """
    Returns (client, model_name) if usable; otherwise (None, None).
    """
    if not OPENAI_AVAILABLE:
        return None, None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        client = OpenAI(api_key=api_key)
        return client, model
    except Exception:
        return None, None

async def call_openai(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Try OpenAI responses API (>=1.0 client). Return text or None on failure.
    """
    client, model = openai_client_or_none()
    if client is None:
        return None
    try:
        # Prefer Responses API for 1.x client
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None

def teacher_fallback_reply(teacher_id: str, user_text: str) -> str:
    """
    Local safe fallback when OpenAI is unavailable.
    Keep it consistent but simple; no external deps.
    """
    t = TEACHER_MAP.get(teacher_id)
    name = t.displayName if t else "担任の先生"
    if teacher_id == "hazuki":
        return f"うん、よく来たね。まずは深呼吸しよ。『{user_text}』って悩み、ちゃんと向き合えてる時点でえらい。1) いま出来る最小の一歩を決める 2) 10分だけ着手 3) 終わったら自分を褒める。ね、肩の力抜いていこう。"
    if teacher_id == "toru":
        return f"{name}です。現状:『{user_text}』。要因を仮説化→優先度順に1つだけ実験しましょう。今日の実験案: “5分タイマーで着手”。データは行動から。結果が出たら一緒に次を調整します。"
    if teacher_id == "rika":
        return f"了解！\n- お悩み: {user_text}\n- いまやること: “10分だけ”スタート\n- コツ: 机の上を30秒で整える\n- 終わったら水をひと口\nいけるいける、まずは小さく！"
    if teacher_id == "rei":
        return f"話してくれてありがとう。『{user_text}』で心が落ち着かないね。まずは3呼吸して、今の気持ちを一語でメモ。次に“いちばん軽い作業”を5分だけ一緒にやろう。大丈夫、ゆっくり進めよう。"
    if teacher_id == "natsuki":
        return f"結論：まず動け。理由：考えすぎるほど重くなる。具体策：1) たった5分だけやる 2) 終わったら立って伸びる 3) 次の5分を足す。『{user_text}』は小分けにすりゃ怖くない。"
    # default
    return f"{name}だよ。『{user_text}』、まずは小さく始めて成功体験を積もう。5〜10分だけ着手→終わったら水を飲む→次の一歩を1行でメモ、でOK。"

@app.post("/consult", response_model=ConsultResponse)
async def consult(req: ConsultRequest) -> ConsultResponse:
    teacher_id = req.teacher
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # Prepare messages for LLM
    system = build_system_prompt(teacher_id)
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
    if req.history:
        for h in req.history:
            # guard roles
            role = h.role if h.role in ("system", "user", "assistant") else "user"
            msgs.append({"role": role, "content": h.content})
    msgs.append({"role": "user", "content": text})

    # Try LLM first
    reply = await call_openai(msgs)
    if not reply:
        # Fallback local reply
        reply = teacher_fallback_reply(teacher_id, text)

    return ConsultResponse(reply=reply)
