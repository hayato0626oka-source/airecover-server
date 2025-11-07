# main.py (FastAPI 1.10対応 + 同期OpenAI + alias対応) v1.2.0
from __future__ import annotations
import os, time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- OpenAI optional import ---
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(title="homeroom-api", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番は絞ってOK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Teacher data
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

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class HistoryMsg(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ConsultRequest(BaseModel):
    # iOS側が teacher_id / message を送っても受け取れるよう alias を設定
    teacher: str = Field(..., alias="teacher_id")
    text: str = Field(..., alias="message")
    history: Optional[List[HistoryMsg]] = None

    class Config:
        allow_population_by_field_name = True  # aliasでもフィールド名でも受理

class ConsultResponse(BaseModel):
    reply: str

# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "homeroom-api", "version": app.version, "time": int(time.time())}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/teachers", response_model=List[Teacher])
def get_teachers():
    return TEACHERS

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

# --- OpenAI Client (sync) ---
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

def teacher_fallback_reply(teacher_id: str, text: str) -> str:
    t = TEACHER_MAP.get(teacher_id)
    name = t.displayName if t else "担任の先生"
    if teacher_id == "hazuki":
        return f"うん、よく来たね。『{text}』の悩みも、まず向き合えてるのがえらいよ。10分だけやってみよ。"
    if teacher_id == "toru":
        return f"{name}です。『{text}』の件、1つだけ仮説を立てて行動実験してみよう。"
    if teacher_id == "rika":
        return f"なるほど！\n- お悩み: {text}\n- 今日の一歩: “10分だけやる”\nいけるいける！"
    if teacher_id == "rei":
        return f"話してくれてありがとう。『{text}』で不安なんですね。一緒に小さく始めましょう。"
    if teacher_id == "natsuki":
        return f"結論：やる。理由：考えてるより早い。具体策：5分だけ動け。『{text}』は分割すれば大丈夫。"
    return f"{name}だよ。『{text}』はまず小さく動こう。"

@app.post("/consult", response_model=ConsultResponse)
def consult(req: ConsultRequest) -> ConsultResponse:
    teacher_id = req.teacher
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    msgs = [{"role": "system", "content": build_system_prompt(teacher_id)}]
    if req.history:
        for h in req.history:
            msgs.append({"role": h.role, "content": h.content})
    msgs.append({"role": "user", "content": text})

    reply = call_openai_sync(msgs)
    if not reply:
        reply = teacher_fallback_reply(teacher_id, text)

    return ConsultResponse(reply=reply)
