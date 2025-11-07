# main.py  — homeroom-api (v1.2.4)
from __future__ import annotations

import os
import time
import traceback
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===== OpenAI (optional) =====
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI  # openai>=1.x
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="homeroom-api", version="1.2.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 本番では必要に応じて絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Domain: Teachers
# -----------------------------------------------------------------------------
class Teacher(BaseModel):
    id: str
    displayName: str
    subject: str
    color: str = "#4f46e5"
    imageName: Optional[str] = None  # クライアントのアセット名

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

class ConsultResponse(BaseModel):
    reply: str

# -----------------------------------------------------------------------------
# Health & Root
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "homeroom-api",
        "version": app.version,
        "time": int(time.time()),
    }

@app.get("/health")
@app.get("/healthz")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/diag")
def diag() -> Dict[str, Any]:
    """
    デバッグ用: OpenAI の利用可否やモデル名を確認（鍵値そのものは返さない）
    """
    key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return {
        "version": app.version,
        "openai_pkg_installed": OPENAI_AVAILABLE,
        "api_key_present": key_present,
        "model": model,
    }

# -----------------------------------------------------------------------------
# Teachers
# -----------------------------------------------------------------------------
@app.get("/teachers", response_model=List[Teacher])
def get_teachers() -> List[Teacher]:
    return TEACHERS

# -----------------------------------------------------------------------------
# LLM helpers
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

def call_openai(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    OpenAI Chat Completions (openai>=1.x)。失敗時は None。
    """
    client, model = openai_client_or_none()
    if not client:
        print("[OPENAI] client not available or API key missing")
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            print("[OPENAI] empty content from API")
        return text or None
    except Exception as e:
        print("[OPENAI_ERR]", repr(e))
        traceback.print_exc()
        return None

def teacher_fallback_reply(teacher_id: str, user_text: str) -> str:
    """
    API不調時の簡易ローカル返信。固定文すぎないよう最低限の可変要素あり。
    """
    t = TEACHER_MAP.get(teacher_id)
    name = t.displayName if t else "担任の先生"
    snippet = (user_text[:40] + "…") if len(user_text) > 40 else user_text
    if teacher_id == "hazuki":
        return f"うん、来てくれてありがと。『{snippet}』の件、まずは深呼吸。いま出来る最小の一歩を10分だけやってみよ。終わったら自分を褒める、ね。"
    if teacher_id == "toru":
        return f"{name}です。現状:『{snippet}』。仮説を1つ立て、5分の実験で検証しましょう。結果に応じて次の対策を一緒に決めます。"
    if teacher_id == "rika":
        return f"了解！\n- お悩み: {snippet}\n- 最初の一歩: “10分だけ着手”\n- コツ: 机を30秒で整える\nいけるいける！"
    if teacher_id == "rei":
        return f"話してくれてありがとう。『{snippet}』で疲れたね。3呼吸→いちばん軽い作業を5分。ゆっくり進めよう。"
    if teacher_id == "natsuki":
        return f"結論：まず動け。理由：考えすぎると重くなる。具体策：1) 5分だけやる 2) 立って伸びる 3) もう5分。『{snippet}』は小分けにすりゃ怖くない。"
    return f"{name}だよ。『{snippet}』、まずは小さく始めて成功体験を作ろう。"

# -----------------------------------------------------------------------------
# Consult
# -----------------------------------------------------------------------------
@app.post("/consult", response_model=ConsultResponse)
def consult(payload: Dict[str, Any] = Body(...)) -> ConsultResponse:
    """
    フロントから:
      { teacher: "hazuki", text: "..." , history: [{role, content}, ...] }
    互換キー (teacher_id / message) も受ける。
    """
    teacher = (payload.get("teacher") or payload.get("teacher_id") or "").strip()
    text = (payload.get("text") or payload.get("message") or "").strip()
    raw_hist = payload.get("history") or []

    if not teacher:
        raise HTTPException(status_code=400, detail="missing 'teacher' (or 'teacher_id')")
    if not text:
        raise HTTPException(status_code=400, detail="missing 'text' (or 'message')")

    # history 正規化
    history: List[HistoryMsg] = []
    if isinstance(raw_hist, list):
        for h in raw_hist:
            try:
                if isinstance(h, dict):
                    r = h.get("role") or "user"
                    c = h.get("content") or ""
                else:
                    r, c = "user", ""
                if r not in ("system", "user", "assistant"):
                    r = "user"
                if c:
                    history.append(HistoryMsg(role=r, content=c))
            except Exception:
                continue

    # メッセージ構築
    messages: List[Dict[str, str]] = [{"role": "system", "content": build_system_prompt(teacher)}]
    for h in history:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": text})

    # まずは LLM
    reply = call_openai(messages)
    if not reply:
        reply = teacher_fallback_reply(teacher, text)
    return ConsultResponse(reply=reply)
