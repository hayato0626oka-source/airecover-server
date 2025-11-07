from __future__ import annotations
import os, time, traceback
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===== Optional OpenAI import（無くても起動はする） =====
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

app = FastAPI(title="homeroom-api", version="1.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Domain --------------------
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

# -------------------- Health --------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "service": "homeroom-api", "version": app.version, "time": int(time.time())}

@app.get("/health"), app.get("/healthz")
def health() -> Dict[str, str]:  # type: ignore[no-redef]
    return {"status": "ok"}

@app.get("/teachers", response_model=List[Teacher])
def get_teachers() -> List[Teacher]:
    return TEACHERS

# -------------------- LLM helpers --------------------
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
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        client = OpenAI(api_key=key)
        return client, model
    except Exception:
        return None, None

def call_openai_sync(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    失敗時は None を返す。ログに 'OPENAI_ERR:' として理由を出す。
    """
    client, model = openai_client_or_none()
    if not client:
        print("[OPENAI_ERR] client missing or api key absent")
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
            print("[OPENAI_ERR] empty content")
        return text or None
    except Exception as e:
        print("[OPENAI_ERR]", repr(e))
        traceback.print_exc()
        return None

def teacher_fallback_reply(teacher_id: str, user_text: str) -> str:
    t = TEACHER_MAP.get(teacher_id)
    name = t.displayName if t else "担任の先生"
    if teacher_id == "hazuki":
        return f"うん、よく来たね。まずは深呼吸しよ。『{user_text}』に向けて、10分だけ手をつけよう。やれば進む、進めば楽になる。"
    if teacher_id == "toru":
        return f"{name}です。現状:『{user_text}』。仮説を1つ立て、5分の実験で検証しましょう。"
    if teacher_id == "rika":
        return f"了解！\n- お悩み: {user_text}\n- いまやること: “10分だけスタート”\n- コツ: 机を30秒で整える"
    if teacher_id == "rei":
        return f"話してくれてありがとう。『{user_text}』で落ち着かないね。3呼吸→軽い作業を5分。ゆっくりでOK。"
    if teacher_id == "natsuki":
        return f"結論：まず動け。理由：考えすぎるほど重くなる。具体策：1) 5分やる 2) 立って伸びる 3) もう5分。"
    return f"{name}だよ。小さく始めて成功体験を作ろう。"

# -------------------- Consult --------------------
@app.post("/consult", response_model=ConsultResponse)
def consult(payload: Dict[str, Any] = Body(...)) -> ConsultResponse:
    # 両方のキーに対応
    teacher = (payload.get("teacher") or payload.get("teacher_id") or "").strip()
    text    = (payload.get("text")    or payload.get("message")     or "").strip()
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
                r = (h.get("role") if isinstance(h, dict) else None) or "user"
                c = (h.get("content") if isinstance(h, dict) else None) or ""
                if r not in ("system", "user", "assistant"):
                    r = "user"
                if c:
                    history.append(HistoryMsg(role=r, content=c))
            except Exception:
                continue

    messages: List[Dict[str, str]] = [{"role": "system", "content": build_system_prompt(teacher)}]
    for h in history:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": text})

    reply = call_openai_sync(messages)
    if not reply:
        reply = teacher_fallback_reply(teacher, text)
    return ConsultResponse(reply=reply)

# -------------------- Diagnostics --------------------
@app.get("/diag")
def diag() -> Dict[str, Any]:
    """
    デバッグ用: OpenAI が使えるかのブール、モデル名、鍵の有無のみを返す。
    セキュアのため鍵の値は返さない。
    """
    key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return {
        "version": app.version,
        "openai_pkg_installed": OPENAI_AVAILABLE,
        "api_key_present": key_present,
        "model": model,
    }
