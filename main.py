import os
import time
import traceback
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ====== 環境変数 ======
PROVIDER = os.getenv("PROVIDER", "openai")  # openai / groq / openrouter
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
USE_FAKE = os.getenv("USE_FAKE", "0")

# ====== FastAPI ======
app = FastAPI(title="AI Recover API", version="2.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ====== I/O ======
class QuestionIn(BaseModel):
    question: str

class ConsultIn(BaseModel):
    message: str
    persona: str = "gentle_brother"    # 口調キー
    user_name: Optional[str] = None    # 例: "はやと"
    teacher_name: Optional[str] = None # 例: "ナツキ"
    history: Optional[List[str]] = None  # ["ユーザー: ...", "先生: ..."]

# ====== LLM呼び出し ======
def chat_api(messages, retries: int = 1, timeout_sec: int = 30) -> str:
    if PROVIDER == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"; key = GROQ_API_KEY
    elif PROVIDER == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"; key = OPENROUTER_API_KEY
    else:
        url = "https://api.openai.com/v1/chat/completions"; key = OPENAI_API_KEY

    if not key:
        return f"Server not configured: missing API key for provider '{PROVIDER}'."

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": MODEL, "messages": messages, "max_tokens": 400, "temperature": 0.4}

    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
            if r.status_code >= 400:
                return f"{PROVIDER} error {r.status_code}: {r.text[:500]}"
            j = r.json()
            return j.get("choices", [{}])[0].get("message", {}).get("content") or "(no content)"
        except Exception as e:
            last_err = e; time.sleep(1)
    return f"Server exception while calling provider '{PROVIDER}': {last_err}"

# ====== ping ======
@app.get("/")
def root():
    return {"ok": True, "service": "airecover", "provider": PROVIDER, "model": MODEL, "version": "2.3.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ====== /question：ステップ形式 ======
@app.post("/question")
def question_api(data: QuestionIn):
    try:
        if USE_FAKE == "1":
            return (
                "1. 問題の要点を整理: 条件を確認しよう。\n"
                "2. 式や条件を立てる: 必要な式をつくる。\n"
                "3. 代入・計算: 計算して値を出す。\n"
                "4. 検算・見直し: 最後に確認。"
            )
        if PROVIDER == "openai" and not OPENAI_API_KEY:
            return "Server not configured: missing OPENAI_API_KEY."

        system = (
            "You are a helpful Japanese tutor. "
            "Return the explanation in EXACTLY four lines of this format:\n"
            "1. タイトル: 内容\n2. タイトル: 内容\n3. タイトル: 内容\n4. タイトル: 内容\n"
            "Use plain text equations like 2x+3=7 → 2x=4 → x=2. "
            "No markdown code blocks, no bullet lists, no extra lines."
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.question},
        ]
        return chat_api(messages=msgs)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== /consult：短文2行＋優しい質問。SUGGESTは返さない ======
@app.post("/consult")
def consult_api(data: ConsultIn):
    try:
        if USE_FAKE == "1":
            uname = data.user_name or ""
            tname = data.teacher_name or ""
            head = f"{uname}、話してくれてありがとう。" if uname else "話してくれてありがとう。"
            tail = "今日はどこが一番良かった？"
            return f"{head}\n{tail}"

        style_map = {
            "gentle_brother": "やわらかい口調。語尾は穏やか。絵文字は少し😊",
            "yankee": "少しフランク。優しさ最優先。絵文字少なめ",
            "energetic_male": "明るくテンポ良い。短く背中を押す。絵文字少しOK",
            "gentle_sister": "包み込むように優しい。ゆっくり。絵文字は控えめ🌙",
            "little_sister": "フレンドリーで可愛い相づち。絵文字OK",
            "cool_female": "落ち着いた丁寧語。端的でやさしい問い。絵文字なし",
        }
        tone = style_map.get(data.persona, style_map["gentle_brother"])

        # 名前（任意）を自然に差し込む
        uname = data.user_name or ""
        tname = data.teacher_name or ""
        name_prompt = ""
        if uname and tname:
            name_prompt = f"あなたは{tname}として、{uname}に話しかけます。"
        elif uname:
            name_prompt = f"あなたは担任として、{uname}に話しかけます。"
        elif tname:
            name_prompt = f"あなたは{tname}として話します。"

        history_block = ""
        if data.history:
            joined = "\n".join(data.history[-8:])
            history_block = f"\n<chat_history>\n{joined}\n</chat_history>"

        system = (
            f"You are a kind Japanese friend on LINE. Style: {tone}. "
            f"{name_prompt} "
            "Reply must feel like a short caring DM.\n"
            "HARD RULES:\n"
            "・Write at most 2 short lines (each <= 60 characters).\n"
            "・Start with empathy (うん/そっか/話してくれてありがとう など)。\n"
            "・End with exactly ONE gentle question.\n"
            "・Japanese only.\n"
            "・Do NOT wrap your reply in quotes.\n"
            "・Do NOT add any 'SUGGEST:' or metadata lines."
            f"{history_block}"
        )

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.message},
        ]
        out = chat_api(messages=msgs)

        # 念のためサニタイズ（モデルが引用符やSUGGESTを付けても消す）
        cleaned = out.strip()
        if cleaned.startswith(("\"", "“", "'")) and cleaned.endswith(("\"", "”", "'")) and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()
        # 改行エスケープ除去
        cleaned = cleaned.replace("\\n", "\n").replace("\\r\\n", "\n")
        # SUGGEST行を除去
        lines = [ln for ln in cleaned.splitlines() if not ln.strip().upper().startswith("SUGGEST:")]
        cleaned = "\n".join(lines).strip()

        # 2行を超えたら先頭2行だけ残す
        two = [ln for ln in cleaned.splitlines() if ln.strip()]
        if len(two) > 2:
            cleaned = "\n".join(two[:2])

        return cleaned or "うん、話してくれてありがとう。\n今は何が一番の関心ごと？"

    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"
