import os
import time
import traceback
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ====== 環境変数 ======
PROVIDER = os.getenv("PROVIDER", "openai")  # "openai" / "groq" / "openrouter"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4o-mini")   # 省コスト既定
USE_FAKE = os.getenv("USE_FAKE", "0")       # "1" ならダミー即レス

# ====== FastAPI ======
app = FastAPI(title="AI Recover API", version="2.2.0")
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
    persona: str = "gentle_brother"
    history: Optional[List[str]] = None  # 直近ログ（任意）

# ====== 共通 LLM 呼び出し ======
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
    body = {"model": MODEL, "messages": messages, "max_tokens": 600, "temperature": 0.4}
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
    return {"ok": True, "service": "airecover", "provider": PROVIDER, "model": MODEL, "version": "2.2.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ====== /question：ステップ形式（アプリのパーサと厳密一致） ======
@app.post("/question")
def question_api(data: QuestionIn):
    try:
        if USE_FAKE == "1":
            return (
                "1. 問題の要点を整理: 条件を確認しよう。\n"
                "2. 式や条件を立てる: 必要な式を作る。\n"
                "3. 代入・計算: 計算して値を出す。\n"
                "4. 検算・見直し: 最後に確認。"
            )
        if PROVIDER == "openai" and not OPENAI_API_KEY:
            return "Server not configured: missing OPENAI_API_KEY."

        system_prompt = (
            "You are a helpful Japanese tutor for students. "
            "Answer in clear **Japanese** with no LaTeX and no code blocks. "
            "Return your explanation in EXACTLY this line-by-line step format:\n"
            "1. タイトル: 内容\n"
            "2. タイトル: 内容\n"
            "3. タイトル: 内容\n"
            "4. タイトル: 内容\n"
            "Rules: Start each line with a number and a dot (1., 2., ...). "
            "Use a colon '：' or ':' to separate a short title and a concise explanation. "
            "Equations must be plain text like 2x+3=7 → 2x=4 → x=2. "
            "Keep it compact and scannable."
        )
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data.question},
        ]
        return chat_api(messages=msgs)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc()); return f"Unhandled server exception: {e}"

# ====== /consult：短文×往復・候補付き（LINE調） ======
@app.post("/consult")
def consult_api(data: ConsultIn):
    try:
        if USE_FAKE == "1":
            return "うん、話してくれてありがとう。\n今日はどこが一番しんどい？\nSUGGEST: [眠れない,学校が重い,人間関係]"

        style_map = {
            "gentle_brother": "語尾はやわらかく、砕けすぎない口調。絵文字はごく少し😊",
            "yankee": "ちょいフランク。優しさ最優先で荒くしすぎない。絵文字少なめ",
            "energetic_male": "明るくテンポよく。短く背中を押す。絵文字少しOK",
            "gentle_sister": "包み込む口調。ゆっくり安心感。絵文字は控えめで🌙など",
            "little_sister": "フレンドリーで可愛い相づち。絵文字OKだけど過剰にしない",
            "cool_female": "落ち着いた丁寧語。短く要点＋優しい問いかけ。絵文字ほぼ無し",
        }
        tone = style_map.get(data.persona, style_map["gentle_brother"])

        history_block = ""
        if data.history:
            joined = "\n".join(data.history[-8:])
            history_block = f"\n<chat_history>\n{joined}\n</chat_history>\n"

        system = (
            f"You are a kind Japanese friend on LINE. Style: {tone}. "
            "Your reply must feel like a short, caring DM.\n"
            "HARD RULES:\n"
            "・Use at most 2 short lines (each <= 60 characters).\n"
            "・Start with empathy (うん/そっか/話してくれてありがとう など)。\n"
            "・End with exactly ONE gentle question to keep conversation going.\n"
            "・No long advice, no lists, no markdown headings.\n"
            "・After the reply, output one line starting with 'SUGGEST: [a,b,c]' for 3 quick-reply candidates.\n"
            "・Japanese only."
            f"{history_block}"
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.message}
        ]
        return chat_api(messages=msgs)

    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc()); return f"Unhandled server exception: {e}"
