import os
import time
import traceback
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ====== 環境変数 ======
PROVIDER = os.getenv("PROVIDER", "openai")  # "openai" / "groq" / "openrouter"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4o-mini")   # 省コスト既定
USE_FAKE = os.getenv("USE_FAKE", "0")       # "1" ならダミー即レス

# ====== FastAPI 基本設定 ======
app = FastAPI(title="AI Recover API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じてドメインを制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== I/O モデル ======
class QuestionIn(BaseModel):
    question: str

class ConsultIn(BaseModel):
    message: str
    persona: str = "gentle_brother"

# ====== ルート/ヘルス ======
@app.get("/")
def root():
    return {"ok": True, "service": "airecover", "provider": PROVIDER, "model": MODEL}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ====== LLM 呼び出し（プロバイダ切替対応・落ちない設計） ======
def chat_api(messages, retries: int = 1, timeout_sec: int = 30) -> str:
    """
    各社の OpenAI 互換APIに POST。
    4xx/5xx は本文ごと返し、例外は握りつぶして文字列化。500を出さない。
    """
    if PROVIDER == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        key = GROQ_API_KEY
    elif PROVIDER == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        key = OPENROUTER_API_KEY
    else:
        url = "https://api.openai.com/v1/chat/completions"
        key = OPENAI_API_KEY

    if not key:
        return f"Server not configured: missing API key for provider '{PROVIDER}'."

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 400,     # コスト抑制
        "temperature": 0.3,    # ぶれを抑えて安定した解答
    }

    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
            if r.status_code >= 400:
                # 401/429などの本文をそのまま返して原因見える化
                return f"{PROVIDER} error {r.status_code}: {r.text[:500]}"
            j = r.json()
            # OpenAI互換のレスポンス取り出し
            content = (
                j.get("choices", [{}])[0]
                 .get("message", {})
                 .get("content")
            )
            return content or "(no content)"
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    return f"Server exception while calling provider '{PROVIDER}': {last_err}"

# ====== /question ======
@app.post("/question")
def question_api(data: QuestionIn):
    try:
        # ダミーモード（切り分け・デモ用）
        if USE_FAKE == "1":
            return (
                "### 解説（ダミー）\n"
                f"**質問:** {data.question}\n\n"
                "1. 問題文の条件を整理する\n"
                "2. 必要な式を立てる\n"
                "3. 同じ操作を両辺に行い解を求める\n"
                "4. 代入して検算する\n"
            )

        if PROVIDER == "openai" and not OPENAI_API_KEY:
            return "Server not configured: missing OPENAI_API_KEY."

        # “読みやすい手順カード” を返しやすいプロンプト
        system = (
            "You are a supportive Japanese tutor. "
            "Answer in **clean Japanese Markdown** for iOS display. "
            "Start with a short one-line summary. Then provide a **numbered procedure** using lines like `1.`, `2.`, `3.`. "
            "Each step should have a short title and one or two concise sentences. "
            "Avoid LaTeX or code fences. Write equations plainly, e.g., (2x+3=7 → 2x=4 → x=2). "
            "Keep it compact and scannable."
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.question},
        ]
        return chat_api(messages=msgs)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== /consult ======
@app.post("/consult")
def consult_api(data: ConsultIn):
    try:
        if USE_FAKE == "1":
            return f"（ダミー）{persona_label(data.persona)}として：まずは一息。次の一歩は『5分だけ着手』だよ。"

        styles = {
            "gentle_brother": "優しいお兄さん。親身で柔らかく、具体的な一歩を示す。",
            "yankee": "ヤンキー風。少し荒いが面倒見がいい。乱暴すぎず励ます。",
            "energetic_male": "元気で明るい。短文でテンポよく背中を押す。",
            "gentle_sister": "優しいお姉さん。包み込む安心感と丁寧な語り。",
            "little_sister": "妹系。フレンドリーで可愛い相づち。",
            "cool_female": "クール系。落ち着きと論理、要点→次の一歩で端的に。",
        }
        tone = styles.get(data.persona, styles["gentle_brother"])
        system = (
            "You are a Japanese school counselor. "
            f"Speak in this tone: {tone} "
            "Always include **one actionable next step** at the end."
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.message},
        ]
        return chat_api(messages=msgs)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

def persona_label(key: str) -> str:
    labels = {
        "gentle_brother": "優しいお兄さん",
        "yankee": "ヤンキー",
        "energetic_male": "元気",
        "gentle_sister": "優しいお姉さん",
        "little_sister": "妹",
        "cool_female": "クール",
    }
    return labels.get(key, "優しいお兄さん")
