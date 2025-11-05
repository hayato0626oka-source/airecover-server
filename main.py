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
MODEL = os.getenv("MODEL", "gpt-4o-mini")
USE_FAKE = os.getenv("USE_FAKE", "0")

# ====== FastAPI 基本設定 ======
app = FastAPI(title="AI Recover API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# ====== ルート / ヘルス ======
@app.get("/")
def root():
    return {"ok": True, "service": "airecover", "provider": PROVIDER, "model": MODEL, "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ====== 共通 LLM 呼び出し ======
def chat_api(messages, retries: int = 1, timeout_sec: int = 30) -> str:
    """
    各社の OpenAI 互換APIに POST。
    失敗しても 500 を返さず、常に文字列で返す。
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

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 600,
        "temperature": 0.4,
    }

    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
            if r.status_code >= 400:
                return f"{PROVIDER} error {r.status_code}: {r.text[:500]}"
            j = r.json()
            content = j.get("choices", [{}])[0].get("message", {}).get("content")
            return content or "(no content)"
        except Exception as e:
            last_err = e
            time.sleep(1)
    return f"Server exception while calling provider '{PROVIDER}': {last_err}"

# ====== /question ======
@app.post("/question")
def question_api(data: QuestionIn):
    try:
        if USE_FAKE == "1":
            return (
                "1. 問題の要点を整理: 問題文の条件を確認しよう。\n"
                "2. 式や条件を立てる: 与えられた情報を整理して式を作る。\n"
                "3. 代入・計算: 式を解いて答えを出す。\n"
                "4. 検算・見直し: 最後に答えを確認しよう。"
            )

        if PROVIDER == "openai" and not OPENAI_API_KEY:
            return "Server not configured: missing OPENAI_API_KEY."

        # 🧠 改良済みプロンプト（アプリ側のパーサに完全対応）
        system_prompt = (
            "You are a helpful Japanese tutor for middle and high school students. "
            "Answer in clear **Japanese Markdown** suitable for mobile display. "
            "Provide your explanation in **step-by-step format**, using exactly this structure:\n"
            "1. タイトル: 内容\n"
            "2. タイトル: 内容\n"
            "3. タイトル: 内容\n"
            "4. タイトル: 内容\n"
            "Each line must begin with a number (1., 2., etc.) and include a colon '：' between the title and its explanation. "
            "Avoid LaTeX or code blocks. Write equations plainly (e.g., 2x+3=7 → 2x=4 → x=2). "
            "Keep explanations short, simple, and scannable for students."
        )

        msgs = [
            {"role": "system", "content": system_prompt},
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
            "You are a Japanese counselor. "
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
