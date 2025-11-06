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

# ====== FastAPI ======
app = FastAPI(title="AI Recover API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== I/O ======
class QuestionIn(BaseModel):
    question: str

class ConsultIn(BaseModel):
    message: str
    teacher_key: str | None = None  # "saki" / "natsuki" / "shiori" / "megane" / "genki" など
    student_nick: str | None = None

# ====== ルート ======
@app.get("/")
def root():
    return {"ok": True, "service": "airecover", "provider": PROVIDER, "model": MODEL}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ====== LLMコール ======
def chat_api(messages, retries: int = 1, timeout_sec: int = 30) -> str:
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
        "max_tokens": 300,
        "temperature": 0.5,
    }

    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
            if r.status_code >= 400:
                return f"{PROVIDER} error {r.status_code}: {r.text[:500]}"
            j = r.json()
            return (
                j.get("choices", [{}])[0]
                 .get("message", {})
                 .get("content")
            ) or "(no content)"
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    return f"Server exception while calling provider '{PROVIDER}': {last_err}"

# ====== /question ======
@app.post("/question")
def question_api(data: QuestionIn):
    try:
        if USE_FAKE == "1":
            return (
                "【要点】→ 手順で解こう\n"
                "1. 条件整理\n2. 式を立てる\n3. 計算\n4. 検算\n"
            )
        system = (
            "You are a supportive Japanese tutor. "
            "Answer in **clean Japanese Markdown** for iOS. "
            "Start with a one-line summary, then show a **numbered list** of 3–6 short steps. "
            "No LaTeX/code fences. Equations plain like 2x+3=7 → 2x=4 → x=2."
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.question},
        ]
        return chat_api(msgs)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== 担任プロファイル ======
PERSONA = {
    # 画像：咲（クール系お姉さん）
    "saki": {
        "name": "咲",
        "style": "クールでロジカル。余計な絵文字なし、語尾は端的。相手を見下さないがキレ味あり。",
        "greeting": "要点から行くね。"
    },
    # 画像：ナツキ（チャラいお兄さん）
    "natsuki": {
        "name": "ナツキ",
        "style": "軽快でフレンドリー。タメ口9割、ほどよくノリ良い相づち。短文多め。",
        "greeting": "よっ、任せろ。"
    },
    # 画像：詩織（ふわふわ系お姉さん）
    "shiori": {
        "name": "詩織",
        "style": "やさしく包む。ゆるめの敬体。絵文字は控えめに1つまで。",
        "greeting": "うん、まずは落ち着こ。"
    },
    # 追加キャラ：メガネの優男
    "megane": {
        "name": "湊",
        "style": "丁寧で静か、観察的。相手の言葉を短く反射して受け止める。穏やかな助言。",
        "greeting": "話してくれてありがとう。"
    },
    # 追加キャラ：元気系超イケメン
    "genki": {
        "name": "蓮",
        "style": "明るく前向き。テンポ速め。短い応援＋次の一手を必ず提示。",
        "greeting": "いこいこ！"
    },
}

def persona_prompt(key: str | None, student_nick: str | None) -> str:
    # 不明なら詩織にしない（←ここを修正）→ デフォは **saki** に統一
    key = (key or "").lower()
    p = PERSONA.get(key) or PERSONA["saki"]
    nick = student_nick or "あなた"
    return (
        f"あなたは日本語で会話するカウンセラー。キャラ設定：{p['name']}。"
        f"話し方：{p['style']} "
        f"出力は短めの自然な会話文を2〜3文。過度な長文禁止。"
        f"必ず最後に1つだけ優しい質問で返す。"
        f"禁止：『SUGGEST:〜』等のタグ、テンプレ励ましの連発、箇条書き。"
        f"呼びかけは「{nick}」で。最初の一言は「{p['greeting']}」から始める。"
    )

# ====== /consult ======
@app.post("/consult")
def consult_api(data: ConsultIn):
    try:
        if USE_FAKE == "1":
            who = PERSONA.get((data.teacher_key or "saki").lower(), PERSONA["saki"])["name"]
            return f"{who}：{data.message[:12]}…について、まずは一緒に整えよ。次にどうしたい？"

        system = persona_prompt(data.teacher_key, data.student_nick)
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.message},
        ]
        out = chat_api(msgs)
        # iOSにそのまま貼る前提：引用符やSUGGEST残骸を掃除
        if isinstance(out, str):
            cleaned = out.strip().strip('"').replace("SUGGEST:", "").replace("Suggest:", "")
            return cleaned
        return out
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"
