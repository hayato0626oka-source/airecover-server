import os, requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==== 基本設定 ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"

# ==== アプリ作成 ====
app = FastAPI(title="AI Recover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # iOSアプリからの通信を許可
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==== ChatGPTに問い合わせる関数 ====
def openai_chat(messages):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages, "max_tokens": 1000},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ==== 型定義 ====
class QuestionIn(BaseModel):
    question: str

# ==== ルート（動作確認用） ====
@app.get("/")
def root():
    return {"ok": True, "service": "airecover"}

# ==== 質問API ====
@app.post("/question")
def question_api(data: QuestionIn):
    msgs = [
        {"role": "system", "content": "You are a supportive Japanese tutor."},
        {"role": "user", "content": data.question},
    ]
    return openai_chat(msgs)
