# main.py - AI Recover API
import os, requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"

app = FastAPI(title="AI Recover API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def openai_chat(messages):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages, "max_tokens": 1000},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ----------- 質問API -----------
class QuestionIn(BaseModel):
    question: str

@app.post("/question")
def question_api(data: QuestionIn):
    messages = [
        {"role": "system", "content": "You are a supportive Japanese tutor."},
        {"role": "user", "content": data.question},
    ]
    return openai_chat(messages)

# ----------- 相談API（口調タイプ） -----------
class ConsultIn(BaseModel):
    message: str
    persona: str

@app.post("/consult")
def consult_api(data: ConsultIn):
    persona_styles = {
        "gentle_brother": "優しいお兄さん風。親身で柔らかく励ます。",
        "yankee": "ヤンキー風。口は荒いが根は面倒見が良い。",
        "energetic_male": "元気で明るいテンション。",
        "gentle_sister": "優しいお姉さん風。包み込むような安心感。",
        "little_sister": "妹系。甘えた感じでフレンドリー。",
        "cool_female": "クール系。落ち着いて理知的。",
    }
    style = persona_styles.get(data.persona, "gentle_brother")
    system = f"You are a Japanese AI teacher. Tone: {style}"
    user = data.message
    return openai_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
