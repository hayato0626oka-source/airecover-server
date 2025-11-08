# main.py
import os
from typing import List
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Homeroom API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"

# ---------- Models ----------
class ConsultRequest(BaseModel):
    teacher: str
    text: str

class ConsultResponse(BaseModel):
    reply: str

class LearnRequest(BaseModel):
    teacher: str
    question: str

class LearnStep(BaseModel):
    index: int
    title: str
    body: str

class LearnResponse(BaseModel):
    steps: List[LearnStep]

class PhraseResponse(BaseModel):
    text: str

# ---------- Prompts ----------
TEACHER_PERSONA = {
    "japanese": "あなたは国語の先生・水瀬葉月。優しく、たまにユーモア。",
    "science":  "あなたは理科の先生・五十嵐トオル。落ち着いた大学教授。",
    "math":     "あなたは数学の先生・小町リカ。天才肌で明快なロジック。",
    "english":  "あなたは英語の先生・進藤怜。穏やかで丁寧に説明する。",
    "social":   "あなたは社会の先生・小林夏樹。テンポよく具体例重視。",
}

def persona(teacher: str) -> str:
    return TEACHER_PERSONA.get(teacher, "あなたは親切な家庭教師。")

# ---------- Routes ----------
@app.post("/consult", response_model=ConsultResponse)
def consult(req: ConsultRequest):
    """
    先生キャラで会話。短く要点を返す。
    """
    sys = f"{persona(req.teacher)} 学習者の相談に、優しく簡潔に日本語で答えて。改行は2〜4文に抑える。"
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": req.text}
    ]
    res = client.chat.completions.create(model=MODEL, messages=msg, temperature=0.8, max_tokens=300)
    reply = res.choices[0].message.content.strip()
    return ConsultResponse(reply=reply)

@app.post("/learn", response_model=LearnResponse)
def learn(req: LearnRequest):
    """
    5ステップの解説をJSONで返す。
    """
    sys = f"""{persona(req.teacher)}
あなたは生徒の質問に対し、理解の階段を「5つの小さなステップ」で作る先生です。
各ステップは 'index'(1〜5), 'title'(短い見出し), 'body'(わかりやすい説明) のJSON配列で返して。日本語。"""
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"この質問を5ステップで教えて: {req.question}"}
    ]
    res = client.chat.completions.create(
        model=MODEL,
        messages=msg,
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=800
    )
    content = res.choices[0].message.content
    # 期待するキーで安全に取り出し
    import json
    try:
        data = json.loads(content)
        steps_raw = data.get("steps") or data.get("Steps") or []
        steps = []
        for i, s in enumerate(steps_raw, start=1):
            steps.append(LearnStep(
                index=int(s.get("index", i)),
                title=str(s.get("title", f"Step {i}")),
                body=str(s.get("body", ""))
            ))
    except Exception:
        # フォールバック（LLMがJSON以外を返した場合）
        txt = content.replace("\n", " ")
        steps = [LearnStep(index=1, title="要約", body=txt[:180]),
                 LearnStep(index=2, title="次の一歩", body="キーワードを確認しよう。"),
                 LearnStep(index=3, title="例題", body="簡単な例で仕組みを見よう。"),
                 LearnStep(index=4, title="応用", body="条件を変えても成り立つか考える。"),
                 LearnStep(index=5, title="確認", body="自力で説明できるかチェック。")]
    return LearnResponse(steps=steps[:5])

@app.get("/daily_phrase", response_model=PhraseResponse)
def daily_phrase(teacher: str = Query(...)):
    """
    先生の“ひとこと”。日替わりっぽさはtemperatureで揺らす。
    """
    sys = f"{persona(teacher)} 学習のモチベが上がる短い“ひとこと”を日本語で1行。"
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": "お願いします"}],
        temperature=1.0,
        max_tokens=60
    )
    return PhraseResponse(text=res.choices[0].message.content.strip())

# healthcheck
@app.get("/")
def root():
    return {"ok": True}
