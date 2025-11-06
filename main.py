import os
from typing import List, Optional, Literal, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ---- OpenAI optional ----
USE_FAKE = os.getenv("USE_FAKE", "false").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

try:
    from openai import OpenAI  # openai>=1.0
    oai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and not USE_FAKE else None
except Exception:
    oai_client = None

app = FastAPI(title="homeroom-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TeacherKey = Literal["hazuki", "toru", "rika", "rei", "natsuki"]

class HistoryMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ConsultIn(BaseModel):
    teacher: TeacherKey
    message: str
    history: Optional[List[HistoryMsg]] = None

class ConsultOut(BaseModel):
    reply: str
    teacher: TeacherKey

SYSTEMS: Dict[TeacherKey, str] = {
    "hazuki": (
        "あなたは水瀬葉月。28歳の優しいお姉さん先生（国語）。"
        "丁寧語ベースで柔らかく、たまに軽い意地悪や親父ギャグを1つ混ぜてもよい。"
        "相手を肯定し、最後に次の一歩を1文で提案してください。"
    ),
    "toru": (
        "あなたは五十嵐トオル。38歳の温厚な大学教授（理科）。"
        "落ち着いた敬語で、論理的に要点を3つ以内で整理。"
        "最後に穏やかな励ましを1文。"
    ),
    "rika": (
        "あなたは小町リカ。13歳のギフテッド（数学）。"
        "天真爛漫で少し生意気。良い点は大げさに褒める。"
        "最後は行動を促す短い指示。"
    ),
    "rei": (
        "あなたは進藤怜。15歳の穏やかな帰国子女（英語）。"
        "結論を先に簡潔に。英語表現の例を短く示してもよい。"
    ),
    "natsuki": (
        "あなたは小林夏樹。25歳、ぶっきらぼうだが優しい兄貴分（社会）。"
        "少しからかいながら実用的なアドバイスを出し、最後に背中を押す。"
    ),
}

@app.get("/teachers")
def teachers():
    return {
        "teachers": [
            {"id": "hazuki", "name": "水瀬 葉月", "subject": "国語"},
            {"id": "toru", "name": "五十嵐 トオル", "subject": "理科"},
            {"id": "rika", "name": "小町 リカ", "subject": "数学"},
            {"id": "rei", "name": "進藤 怜", "subject": "英語"},
            {"id": "natsuki", "name": "小林 夏樹", "subject": "社会"},
        ]
    }

@app.post("/consult", response_model=ConsultOut)
def consult(body: ConsultIn):
    teacher = body.teacher
    user_text = body.message.strip()
    if not user_text:
        return ConsultOut(reply="まずは気になっていることを一行で教えてください。", teacher=teacher)

    if USE_FAKE or not oai_client:
        reply = fake_reply(teacher, user_text)
        return ConsultOut(reply=reply, teacher=teacher)

    messages = [{"role": "system", "content": SYSTEMS[teacher]}]
    if body.history:
        for h in body.history[-6:]:
            messages.append({"role": h.role, "content": h.content[:1500]})
    messages.append({"role": "user", "content": user_text})

    try:
        res = oai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        reply = res.choices[0].message.content.strip()
    except Exception as e:
        reply = f"サーバー側でエラーが発生しました：{e}"

    return ConsultOut(reply=reply, teacher=teacher)

def fake_reply(teacher: TeacherKey, text: str) -> str:
    if teacher == "hazuki":
        return f"うん、まずは深呼吸しよ。{text} のポイントを一緒に整えようか。無理せず一歩ずつ、今は『5分だけ手をつける』でどう？"
    if teacher == "toru":
        return f"観点を三つに整理します。(1) 前提 (2) 仮説 (3) 検証。まずは最小の実験から始めましょう。焦らず進めれば大丈夫ですよ。"
    if teacher == "rika":
        return f"それ、面白いね！結論だけ言うと“今やる”。できたら私、全力で褒めるから。まずは1問、スタート。"
    if teacher == "rei":
        return f"First, keep it simple. 要点を短くまとめるのが近道です。Try this: write a 1-sentence goal, then act for 5 minutes."
    if teacher == "natsuki":
        return f"よし、グダグダ言う前にやるぞ。{text} は30分スプリントで決まり。終わったら報告しろ、褒め倒してやる。"
    return "OK. まずは一行で状況を教えて。"
