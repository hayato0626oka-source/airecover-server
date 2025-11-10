import os
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

APP_NAME = "ai-recover"
APP_VERSION = "1.0.0"

# ==== OpenAI client ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
MODEL_CONSULT = os.getenv("MODEL_CONSULT", "gpt-4o-mini")
MODEL_LEARN   = os.getenv("MODEL_LEARN",   "gpt-4o-mini")

# ==== FastAPI ====
app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # 必要なら絞ってOK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
TeacherID = Literal["hazuki", "toru", "rika", "rei", "natsuki"]
SubjectID = Literal["国語","数学","英語","理科","社会","kokugo","suugaku","eigo","rika","shakai"]

class ConsultIn(BaseModel):
    text: str = Field(..., min_length=1)
    teacher_id: TeacherID
    user_name: Optional[str] = None
    user_age: Optional[int] = None
    user_occupation: Optional[str] = None  # "小学生/中学生/高校生/大学・専門学生/その他"

class ConsultOut(BaseModel):
    reply: str

class LearnIn(BaseModel):
    subject: SubjectID
    question: str
    imageBase64: Optional[str] = None
    imageMime: Optional[str] = None   # "image/png" / "image/jpeg"

class LearnOut(BaseModel):
    steps: List[str]

class CoachIn(BaseModel):
    teacher_id: TeacherID
    tasks_today: List[str] = []
    routines: List[str] = []

class CoachOut(BaseModel):
    tip: str

# ---------- Persona ----------
def teacher_persona(tid: TeacherID) -> str:
    table: Dict[TeacherID, str] = {
        "toru": (
            "あなたは38歳の理科の大学教授・五十嵐トオル。温厚で落ち着いた口調。"
            "敬語で、観察→仮説→検証の順に筋道立てて説明します。"
        ),
        "hazuki": (
            "あなたは28歳の国語の先生・水瀬葉月。やさしく親身。"
            "たまに軽い冗談を挟むが言い過ぎない。要点→根拠→結論の型で導きます。"
        ),
        "rika": (
            "あなたは13歳の天才・小町リカ。IQ200。テンポよく核心を突く。"
            "少し生意気でズバッと言うが、できた所は素直に大きく褒めます。"
        ),
        "rei": (
            "あなたは15歳の英語の先生・進藤怜。IQ190。穏やかでやさしい。"
            "焦らせず、語順とチャンクで段階的に教えます。"
        ),
        "natsuki": (
            "あなたは25歳の社会の先生・小林夏樹。ぶっきらぼうに見えるが面倒見がよい。"
            "因果関係と比較で分かりやすく、時々軽くからかうツッコミを入れるが優しい。"
        ),
    }
    return table[tid]

def subject_hint(subj: SubjectID) -> str:
    m = {
        "国語": "国語", "kokugo":"国語",
        "数学": "数学", "suugaku":"数学",
        "英語": "英語", "eigo":"英語",
        "理科": "理科", "rika":"理科",
        "社会": "社会", "shakai":"社会",
    }
    return m.get(subj, "学習")

# ---------- OpenAI helper ----------
def ensure_client():
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI API key is not set on the server.")

def chat_once(model: str, system: str, user: Any) -> str:
    """
    user: str もしくは vision 用の content(list)
    """
    ensure_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.6,
    )
    return resp.choices[0].message.content or ""

# ---------- Routes ----------
@app.get("/", tags=["meta"])
def root():
    return {"service": APP_NAME, "version": APP_VERSION, "docs": "/docs", "status": "ok"}

@app.get("/health", tags=["meta"])
def health():
    return {"ok": True}

# ---- consult ----
@app.post("/consult", response_model=ConsultOut, tags=["ai"])
def consult(inb: ConsultIn):
    persona = teacher_persona(inb.teacher_id)
    profile = []
    if inb.user_name: profile.append(f"呼び名: {inb.user_name}")
    if inb.user_age:  profile.append(f"年齢: {inb.user_age}")
    if inb.user_occupation: profile.append(f"職業: {inb.user_occupation}")
    profile_text = " / ".join(profile) if profile else "（ユーザープロファイル情報なし）"

    system = (
        f"{persona}\n"
        "次のルールを守ってください:\n"
        "・口調と性格は常に維持する\n"
        "・段落は短く、必要に応じて箇条書きを使う\n"
        "・最後に1行だけ、次の行動提案を添える\n"
    )
    user = f"ユーザープロファイル: {profile_text}\n相談内容: {inb.text}"
    reply = chat_once(MODEL_CONSULT, system, user)
    return ConsultOut(reply=reply)

# ---- question/learn ----
@app.post("/question", response_model=LearnOut, tags=["ai"])
def question(inb: LearnIn):
    """
    5〜7ステップで解説を返す。画像base64があればマルチモーダルで渡す。
    """
    system = (
        "あなたは優秀な学習コーチです。出題の教科に合わせて、"
        "『ステップ1〜7』の形で、1ステップあたり最大70字程度、"
        "冗長な前置きなく直接手順を書く。必要なら式や例も簡潔に。\n"
        "出力は箇条書きテキストのみ。"
    )
    subj = subject_hint(inb.subject)
    # visionメッセージ構築
    if inb.imageBase64 and inb.imageMime:
        user_content = [
            {"type": "text", "text": f"教科: {subj}\n質問: {inb.question}\n"
                                     f"5〜7ステップの手順で説明して。"},
            {"type": "image_url",
             "image_url": {"url": f"data:{inb.imageMime};base64,{inb.imageBase64}"}}
        ]
    else:
        user_content = f"教科: {subj}\n質問: {inb.question}\n5〜7ステップの手順で説明して。"

    text = chat_once(MODEL_LEARN, system, user_content)
    # ステップ抽出（箇条書き/番号両対応）
    lines = [s.strip(" ・-　").strip() for s in text.splitlines() if s.strip()]
    steps: List[str] = []
    for s in lines:
        # "1. xxx" / "ステップ1: xxx" などを削る
        cleaned = s
        for token in ["ステップ", "Step", "STEP", "手順"]:
            cleaned = cleaned.replace(token, "")
        cleaned = cleaned.lstrip("0123456789.：:）) 」]　").strip()
        if cleaned:
            steps.append(cleaned)
    # フォールバック
    if not steps:
        steps = [text]

    # 8件以上なら前半4＋残りを結合
    if len(steps) > 8:
        merged = steps[:4]
        merged.append(" / ".join(steps[4:]))
        steps = merged
    return LearnOut(steps=steps)

# ---- todo/coach ----
@app.post("/todo/coach", response_model=CoachOut, tags=["ai"])
def todo_coach(inb: CoachIn):
    persona = teacher_persona(inb.teacher_id)
    system = (
        f"{persona}\n"
        "ToDo・ルーティンを見て、今日のフォーカスを1〜2文で提案。"
        "言い切りで前向きに、実行順序や所要時間の目安を含めてもよい。"
    )
    user = (
        "今日のToDo: " + ("、".join(inb.tasks_today) if inb.tasks_today else "なし") + "\n" +
        "ルーティン: " + ("、".join(inb.routines) if inb.routines else "なし")
    )
    tip = chat_once(MODEL_CONSULT, system, user)
    return CoachOut(tip=tip)
