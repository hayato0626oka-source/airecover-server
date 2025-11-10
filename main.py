import os
from typing import List, Optional, Literal, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

APP_NAME = "ai-recover"
APP_VERSION = "1.0.0"

# ===== OpenAI client =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    # 環境変数が無いなら起動は通すが、API呼出し時に 503 を返す
    client: Optional[OpenAI] = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_CONSULT = os.getenv("MODEL_CONSULT", "gpt-4o-mini")
MODEL_LEARN   = os.getenv("MODEL_LEARN",   "gpt-4o-mini")

# ===== FastAPI =====
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Types ----------
TeacherID = Literal["hazuki", "toru", "rika", "rei", "natsuki"]
SubjectID = Literal[
    "国語","数学","英語","理科","社会","kokugo","suugaku","eigo","rika","shakai"
]

class LearnIn(BaseModel):
    subject: SubjectID
    question: str
    imageBase64: Optional[str] = None
    imageMime: Optional[str] = None

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
            "あなたは38歳の理科の大学教授・五十嵐トオル。温厚で落ち着いた敬語。"
            "観察→仮説→検証の順に筋道立てて説明する。"
        ),
        "hazuki": (
            "あなたは28歳の国語の先生・水瀬葉月。やさしく親身。"
            "要点→根拠→結論の型で導く。冗談は控えめに短く。"
        ),
        "rika": (
            "あなたは13歳・IQ200の天才、小町リカ。テンポよく核心を突く。"
            "少し生意気だが、できた所は大きく褒める。"
        ),
        "rei": (
            "あなたは15歳・IQ190の英語の先生、進藤怜。穏やか。"
            "語順とチャンクで段階的に教える。"
        ),
        "natsuki": (
            "あなたは25歳の社会の先生・小林夏樹。ぶっきらぼうだが面倒見がよい。"
            "因果関係と比較で分かりやすく説明する。"
        ),
    }
    return table[tid]

def subject_hint(subj: SubjectID) -> str:
    m = {
        "国語":"国語","kokugo":"国語",
        "数学":"数学","suugaku":"数学",
        "英語":"英語","eigo":"英語",
        "理科":"理科","rika":"理科",
        "社会":"社会","shakai":"社会",
    }
    return m.get(subj, "学習")

# ---------- OpenAI helper ----------
def require_client():
    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set on the server")

def chat_once(model: str, system: str, user: Any) -> str:
    require_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.6,
        )
        content = resp.choices[0].message.content or ""
        if not content.strip():
            raise HTTPException(status_code=502, detail="empty response from OpenAI")
        return content
    except HTTPException:
        raise
    except Exception as e:
        # OpenAI 側の失敗は 502 に統一
        raise HTTPException(status_code=502, detail=f"upstream_error: {e}")

# ---------- Meta ----------
@app.get("/", tags=["meta"])
def root():
    return {"service": APP_NAME, "version": APP_VERSION, "docs": "/docs", "status": "ok"}

@app.get("/health", tags=["meta"])
def health():
    return {"ok": True}

# ---------- consult（柔軟受付・フォールバック無し） ----------
def _pick_str(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _pick_teacher(d: Dict[str, Any]) -> TeacherID:
    raw = d.get("teacher_id", d.get("teacherId", d.get("teacher")))
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ["hazuki","rei","rika","toru","natsuki"]:
            return s  # type: ignore
        jp_map = {
            "水瀬葉月":"hazuki","葉月":"hazuki",
            "進藤怜":"rei","怜":"rei",
            "小町リカ":"rika","リカ":"rika",
            "五十嵐トオル":"toru","トオル":"toru",
            "小林夏樹":"natsuki","夏樹":"natsuki",
        }
        for k,v in jp_map.items():
            if s == k.lower():
                return v  # type: ignore
    if isinstance(raw, int):
        order = ["hazuki","rika","rei","toru","natsuki"]
        if 0 <= raw < len(order):
            return order[raw]  # type: ignore
    return "hazuki"  # default

@app.post("/consult", tags=["ai"])
async def consult(request: Request):
    """
    相談API。キー名の揺れ（text/message/content, teacher_id/teacherId/teacher）を吸収。
    OpenAI 失敗時は 502 を返す。本文が無ければ 422。
    """
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        payload = {}

    text = _pick_str(payload, ["text","message","content"])
    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    teacher: TeacherID = _pick_teacher(payload)

    system = (
        teacher_persona(teacher)
        + "\nルール: 口調と性格を維持。段落は短く、必要なら箇条書き。"
        + "最後に次の一歩を1行だけ提案する。"
    )
    reply = chat_once(MODEL_CONSULT, system, text)
    return {"reply": reply}

# ---------- question（既存仕様） ----------
@app.post("/question", response_model=LearnOut, tags=["ai"])
def question(inb: LearnIn):
    system = (
        "あなたは優秀な学習コーチ。教科に合わせて"
        "『ステップ1〜7』形式で、1ステップ最大70字。"
        "冗長な前置きなく直接手順を書く。"
    )
    subj = subject_hint(inb.subject)

    if inb.imageBase64 and inb.imageMime:
        user: Any = [
            {"type": "text", "text": f"教科:{subj}\n質問:{inb.question}\n5〜7ステップで説明して。"},
            {"type": "image_url", "image_url": {"url": f"data:{inb.imageMime};base64,{inb.imageBase64}" }},
        ]
    else:
        user = f"教科:{subj}\n質問:{inb.question}\n5〜7ステップで説明して。"

    text = chat_once(MODEL_LEARN, system, user)
    lines = [s.strip(" ・-　").strip() for s in text.splitlines() if s.strip()]
    steps: List[str] = []
    for s in lines:
        cleaned = s
        for token in ["ステップ","Step","STEP","手順"]:
            cleaned = cleaned.replace(token, "")
        cleaned = cleaned.lstrip("0123456789.：:）) 」]　").strip()
        if cleaned:
            steps.append(cleaned)
    if not steps:
        steps = [text]
    if len(steps) > 8:
        steps = steps[:4] + [" / ".join(steps[4:])]
    return LearnOut(steps=steps)

# ---------- todo/coach ----------
@app.post("/todo/coach", response_model=CoachOut, tags=["ai"])
def todo_coach(inb: CoachIn):
    persona = teacher_persona(inb.teacher_id)
    system = (
        f"{persona}\n"
        "ToDoとルーティンから今日のフォーカスを1〜2文で提案。"
        "言い切りで前向きに、実行順や所要時間の目安を入れてもよい。"
    )
    user = (
        "今日のToDo: " + ("、".join(inb.tasks_today) if inb.tasks_today else "なし") + "\n" +
        "ルーティン: " + ("、".join(inb.routines) if inb.routines else "なし")
    )
    tip = chat_once(MODEL_CONSULT, system, user)
    return CoachOut(tip=tip)
