# main.py — FastAPI (口調反映版・フル置き換え)

import os
from typing import List, Optional, Literal, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

APP_NAME = "ai-recover"
APP_VERSION = "1.1.0"

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
SubjectID = Literal["国語","数学","英語","理科","社会","kokugo","suugaku","eigo","rika","shakai"]

# --- I/O models ---
class LearnIn(BaseModel):
    subject: SubjectID
    question: str
    imageBase64: Optional[str] = None
    imageMime: Optional[str] = None
    # 追加: 先生ID（省略可。指定が無いと教科から既定の先生を当てる）
    teacher_id: Optional[TeacherID] = None

class LearnOut(BaseModel):
    steps: List[str]

class CoachIn(BaseModel):
    teacher_id: TeacherID
    tasks_today: List[str] = []
    routines: List[str] = []

class CoachOut(BaseModel):
    tip: str

# ---------- Persona / Style ----------
def teacher_persona(tid: TeacherID) -> str:
    table: Dict[TeacherID, str] = {
        "toru": (
            "あなたは38歳の理科の大学教授・五十嵐トオル。温厚で落ち着いた敬語。"
            "観察→仮説→検証の順で筋道立てて説明する。比喩は控えめ。"
        ),
        "hazuki": (
            "あなたは28歳の国語の先生・水瀬葉月。やさしく親身な口調。"
            "要点→根拠→結論の型で導く。語尾は柔らかく、断定は優しめ。"
        ),
        "rika": (
            "あなたは13歳・IQ200の天才、小町リカ。テンポが速く、少しタメ口。"
            "言い切りで核心を突き、できた所は大げさに褒める。"
        ),
        "rei": (
            "あなたは15歳・IQ190の英語の先生、進藤怜。穏やかな丁寧語。"
            "落ち着いた励ましを添えつつ、段階的に教える。"
        ),
        "natsuki": (
            "あなたは25歳の社会の先生・小林夏樹。ぶっきらぼうだが面倒見がよい口調。"
            "因果と比較で要点をズバッと示す。"
        ),
    }
    return table[tid]

def teacher_style_rules(tid: TeacherID) -> str:
    styles: Dict[TeacherID, str] = {
        "hazuki": "語尾例:『〜だよ』『〜してみようね』。優しく丁寧、冗談は最小限。",
        "toru":   "敬語で論理的。接続語『まず』『次に』『だから』を適度に使う。",
        "rika":   "テンポ速めのタメ口。短文中心。相手を褒める相槌を1回入れる(例:『いいね！』)。",
        "rei":    "落ち着いた丁寧語。安心感のある励ましを最後に一言添える。",
        "natsuki":"やや砕けた口調。結論先出しOK。語尾『〜だな』『〜しよう』を時々使う。",
    }
    return styles[tid]

def subject_hint(subj: SubjectID) -> str:
    m = {
        "国語":"国語","kokugo":"国語",
        "数学":"数学","suugaku":"数学",
        "英語":"英語","eigo":"英語",
        "理科":"理科","rika":"理科",
        "社会":"社会","shakai":"社会",
    }
    return m.get(subj, "学習")

def default_teacher_for(subj: SubjectID) -> TeacherID:
    mapping: Dict[str, TeacherID] = {
        "国語":"hazuki","kokugo":"hazuki",
        "数学":"rika","suugaku":"rika",
        "英語":"rei","eigo":"rei",
        "理科":"toru","rika":"toru",
        "社会":"natsuki","shakai":"natsuki",
    }
    return mapping.get(subj, "hazuki")  # フォールバック

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

# ---------- consult ----------
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
        teacher_persona(teacher) + "\n" +
        "ルール: 口調と性格を必ず維持。段落は短く、必要なら箇条書き。"
        "最後に次の一歩を1行だけ提案する。\n" +
        teacher_style_rules(teacher)
    )
    reply = chat_once(MODEL_CONSULT, system, text)
    return {"reply": reply}

# ---------- question（口調反映・ステップ出力） ----------
@app.post("/question", response_model=LearnOut, tags=["ai"])
def question(inb: LearnIn):
    subj = subject_hint(inb.subject)
    teacher: TeacherID = inb.teacher_id or default_teacher_for(inb.subject)

    system = (
        teacher_persona(teacher) + "\n" +
        teacher_style_rules(teacher) + "\n" +
        "あなたは学習コーチ。以下の厳密な出力形式に従う。\n"
        "【出力形式】\n"
        "・日本語で5〜7ステップ。\n"
        "・各ステップは最大70字、1行に収める。\n"
        "・前置き/後書き/まとめは不要。手順だけを列挙。\n"
        "・各行の先頭に 1〜7 の番号を付けてもよい。"
    )

    if inb.imageBase64 and inb.imageMime:
        user: Any = [
            {"type": "text", "text": f"教科:{subj}\n質問:{inb.question}\n5〜7ステップで説明して。"},
            {"type": "image_url", "image_url": {"url": f"data:{inb.imageMime};base64,{inb.imageBase64}" }},
        ]
    else:
        user = f"教科:{subj}\n質問:{inb.question}\n5〜7ステップで説明して。"

    text = chat_once(MODEL_LEARN, system, user)

    # 行→steps への整形（番号や「ステップn」を除去）
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
        "\n" + teacher_style_rules(inb.teacher_id)
    )
    user = (
        "今日のToDo: " + ("、".join(inb.tasks_today) if inb.tasks_today else "なし") + "\n" +
        "ルーティン: " + ("、".join(inb.routines) if inb.routines else "なし")
    )
    tip = chat_once(MODEL_CONSULT, system, user)
    return CoachOut(tip=tip)
