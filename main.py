# main.py — FastAPI（相談タブだけ超短文・人間っぽい間）

import os
from typing import List, Optional, Literal, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import re

APP_NAME = "ai-recover"
APP_VERSION = "1.1.2"  # ← 相談タブの短文＆間 修正版

# ===== OpenAI client =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
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
    teacher_id: Optional[TeacherID] = None  # 学習タブは据え置き（任意）

class LearnOut(BaseModel):
    steps: List[str]

class CoachIn(BaseModel):
    teacher_id: TeacherID
    tasks_today: List[str] = []
    routines: List[str] = []

class CoachOut(BaseModel):
    tip: str

# ---------- Personas ----------
def teacher_persona(tid: TeacherID) -> str:
    table: Dict[TeacherID, str] = {
        "toru":   "あなたは38歳の理科の大学教授・五十嵐トオル。温厚で落ち着いた敬語。観察→仮説→検証で筋道立てる。",
        "hazuki": "あなたは28歳の国語の先生・水瀬葉月。やさしく親身。要点→根拠→結論で導く。語尾は柔らかめ。",
        "rika":   "あなたは13歳・IQ200の天才、小町リカ。テンポ速め、少しタメ口。できた所はよく褒める。",
        "rei":    "あなたは15歳・IQ190の英語の先生、進藤怜。穏やかな丁寧語。安心感のある励ましを添える。",
        "natsuki":"あなたは25歳の社会の先生・小林夏樹。ぶっきらぼうだが面倒見が良い。因果と比較が得意。",
    }
    return table[tid]

def teacher_style_rules(tid: TeacherID) -> str:
    styles: Dict[TeacherID, str] = {
        "hazuki": "語尾『〜だよ』『〜してみようね』。やさしく短く。",
        "toru":   "敬語で簡潔。『まず/次に』を最小限に。",
        "rika":   "短文・タメ口・テンポ早め。相槌1回（例『いいね！』）。",
        "rei":    "落ち着いた丁寧語。安心させる一言を最後に短く。",
        "natsuki":"砕けた口調。結論先出し。語尾『〜だな』『〜しよう』を時々。",
    }
    return styles[tid]

def subject_hint(subj: SubjectID) -> str:
    m = {"国語":"国語","kokugo":"国語","数学":"数学","suugaku":"数学","英語":"英語","eigo":"英語","理科":"理科","rika":"理科","社会":"社会","shakai":"社会"}
    return m.get(subj, "学習")

def default_teacher_for(subj: SubjectID) -> TeacherID:
    mapping: Dict[str, TeacherID] = {
        "国語":"hazuki","kokugo":"hazuki",
        "数学":"rika","suugaku":"rika",
        "英語":"rei","eigo":"rei",
        "理科":"toru","rika":"toru",
        "社会":"natsuki","shakai":"natsuki",
    }
    return mapping.get(subj, "hazuki")

# ---------- OpenAI helper ----------
def require_client():
    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set on the server")

def chat_once(model: str, system: str, user: Any, temperature: float = 0.6) -> str:
    require_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=temperature,
        )
        content = resp.choices[0].message.content or ""
        if not content.strip():
            raise HTTPException(status_code=502, detail="empty response from OpenAI")
        return content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream_error: {e}")

# ---------- Meta ----------
@app.get("/", tags=["meta"])
def root():
    return {"service": APP_NAME, "version": APP_VERSION, "docs": "/docs", "status": "ok"}

@app.get("/health", tags=["meta"])
def health():
    return {"ok": True}

# ---------- consult（ここだけ短文化＋“間”） ----------
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
        jp_map = {"水瀬葉月":"hazuki","葉月":"hazuki","進藤怜":"rei","怜":"rei","小町リカ":"rika","リカ":"rika","五十嵐トオル":"toru","トオル":"toru","小林夏樹":"natsuki","夏樹":"natsuki"}
        for k,v in jp_map.items():
            if s == k.lower():
                return v  # type: ignore
    if isinstance(raw, int):
        order = ["hazuki","rika","rei","toru","natsuki"]
        if 0 <= raw < len(order):
            return order[raw]  # type: ignore
    return "hazuki"

# 先生ごとの“軽い入り”と“短い締め”
_OPENERS: Dict[TeacherID, str] = {
    "hazuki": "…そっか。",
    "toru":   "なるほど。",
    "rika":   "ん、分かった！",
    "rei":    "…うん。",
    "natsuki":"…おう。",
}
_CLOSERS: Dict[TeacherID, str] = {
    "hazuki": "無理しないでね。",
    "toru":   "詳しくは一つずつ見ていこう。",
    "rika":   "だいじょうぶ。私がついてる。",
    "rei":    "大丈夫、一緒に整えていこう。",
    "natsuki":"よし、ここからだな。",
}

# LLM 指示：2文以内＋最後に短い問い
def _consult_system(t: TeacherID) -> str:
    return (
        teacher_persona(t) + "\n" +
        teacher_style_rules(t) + "\n" +
        "あなたはチャット相談の相手。出力は以下を厳守：\n"
        "・最大全角200文字・2文以内。\n"
        "・必要なら文頭にごく短い相槌（1語〜5語）。\n"
        "・最後は短い質問を1つだけ返す。\n"
        "・箇条書き/長文/要約/結論の羅列は禁止。\n"
        "・同じ内容の繰り返しは禁止。\n"
        "・丁寧すぎる定型文は避け、自然な口語で。\n"
    )

# 念のためのサーバー側短縮（LLMの暴走止め）
def _shrink_two_sentences(s: str, limit: int = 200) -> str:
    # 改行→スペース
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s.strip())
    # 箇条書き接頭辞を除去
    s = re.sub(r"^[\-\•\・\*]\s*", "", s, flags=re.MULTILINE)
    # 文スプリット（。！？）
    parts = re.split(r"(?<=[。.!?！？])\s*", s)
    parts = [p for p in parts if p]
    if len(parts) > 2:
        s = parts[0] + (" " if not parts[0].endswith(("。","!","?","！","？")) else "") + parts[1]
    # 文字数制限
    if len(s) > limit:
        s = s[:limit-1] + "…"
    # 末尾に過剰な定型を置かない
    s = re.sub(r"(よろしくお願いします|ご安心ください).*?$", "", s)
    return s.strip()

@app.post("/consult", tags=["ai"])
async def consult(request: Request):
    """
    相談API：先生ごとの口調で『短い相槌→一言→短い質問』に強制。
    レスポンス形式は従来通り { reply: string } のみ（フロント改修不要）。
    """
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        payload = {}

    text = _pick_str(payload, ["text","message","content"])
    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    teacher: TeacherID = _pick_teacher(payload)

    opener = _OPENERS.get(teacher, "")
    closer = _CLOSERS.get(teacher, "")

    system = _consult_system(teacher)
    raw = chat_once(MODEL_CONSULT, system, text, temperature=0.7)
    body = _shrink_two_sentences(raw)

    # “相槌”と“軽い締め”を足す（重複しないように）
    if opener and not body.lstrip().startswith(opener):
        body = f"{opener} {body}"
    if closer and not body.endswith(("。","!","！","？","?")):
        body = f"{body} {closer}"

    return {"reply": body}

# ---------- question（学習タブ：据え置き・口調だけ反映） ----------
@app.post("/question", response_model=LearnOut, tags=["ai"])
def question(inb: LearnIn):
    subj = subject_hint(inb.subject)
    teacher: TeacherID = inb.teacher_id or default_teacher_for(inb.subject)

    system = (
        teacher_persona(teacher) + "\n" +
        teacher_style_rules(teacher) + "\n" +
        "あなたは学習コーチ。出力は厳密に：\n"
        "・日本語で5〜7ステップ。\n"
        "・各ステップは最大70字、1行のみ。\n"
        "・前置き/まとめは不要。手順だけを列挙。"
    )

    if inb.imageBase64 and inb.imageMime:
        user: Any = [
            {"type": "text", "text": f"教科:{subj}\n質問:{inb.question}\n5〜7ステップで説明して。"},
            {"type": "image_url", "image_url": {"url": f"data:{inb.imageMime};base64,{inb.imageBase64}" }},
        ]
    else:
        user = f"教科:{subj}\n質問:{inb.question}\n5〜7ステップで説明して。"

    text = chat_once(MODEL_LEARN, system, user)

    # 行整形（番号/『ステップ』等を取り除く）
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

# ---------- todo/coach（据え置き） ----------
@app.post("/todo/coach", response_model=CoachOut, tags=["ai"])
def todo_coach(inb: CoachIn):
    persona = teacher_persona(inb.teacher_id)
    system = (
        f"{persona}\n"
        "ToDoとルーティンから今日のフォーカスを1〜2文で提案。"
        "言い切りで前向きに、実行順や所要時間の目安を入れてもよい。\n" +
        teacher_style_rules(inb.teacher_id)
    )
    user = (
        "今日のToDo: " + ("、".join(inb.tasks_today) if inb.tasks_today else "なし") + "\n" +
        "ルーティン: " + ("、".join(inb.routines) if inb.routines else "なし")
    )
    tip = chat_once(MODEL_CONSULT, system, user)
    return CoachOut(tip=_shrink_two_sentences(tip, limit=180))
