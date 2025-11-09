from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os, json, httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

app = FastAPI(title="Homeroom API", version="1.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---------- Root / Health ----------
@app.get("/")
async def root():
    return {"status": "ok", "service": "homeroom", "endpoints": ["/health", "/question", "/consult"]}

@app.get("/health")
async def health():
    return {"ok": True}

# ---------- Schemas ----------
class QuestionIn(BaseModel):
    subject: str = Field(..., description="例: 国語/数学/英語/理科/社会")
    question: str
    image_base64: Optional[str] = None
    image_mime: Optional[str] = None  # "image/png" など

class LearnOut(BaseModel):
    steps: list[str]

class ConsultIn(BaseModel):
    text: str
    teacher: Optional[str] = None

class ConsultOut(BaseModel):
    reply: str

# ---------- OpenAI helpers ----------
async def _post_openai(payload: dict) -> dict:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(OPENAI_URL, headers=HEADERS, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text}")
        return r.json()

# ---------- Endpoints ----------
@app.post("/question", response_model=LearnOut)
async def question_api(inp: QuestionIn):
    system = {
        "role": "system",
        "content": ("あなたは優秀な日本語の家庭教師です。ユーザーの質問に対し、"
                    "分かりやすい『段階解説』を5〜6個の短いステップで示してください。"
                    "出力は JSON（steps: 文字列配列）のみ。")
    }

    user_content = [{
        "type": "text",
        "text": f"教科: {inp.subject}\n質問: {inp.question}\nJSONで steps のみ返して。"
    }]

    if inp.image_base64:
        mime = inp.image_mime or "image/png"
        data_url = inp.image_base64
        if not data_url.startswith("data:"):
            data_url = f"data:{mime};base64,{inp.image_base64}"
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            system,
            {"role": "user", "content": user_content}
        ]
    }

    data = await _post_openai(payload)
    content = data["choices"][0]["message"]["content"]

    try:
        obj = json.loads(content)
        steps = obj.get("steps", [])
        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
            raise ValueError
    except Exception:
        text = content if isinstance(content, str) else "問題の要点を一文で言い換えよう。"
        steps = [s for s in text.split("。") if s.strip()]
        if len(steps) < 2:
            steps = [text, "与えられた条件を整理しよう。", "必要な定義/公式を特定。", "途中式を丁寧に。", "結果の意味を確認。"]

    return LearnOut(steps=steps[:6])

@app.post("/consult", response_model=ConsultOut)
async def consult_api(inp: ConsultIn):
    system = {
        "role": "system",
        "content": "あなたは思いやりのある日本語カウンセラー。敬語で、LINEのように自然な1〜3文で返答。説教はしない。"
    }
    name = f"（{inp.teacher}）" if inp.teacher else ""
    user = {"role": "user", "content": name + inp.text}
    payload = {"model": "gpt-4o-mini", "temperature": 0.3, "messages": [system, user]}
    data = await _post_openai(payload)
    reply = data["choices"][0]["message"]["content"].strip()
    return ConsultOut(reply=reply)
