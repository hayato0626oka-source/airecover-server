import os, time, requests, traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"

app = FastAPI(title="AI Recover API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class QuestionIn(BaseModel):
    question: str

@app.get("/")
def root():
    return {"ok": True, "service": "airecover"}

def openai_chat(messages, retries=1, timeout_sec=30):
    """OpenAI呼び出し。失敗時は本文を返して原因が見えるようにする。"""
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": MODEL, "messages": messages, "max_tokens": 800},
                timeout=timeout_sec,
            )
            # 4xx/5xxは本文ごと返す（500で落とさない）
            if r.status_code >= 400:
                return f"OpenAI error {r.status_code}: {r.text[:300]}"
            j = r.json()
            return j.get("choices", [{}])[0].get("message", {}).get("content", "(no content)")
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    return f"Server exception while calling OpenAI: {last_err}"

@app.post("/question")
def question_api(data: QuestionIn):
    try:
        # 切り分け用：ダミー応答
        if os.getenv("USE_FAKE") == "1":
            return f"【ダミー】「{data.question}」への解説: 条件を整理→式を立てる→同じ操作で両辺を処理→答え。"

        if not OPENAI_API_KEY:
            return "Server not configured: missing OPENAI_API_KEY."

        msgs = [
  {"role": "system", "content":
   "You are a supportive Japanese tutor. Answer in clean Japanese Markdown. \
    Provide a short intro, then a numbered procedure using '1.','2.','3.' lines. \
    Each step should begin with a short title, then one short sentence detail. \
    Avoid LaTeX and code fences. Keep lines simple."},
  {"role": "user", "content": data.question},
]
        return openai_chat(msgs)
    except Exception as e:
        # 500にせず、本文でエラーを返す
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"
