import os
import time
import traceback
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ====== 環境変数 ======
PROVIDER = os.getenv("PROVIDER", "openai")  # "openai" / "groq" / "openrouter"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4o-mini")   # 省コスト既定
USE_FAKE = os.getenv("USE_FAKE", "0")       # "1" ならダミー即レス

# ====== FastAPI 基本設定 ======
app = FastAPI(title="AI Recover API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 必要に応じて制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== I/O モデル ======
class QuestionIn(BaseModel):
    question: str
    teacher_key: Optional[str] = None
    student_nick: Optional[str] = None

class ConsultIn(BaseModel):
    message: str
    teacher_key: Optional[str] = None
    student_nick: Optional[str] = None

class TodoTask(BaseModel):
    id: Optional[str] = None
    title: str
    due: Optional[float] = None  # epoch seconds
    done: bool = False

class TodoCoachIn(BaseModel):
    tasks: List[TodoTask]
    teacher_key: Optional[str] = None
    student_nick: Optional[str] = None

# ====== ルート/ヘルス ======
@app.get("/")
def root():
    return {"ok": True, "service": "airecover", "provider": PROVIDER, "model": MODEL, "version": "1.2.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ====== LLM 呼び出し（プロバイダ切替対応・落ちない設計） ======
def chat_api(messages, retries: int = 1, timeout_sec: int = 30) -> str:
    """
    各社の OpenAI 互換APIに POST。
    4xx/5xx は本文ごと返し、例外は握りつぶして文字列化。500を出さない。
    """
    if PROVIDER == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        key = GROQ_API_KEY
    elif PROVIDER == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        key = OPENROUTER_API_KEY
    else:
        url = "https://api.openai.com/v1/chat/completions"
        key = OPENAI_API_KEY

    if not key:
        return f"Server not configured: missing API key for provider '{PROVIDER}'."

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 300,   # コスト抑制
        "temperature": 0.5,  # 相談は少しだけ揺らぎ許容
    }

    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
            if r.status_code >= 400:
                return f"{PROVIDER} error {r.status_code}: {r.text[:500]}"
            j = r.json()
            content = (j.get("choices", [{}])[0].get("message", {}).get("content")) or ""
            return content.strip() or "(no content)"
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    return f"Server exception while calling provider '{PROVIDER}': {last_err}"

# ====== 担任プロファイル ======
PERSONA = {
    # 画像：咲（クール系お姉さん）
    "saki": {
        "name": "咲",
        "style": "クールでロジカル。余計な絵文字なし、語尾は端的。相手を見下さないがキレ味あり。",
        "greeting": "要点から行くね。"
    },
    # 画像：ナツキ（チャラいお兄さん）
    "natsuki": {
        "name": "ナツキ",
        "style": "軽快でフレンドリー。タメ口9割、ほどよくノリ良い相づち。短文多め。",
        "greeting": "よっ、任せろ。"
    },
    # 画像：詩織（ふわふわ系お姉さん）
    "shiori": {
        "name": "詩織",
        "style": "やさしく包む。ゆるめの敬体。絵文字は控えめに1つまで。",
        "greeting": "うん、まずは落ち着こ。"
    },
    # 追加キャラ：メガネの優男
    "megane": {
        "name": "湊",
        "style": "丁寧で静か、観察的。相手の言葉を短く反射して受け止める。穏やかな助言。",
        "greeting": "話してくれてありがとう。"
    },
    # 追加キャラ：元気系超イケメン
    "genki": {
        "name": "蓮",
        "style": "明るく前向き。テンポ速め。短い応援＋次の一手を必ず提示。",
        "greeting": "いこいこ！"
    },
}

def persona_prompt(key: Optional[str], student_nick: Optional[str]) -> str:
    k = (key or "").strip().lower()
    p = PERSONA.get(k, PERSONA["saki"])
    nick = student_nick or "あなた"
    # 会話スタイルの核。相談・学習・ToDoそれぞれで末尾に指示を足す
    return (
        f"あなたは日本語で会話するカウンセラー。キャラ設定：{p['name']}。"
        f"話し方：{p['style']} "
        f"出力は短めの自然な会話文を2〜3文。過度な長文禁止。"
        f"必ず最後に1つだけ優しい質問で返す。"
        f"禁止：『SUGGEST:〜』等のタグ、テンプレ励ましの連発、箇条書き。"
        f"呼びかけは「{nick}」。"
        f"最初の一言は「{p['greeting']}」で始める。"
    )

# ====== /question ======
@app.post("/question")
def question_api(data: QuestionIn):
    try:
        if USE_FAKE == "1":
            who = PERSONA.get((data.teacher_key or "saki").lower(), PERSONA["saki"])["name"]
            return (f"{who}：ざっくりの流れだよ。\n"
                    "1. 条件整理\n2. 式を立てる\n3. 計算\n4. 検算\n")

        # teacher_key があれば担任口調で「学習解説モード」
        if data.teacher_key:
            system = (
                persona_prompt(data.teacher_key, data.student_nick) +
                " 出力は**学習解説モード**。最初に一行の要約。"
                "その後、3〜6個の**番号付きステップ**を短文で。"
                "式はプレーン（例：2x+3=7 → 2x=4 → x=2）。"
                "日本語Markdown、記号は控えめ、簡潔に。"
                "最後は短い確認質問で締める。"
            )
        else:
            # 従来の中立チューター
            system = (
                "You are a supportive Japanese tutor. "
                "Answer in clean Japanese Markdown for iOS display. "
                "Start with a one-line summary. Then provide a numbered procedure (3–6 steps). "
                "No LaTeX or code fences. Equations plain like 2x+3=7 → 2x=4 → x=2."
            )

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.question},
        ]
        out = chat_api(msgs)
        return clean_out(out)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== /consult ======
@app.post("/consult")
def consult_api(data: ConsultIn):
    try:
        if USE_FAKE == "1":
            who = PERSONA.get((data.teacher_key or "saki").lower(), PERSONA["saki"])["name"]
            return f"{who}：それ、まずは一息つこ。次にどうしたい？"

        system = persona_prompt(data.teacher_key, data.student_nick)
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.message},
        ]
        out = chat_api(msgs)
        return clean_out(out)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== /todo/coach ======
@app.post("/todo/coach")
def todo_coach_api(data: TodoCoachIn):
    """
    タスク配列から“今日の一言”を短文で返す。
    存在しない環境でも簡単に動くよう、/consult スタイルのプロンプトを内部で組む。
    """
    try:
        if USE_FAKE == "1":
            who = PERSONA.get((data.teacher_key or "saki").lower(), PERSONA["saki"])["name"]
            return f"{who}：まず1分だけ手を付けよ。進み始めれば、勢いは出るよ。どう始める？"

        pending = [t for t in data.tasks if not t.done]
        count = len(pending)
        first_title = pending[0].title if pending else ""
        system = (
            persona_prompt(data.teacher_key, data.student_nick) +
            " 出力は**短い一言**モード。10〜40字で、今すぐ着手できる行動を一つだけ提案。"
            "命令形OK。顔文字・過剰絵文字は禁止。質問は最後に短く1つだけ。"
        )
        user = (
            f"未完了タスクは {count} 件。例: {first_title[:40]} "
            "締切が近いものから一歩だけ動ける言い回しで。"
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        out = chat_api(msgs)
        # 念のため 40字超なら短縮
        txt = clean_out(out)
        if len(txt) > 44:
            txt = txt[:44].rstrip() + "…"
        return txt
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== 出力クレンジング ======
def clean_out(s: str) -> str:
    t = (s or "").strip()
    if len(t) >= 2 and (
        (t[0] == '"' and t[-1] == '"') or
        (t[0] == '“' and t[-1] == '”') or
        (t[0] == '「' and t[-1] == '」') or
        (t[0] == '『' and t[-1] == '』')
    ):
        t = t[1:-1]
    t = t.replace("SUGGEST:", "").replace("Suggest:", "")
    return t.strip()
