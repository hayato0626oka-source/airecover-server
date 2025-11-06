import os
import re
import time
import traceback
from typing import List, Optional

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ====== 環境変数 ======
PROVIDER = os.getenv("PROVIDER", "openai")         # "openai" / "groq" / "openrouter"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4o-mini")          # 任意のデフォルト
USE_FAKE = os.getenv("USE_FAKE", "0")              # "1" でダミー応答

# ====== FastAPI ======
app = FastAPI(title="homeroom-server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ====== Pydantic Models ======
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
    done: bool = False
    due: Optional[float] = None

class TodoCoachIn(BaseModel):
    tasks: List[TodoTask]
    teacher_key: Optional[str] = None
    student_nick: Optional[str] = None


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
        "style": "柔らかく共感的。語尾はやさしめ。落ち着くテンポで短く要点。",
        "greeting": "まずは深呼吸しよ。"
    },
    # 画像：悠真（理知的な優男）
    "yuuma": {
        "name": "悠真",
        "style": "丁寧で知的。穏やかな肯定から入り、具体の一歩を促す。冗長禁止。",
        "greeting": "状況を一緒に整理しよう。"
    },
}
PERSONA_KEYS = set(PERSONA.keys())

# アプリ側キー → サーバ側キーのエイリアス
KEY_ALIAS = {
    "cool_female": "saki",
    "yankee": "natsuki",
    "gentle_sister": "shiori",
    "gentle_brother": "yuuma",
    "energetic_male": "yuuma",
    "little_sister": "shiori",
}

PERSONA_TAG_RE = re.compile(r"^\s*\[persona:([a-zA-Z0-9_]+)\]\s*")

def resolve_teacher_key(raw_key: Optional[str], message: Optional[str]) -> str:
    """
    1) teacher_key を正規化（別名→公式キー）
    2) メッセージ先頭の [persona:xxx] タグがあれば優先
    3) 不正キーは 'saki' にフォールバック
    """
    k = (raw_key or "").strip().lower()
    k = KEY_ALIAS.get(k, k)

    if message:
        m = PERSONA_TAG_RE.match(message)
        if m:
            tag = m.group(1).lower()
            tag = KEY_ALIAS.get(tag, tag)
            if tag in PERSONA_KEYS:
                return tag

    return k if k in PERSONA_KEYS else "saki"

def strip_persona_tag(s: str) -> str:
    return PERSONA_TAG_RE.sub("", s or "", count=1)


# ====== LLM 呼び出し ======
def chat_api(msgs: list) -> str:
    """
    各プロバイダへ最小構成で問い合わせ。
    失敗したら例外を投げる（上位で握る）。
    """
    if PROVIDER == "openai" and OPENAI_API_KEY:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {
            "model": MODEL,
            "messages": msgs,
            "temperature": 0.7,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

    if PROVIDER == "groq" and GROQ_API_KEY:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {"model": MODEL, "messages": msgs, "temperature": 0.7}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

    if PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://render.com",
            "X-Title": "homeroom-server",
        }
        payload = {"model": MODEL, "messages": msgs, "temperature": 0.7}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

    # プロバイダ未設定時は簡易応答でフォールバック
    return "（サーバ設定が未完了のため、簡易応答です）"


# ====== 共通：出力クレンジング ======
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


# ====== プロンプト組立 ======
def persona_prompt(final_key: str, student_nick: Optional[str]) -> str:
    p = PERSONA.get(final_key, PERSONA["saki"])
    nick = student_nick or "あなた"
    return (
        f"あなたは日本語で会話するカウンセラー。キャラ設定：{p['name']}。"
        f"話し方：{p['style']} "
        f"出力は短めの自然な会話文を2〜3文。過度な長文禁止。"
        f"必ず最後に1つだけ優しい質問で返す。"
        f"禁止：『SUGGEST:〜』等のタグ、テンプレ励ましの連発、箇条書き。"
        f"呼びかけは「{nick}」。"
        f"最初の一言は「{p['greeting']}」で始める。"
    )

def tutor_prompt(final_key: str, student_nick: Optional[str]) -> str:
    # person化チューター（必要に応じて性格差分）
    base = (
        "You are a supportive Japanese tutor. "
        "Answer in clean Japanese Markdown for iOS display. "
        "Start with a one-line summary. Then provide a numbered procedure (3–6 steps). "
        "No LaTeX or code fences. Equations plain like 2x+3=7 → 2x=4 → x=2."
    )
    # 最低限、人格選択の影響を与える（口調微差）
    if final_key == "saki":
        return base + " Tone: concise, logical, no emojis."
    if final_key == "natsuki":
        return base + " Tone: casual and friendly."
    if final_key == "shiori":
        return base + " Tone: warm and gentle."
    if final_key == "yuuma":
        return base + " Tone: polite and calm."
    return base


# ====== ルート/ヘルス ======
@app.get("/")
def root():
    return {"ok": True, "provider": PROVIDER, "model": MODEL, "use_fake": USE_FAKE}


# ====== エンドポイント ======
@app.post("/consult")
def consult_api(data: ConsultIn):
    try:
        # 最終キーを確定し、タグは本文から除去
        final_key = resolve_teacher_key(data.teacher_key, data.message)
        clean_msg = strip_persona_tag(data.message)

        # ダミー応答モード
        if USE_FAKE == "1":
            who = PERSONA[final_key]["name"]
            return {"reply": f"{who}：それ、まずは一息つこ。次にどうしたい？"}

        system = persona_prompt(final_key, data.student_nick)
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": clean_msg},
        ]
        out = chat_api(msgs)
        return {"reply": clean_out(out)}
    except Exception:
        print("TRACEBACK:\n", traceback.format_exc())
        return {"reply": "Unhandled server exception."}


@app.post("/question")
def question_api(data: QuestionIn):
    try:
        final_key = resolve_teacher_key(data.teacher_key, data.question)

        if USE_FAKE == "1":
            who = PERSONA[final_key]["name"]
            return {"reply": f"{who}：要点だけで解説するね。まず与件の式を書き出そう。"}

        system = tutor_prompt(final_key, data.student_nick)
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": data.question},
        ]
        out = chat_api(msgs)
        return {"reply": clean_out(out)}
    except Exception:
        print("TRACEBACK:\n", traceback.format_exc())
        return {"reply": "Unhandled server exception."}


@app.post("/todo/coach")
def todo_coach_api(data: TodoCoachIn):
    """
    タスク配列から“今日の一言”を短文で返す。
    """
    try:
        final_key = resolve_teacher_key(data.teacher_key, None)

        if USE_FAKE == "1":
            who = PERSONA[final_key]["name"]
            return {"reply": f"{who}：まずは1分だけ手を付けよ。最初の一歩が一番軽い。"}

        pending = [t for t in data.tasks if not t.done]
        count = len(pending)
        first_title = pending[0].title if pending else ""
        system = (
            persona_prompt(final_key, data.student_nick) +
            " 出力は**短い一言**モード。10〜40字で、今すぐ着手できる行動を一つだけ提案。"
        )
        user = f"未完了 {count} 件。最優先: {first_title}" if count else "未完了は0件。"
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        out = chat_api(msgs)
        return {"reply": clean_out(out)}
    except Exception:
        print("TRACEBACK:\n", traceback.format_exc())
        return {"reply": "Unhandled server exception."}
