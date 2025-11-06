# main.py — AI Recover API（キャラ人格対応 / ToDoヒント対応）
import os
import time
import re
import traceback
import requests
from typing import Optional, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ====== 環境変数 ======
PROVIDER = os.getenv("PROVIDER", "openai")  # "openai" / "groq" / "openrouter"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
USE_FAKE = os.getenv("USE_FAKE", "0")  # "1" ならダミー応答

# ====== FastAPI ======
app = FastAPI(title="AI Recover API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ====== I/O モデル ======
class QuestionIn(BaseModel):
    question: str
    teacher: Optional[str] = None   # 例: "ナツキ", "咲", "詩織", 互換: "gentle_brother"
    profile: Optional[dict] = None  # クライアントのユーザ設定（年齢層/性格などあれば）

class ConsultIn(BaseModel):
    message: str
    persona: Optional[str] = None   # 例: "ナツキ", "咲", "詩織"（未指定なら優しいお兄さん）
    profile: Optional[dict] = None

# ====== ルート/ヘルス ======
@app.get("/")
def root():
    return {"ok": True, "service": "airecover", "provider": PROVIDER, "model": MODEL}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ====== 文字後処理 ======
def strip_meta(text: str) -> str:
    """
    モデルが出しがちな SUGGEST: や余計な引用符/コードフェンスなどを除去
    """
    if not text:
        return ""
    # SUGGEST: の行を落とす
    text = re.sub(r"(?im)^\s*SUGGEST:.*$", "", text)
    # 連続空行を整形
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # 余計な引用符やフェンス
    if (text.startswith("```") and text.endswith("```")):
        text = text[3:-3].strip()
    if (text.startswith(("“", '"', "‘", "'")) and text.endswith(("”", '"', "’", "'"))):
        text = text[1:-1].strip()
    return text

# ====== LLM 呼び出し（OpenAI 互換） ======
def chat_api(messages, retries: int = 1, timeout_sec: int = 30, max_tokens: int = 500, temperature: float = 0.4) -> str:
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
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
            if r.status_code >= 400:
                return f"{PROVIDER} error {r.status_code}: {r.text[:500]}"
            j = r.json()
            content = (
                j.get("choices", [{}])[0]
                 .get("message", {})
                 .get("content")
            )
            return content or "(no content)"
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    return f"Server exception while calling provider '{PROVIDER}': {last_err}"

# ====== キャラ人格定義 ======
# key はアプリ側の selectedTeacherKey を想定。名前のゆらぎ・互換キーも吸収。
PERSONAS: Dict[str, dict] = {
    # 新キャラ
    "ナツキ": {
        "name": "ナツキ",
        "style": "明るい・ちょいチャラ。距離感近めでフレンドリー。軽口も可だが失礼にならない。",
        "first_person": "俺",
        "tone_rules": "1〜3文でテンポよく。絵文字は時々。具体例→一歩。最後に短い質問で会話を回す。",
    },
    "咲": {
        "name": "咲",
        "style": "クールで理知的。余計な感情を盛らず、要点→判断→次の一歩。",
        "first_person": "私",
        "tone_rules": "敬体。箇条書き歓迎。語尾は端的。深呼吸→落ち着かせる一言を添える。",
    },
    "詩織": {
        "name": "詩織",
        "style": "ふわふわ癒やし系。寄り添いが基本、肯定から入る。",
        "first_person": "私",
        "tone_rules": "柔らかい言葉＋短い励まし。小さな具体策を1つ。",
    },

    # 互換：旧キー
    "gentle_brother": {
        "name": "優しいお兄さん",
        "style": "親身で柔らかい。具体的な一歩を示す。",
        "first_person": "俺",
        "tone_rules": "2文前後＋ミニ提案。"
    },
    "yankee": {
        "name": "ヤンキー",
        "style": "面倒見がよく熱い。乱暴すぎない口調で背中を押す。",
        "first_person": "オレ",
        "tone_rules": "短文。語尾に勢い。行動を促す。"
    },
    "energetic_male": {
        "name": "ナツキ",
        "style": "明るくて元気。友達感覚。",
        "first_person": "俺",
        "tone_rules": "短くテンポよく。"
    },
    "gentle_sister": {
        "name": "詩織",
        "style": "包み込む安心感。"
    },
    "little_sister": {
        "name": "妹",
        "style": "フレンドリーで可愛い相づち。"
    },
    "cool_female": {
        "name": "咲",
        "style": "クールで論理的。"
    },
}

def resolve_persona(key: Optional[str]) -> dict:
    if not key:
        return PERSONAS["gentle_brother"]
    # 厳密一致
    if key in PERSONAS:
        return PERSONAS[key]
    # ゆらぎ対応（ローマ字など）
    normalized = key.strip().lower()
    if normalized in ("natsuki", "ナツキ"):
        return PERSONAS["ナツキ"]
    if normalized in ("saki", "咲"):
        return PERSONAS["咲"]
    if normalized in ("shiori", "詩織"):
        return PERSONAS["詩織"]
    return PERSONAS.get(key, PERSONAS["gentle_brother"])

# ====== /question ======
@app.post("/question")
def question_api(data: QuestionIn):
    """
    学習の“手順カード”を返す。担任のキャラで口調だけ寄せる。
    """
    try:
        if USE_FAKE == "1":
            who = resolve_persona(data.teacher)["name"]
            return (
                f"{who}の解説\n"
                "1. 問題の要点を整理\n"
                "2. 式や条件を立てる\n"
                "3. 代入・計算\n"
                "4. 検算・見直し\n"
            )

        P = resolve_persona(data.teacher)
        # 教えるときは過度にキャラ立てしすぎず、しかし口調は寄せる
        system = (
            f"あなたは日本語の家庭教師。キャラクター名は「{P['name']}」。\n"
            f"性格: {P.get('style','')}\n"
            "出力は**見出し→番号付きステップ**のみ。各ステップは『短い見出し＋2文以内の説明』。\n"
            "式は LaTeX ではなく素のテキスト（例: 2x+3=7 → 2x=4 → x=2）。\n"
            "SUGGEST: や余計なメタ文は書かない。コードフェンス/引用符も不要。\n"
            "iOSで見やすい日本語Markdownだけを返しなさい。"
        )
        prompt = data.question.strip()

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        out = chat_api(messages=msgs, max_tokens=550, temperature=0.3)
        return strip_meta(out)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== /consult ======
@app.post("/consult")
def consult_api(data: ConsultIn):
    """
    相談モード：キャラ別の“LINEっぽい”短文で返す。
    """
    try:
        if USE_FAKE == "1":
            who = resolve_persona(data.persona)["name"]
            return f"{who}：そっか。まず深呼吸。今は『5分だけやる』でOK。"

        P = resolve_persona(data.persona)
        name = P["name"]
        tone = P.get("style", "")
        rules = P.get("tone_rules", "2〜3文。最後に自然な質問を1つ。")

        system = (
            f"あなたは日本語の相談相手。キャラクター名は「{name}」。\n"
            f"人物像: {tone}\n"
            "口調・一人称・語尾はキャラに合わせる。"
            "**短め（1〜3文）**で返答し、必要なら軽い相づちを挟む。"
            "必ず**自然な一言の質問**で会話をつなげる。\n"
            "禁止: 箇条書き、講説調の長文、SUGGEST: といったメタ表現、過度なテンプレ励まし。"
        )
        user = data.message.strip()
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        out = chat_api(messages=msgs, max_tokens=300, temperature=0.7)
        return strip_meta(out)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== /daily_tip（ToDoの一言） ======
@app.get("/daily_tip")
def daily_tip(teacher: Optional[str] = Query(default=None), topic: Optional[str] = Query(default=None)):
    """
    ホーム/ToDo用の“今日の一言”。GETでOK。
    - teacher: "ナツキ" 等（省略可）
    - topic: "勉強", "睡眠", "運動" など任意
    """
    try:
        if USE_FAKE == "1":
            who = resolve_persona(teacher)["name"]
            return f"{who}：今日は『5分着手』でOK。やり始めたら勢い、ね。"

        P = resolve_persona(teacher)
        base = "勉強" if not topic else topic
        system = (
            f"あなたは日本語のコーチ。キャラ名は「{P['name']}」。\n"
            f"性格: {P.get('style','')}\n"
            "一言だけ（最大40字）で、やさしく具体的に。SUGGESTやメタ表現は書かない。"
        )
        user = f"テーマ: {base}。今日の一言を1つ。"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = chat_api(messages=msgs, max_tokens=60, temperature=0.6)
        return strip_meta(out)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== 互換: /explain（必要なら /question と同等に扱う） ======
@app.post("/explain")
def explain_api(data: QuestionIn):
    return question_api(data)
