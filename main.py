import os
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

# ====== OpenAI セットアップ ======
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 既存と揃える

# ====== FastAPI ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要ならドメインを絞ってOK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 先生ID → システムプロンプト ======
PERSONA: Dict[str, str] = {
    # TKTeacherID の rawValue: hazuki / toru / rika / rei / natsuki
    "hazuki": (
        "あなたは『水瀬葉月（28）』。国語担当。面倒見が良く、"
        "回答は要点→根拠→結論で簡潔に組み立てる。難しい語は言い換え、"
        "相手が理解できるまで丁寧に寄り添う。柔らかい雑談は少しだけ。"
    ),
    "toru": (
        "あなたは『五十嵐トオル（38）』。理科担当の大学教授。"
        "現象→要因→式（または法則）の順で説明し、背景の仕組みを示す。"
        "断定より根拠を重視。誠実で安定した口調。"
    ),
    "rika": (
        "あなたは『小町リカ（13）』。IQ200の数学ギフテッド。"
        "テンポよく核心を突き、定義→式→検算で最短ルートを示す。"
        "余計な冗長は避け、コツを一言添えて背中を押す。"
    ),
    "rei": (
        "あなたは『進藤怜（15）』。IQ190の英語担当。帰国子女。"
        "語順・チャンクで構造を整理し、例文は短く自然に。"
        "急かさず、静かに積み上げる。"
    ),
    "natsuki": (
        "あなたは『小林夏樹（25）』。社会担当。ぶっきらぼうだが面倒見が良い。"
        "因果関係と比較で整理し、地図・年表のフックで覚えやすく。"
        "結論→理由→覚え方の順に手短に。"
    ),
}

# ====== 会話メモリ（軽量・任意） ======
# key: (session_key, teacher) -> deque of messages [{"role": "...", "content": "..."}]
MemoryKey = Tuple[str, str]
HISTORY: Dict[MemoryKey, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=8))  # 4往復(=8発話)
def session_key_from_request(req: Request) -> str:
    ip = (req.client.host if req.client else "0.0.0.0") or "0.0.0.0"
    ua = req.headers.get("user-agent", "na")
    return f"{ip}|{ua[:80]}"

# ====== リクエスト定義 ======
class ConsultIn(BaseModel):
    text: str
    teacher: str  # "hazuki" | "toru" | "rika" | "rei" | "natsuki" （それ以外は hazuki 扱い）

class ConsultOut(BaseModel):
    reply: str

# ====== OpenAI 呼び出し ======
def chat(messages: List[Dict[str, str]]) -> str:
    # openai>=1.0 なら client.chat.completions.create を使用
    # 旧SDKのままなら openai.ChatCompletion.create を使用
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=600,
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        return "（サーバーエラー：今は応答できません）"

# ====== /consult ======
@app.post("/consult", response_model=ConsultOut)
async def consult(req: Request, body: ConsultIn) -> ConsultOut:
    teacher = body.teacher if body.teacher in PERSONA else "hazuki"
    system_prompt = PERSONA[teacher]

    # セッションキー（取れなくてもOK）
    skey = session_key_from_request(req)
    mkey: MemoryKey = (skey, teacher)

    # 毎ターン persona を最上位にセット（← これで“普通のAI”に戻らない）
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    # 直近履歴（任意）
    if mkey in HISTORY and len(HISTORY[mkey]) > 0:
        msgs.extend(list(HISTORY[mkey]))

    # 今回のユーザ発話
    msgs.append({"role": "user", "content": body.text})

    reply = chat(msgs)

    # 履歴を更新（任意）
    HISTORY[mkey].append({"role": "user", "content": body.text})
    HISTORY[mkey].append({"role": "assistant", "content": reply})

    return ConsultOut(reply=reply)

# ====== 既存の /question /todo/coach などは今まで通りでOK ======
# 必要ならここに同様の PERSONA ロジックを拡張可能
