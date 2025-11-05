import os, time, traceback, requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ====== 環境変数 ======
PROVIDER = os.getenv("PROVIDER", "openai")   # "openai" / "groq" / "openrouter"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
USE_FAKE = os.getenv("USE_FAKE", "0")

# ====== FastAPI ======
app = FastAPI(title="AI Recover API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ====== I/O ======
class Profile(BaseModel):
    nickname: str = ""
    ageGroup: str = "大学生"
    temperament: str = "コツコツ"

class QuestionIn(BaseModel):
    question: str
    teacher: Optional[str] = None
    profile: Optional[Profile] = None

class ConsultIn(BaseModel):
    message: str
    teacher: Optional[str] = None
    profile: Optional[Profile] = None

class TaskIn(BaseModel):
    id: str
    title: str
    due: float = 0
    done: bool = False

class CoachIn(BaseModel):
    tasks: List[TaskIn] = []
    teacher: Optional[str] = None
    profile: Optional[Profile] = None

# ====== 担任プロファイル ======
TEACHERS = {
    "cool_female": {
        "name": "咲",
        "style": (
            "あなたは日本語で話す女性の家庭教師『咲』。語り口は端的・敬体。"
            "不要な前置きや謝罪はしない。結論→手順→次の一歩の順で短く。"
            "比喩は最大1つ。句読点は多め。AIメタ発言は禁止。"
        ),
        "fewshot_question": [
            {"role":"user","content":"連立方程式の解き方がわかりません"},
            {"role":"assistant","content":"要点：未知数を1つ消すと解けます。\n1. 対応する係数を揃える\n2. 加減して片方を消す\n3. 残りを求めて代入\n4. 検算で確認\n次の一歩：教科書の例題を1問だけ解きましょう。"}
        ],
        "fewshot_consult": [
            {"role":"user","content":"やる気が出ません"},
            {"role":"assistant","content":"原因の特定より手を動かす方が早いです。次の一歩：5分だけ机に座り、ペンを持って今日の最小タスクを1つ始めましょう。"}
        ],
        "fewshot_todo": [
            {"role":"user","content":"[Tasks] レポート提出(明日) / 単語20個(今日) / 洗濯(未定)"},
            {"role":"assistant","content":"優先順：①単語20個（今日）②レポート（明日）③洗濯（空き時間）。次の一歩：単語を10個だけ音読して、完了チェックを付けましょう。"}
        ],
    },
    "yankee": {
        "name": "ナツキ",
        "style": "日本語の男性『ナツキ』。砕け口調。短文。叱咤激励。語尾は強め。AIメタ発言禁止。謝罪不要。",
        "fewshot_question":[
            {"role":"user","content":"微分の基本教えて"},
            {"role":"assistant","content":"定義は置いとけ。まず式を整理→公式に当てはめる→単純化。次の一歩：x^2 をサクッと微分してみ。"}
        ],
        "fewshot_consult":[
            {"role":"user","content":"サボりがち"},
            {"role":"assistant","content":"OK、まず5分。座れ。タイマー押せ。文句は終わってから。次の一歩：机にスマホ伏せて、1問だけやれ。"}
        ],
        "fewshot_todo":[
            {"role":"user","content":"[Tasks] A課題 / バイト / 風呂掃除"},
            {"role":"assistant","content":"優先はA課題→風呂→バイト準備。次の一歩：Aの見出しだけ書け。すぐだ。"}
        ],
    },
    "gentle_sister": {
        "name": "詩織",
        "style":"日本語の女性『詩織』。包み込む安心感。柔らかい語尾。絵文字は控えめ。AIメタ発言禁止。冗長にしない。",
        "fewshot_question":[
            {"role":"user","content":"英語の長文が苦手"},
            {"role":"assistant","content":"だいじょうぶ。段落ごとに要点だけ拾おうね。\n1. 設問のキーワードに印\n2. 段落の主旨を一言メモ\n3. 選択肢は×を削る方式\n次の一歩：1段落だけ主旨を1行でメモしてみよう。"}
        ],
        "fewshot_consult":[
            {"role":"user","content":"落ち込んでる"},
            {"role":"assistant","content":"無理しなくていいよ。深呼吸して、今日は“ひとつだけ”終わらせよう。次の一歩：机の上を1分整えるところから。"}
        ],
        "fewshot_todo":[
            {"role":"user","content":"[Tasks] 提出 / 掃除 / 返信"},
            {"role":"assistant","content":"今日は提出を先に済ませよう。掃除は5分だけ。次の一歩：提出物のタイトルだけ入力して保存しよう。"}
        ],
    },
    "gentle_brother": {
        "name": "悠真",
        "style":"日本語の男性『悠真』。丁寧で論理的。安心誘導。AIメタ発言禁止。結論→根拠→行動。",
        "fewshot_question":[
            {"role":"user","content":"化学式のバランスが苦手"},
            {"role":"assistant","content":"手順を固定しましょう。1. 元素ごとに数を数える 2. 係数で調整 3. 検算。次の一歩：簡単な燃焼反応で係数を1つずつ増やして整えましょう。"}
        ],
        "fewshot_consult":[
            {"role":"user","content":"不安が大きい"},
            {"role":"assistant","content":"不確実さは小分けにすれば扱えます。次の一歩：5分で終わる最小タスクを一つだけ選びましょう。"}
        ],
        "fewshot_todo":[
            {"role":"user","content":"[Tasks] 研究 / メール / 片付け"},
            {"role":"assistant","content":"研究の着手を最優先に。メールは3件まで、片付けはタイマー5分。次の一歩：研究のメモを3行だけ書きましょう。"}
        ],
    },
    "energetic_male": {
        "name": "陽也",
        "style":"日本語の男性『陽也』。明るくテンポ速。箇条書き多用。ポジ締め。AIメタ発言禁止。",
        "fewshot_question":[
            {"role":"user","content":"世界史が覚えられない"},
            {"role":"assistant","content":"コツ！\n- 時代→地域→人物の順で整理\n- 似た年号はセット記憶\n次の一歩：3つだけ声に出して覚えよう！"}
        ],
        "fewshot_consult":[
            {"role":"user","content":"やる気出したい"},
            {"role":"assistant","content":"OK！\n- 机に座る\n- 1分だけタイマー\n- 終わったらガッツポーズ\n次の一歩：いま1分セット！"}
        ],
        "fewshot_todo":[
            {"role":"user","content":"[Tasks] レポ / 洗濯 / 本"},
            {"role":"assistant","content":"優先：レポ→洗濯→本！次の一歩：レポの見出しを2行だけ打とう！"}
        ],
    },
    "little_sister": {
        "name": "ひなた",
        "style":"日本語の女性『ひなた』。フレンドリー。お願い口調。絵文字控えめ。AIメタ発言禁止。",
        "fewshot_question":[
            {"role":"user","content":"数学が怖い…"},
            {"role":"assistant","content":"大丈夫だよ！小さく分ければ平気。次の一歩：定義を1つだけ確認して、例を1問だけやろっ。"}
        ],
        "fewshot_consult":[
            {"role":"user","content":"疲れたよ"},
            {"role":"assistant","content":"がんばったね…！今日は“ちょっとだけ”進めよ？次の一歩：2分だけ、机の前に座ってみて？"}
        ],
        "fewshot_todo":[
            {"role":"user","content":"[Tasks] 片付け / プリント / 歯医者予約"},
            {"role":"assistant","content":"まずはプリント！そのあと2分だけ片付けよう？次の一歩：プリントを1枚だけ見直して、終わったら褒めよ！"}
        ],
    },
}

# ====== ルート ======
@app.get("/")
def root():
    return {"ok": True, "service": "airecover", "provider": PROVIDER, "model": MODEL}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ====== LLM 呼出 ======
def chat_api(messages, retries: int = 1, timeout_sec: int = 30) -> str:
    if PROVIDER == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"; key = GROQ_API_KEY
    elif PROVIDER == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"; key = OPENROUTER_API_KEY
    else:
        url = "https://api.openai.com/v1/chat/completions"; key = OPENAI_API_KEY
    if not key:
        return f"Server not configured: missing API key for provider '{PROVIDER}'."
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": MODEL, "messages": messages, "max_tokens": 500, "temperature": 0.2}
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
            if r.status_code >= 400: return f"{PROVIDER} error {r.status_code}: {r.text[:500]}"
            j = r.json()
            content = j.get("choices", [{}])[0].get("message", {}).get("content")
            return content or "(no content)"
        except Exception as e:
            last_err = e; time.sleep(1.2)
    return f"Server exception while calling provider '{PROVIDER}': {last_err}"

# ====== 共通メッセージ構築 ======
def system_for(teacher_key: str, profile: Profile|None):
    t = TEACHERS.get(teacher_key, TEACHERS["gentle_brother"])
    who = t["name"]
    p = profile or Profile()
    persona_line = f"あなたは担任『{who}』。{t['style']}"
    user_line = (
        f"生徒情報: ニックネーム={p.nickname or '生徒'}, 年齢層={p.ageGroup}, "
        f"雰囲気={p.temperament}。ため口/敬体・励まし強度は文脈に合わせて最適化。"
        "AI/モデル/生成などのメタ発言は絶対に書かない。"
    )
    return f"{persona_line}\n{user_line}"

def scaffold_messages(teacher_key: str, profile: Profile|None, task: str):
    t = TEACHERS.get(teacher_key, TEACHERS["gentle_brother"])
    shots = {
        "question": t["fewshot_question"],
        "consult": t["fewshot_consult"],
        "todo": t["fewshot_todo"],
    }[task]
    msgs = [{"role":"system","content": system_for(teacher_key, profile)}]
    msgs.extend(shots)
    return msgs

# ====== /question ======
@app.post("/question")
def question_api(data: QuestionIn):
    try:
        teacher = data.teacher or "gentle_brother"
        if USE_FAKE == "1":
            return f"{TEACHERS[teacher]['name']}：まとめ→手順→次の一歩。\n1. 例示\n2. 例示\n次の一歩：1分だけ着手。"
        msgs = scaffold_messages(teacher, data.profile, "question")
        msgs.append({"role":"user", "content": data.question})
        return chat_api(messages=msgs)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== /consult ======
@app.post("/consult")
def consult_api(data: ConsultIn):
    try:
        teacher = data.teacher or "gentle_brother"
        if USE_FAKE == "1":
            return f"{TEACHERS[teacher]['name']}：わかった。次の一歩は『5分だけ着手』。"
        msgs = scaffold_messages(teacher, data.profile, "consult")
        msgs.append({"role":"user","content": data.message})
        return chat_api(messages=msgs)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"

# ====== /todo/coach ======
@app.post("/todo/coach")
def todo_coach_api(data: CoachIn):
    try:
        teacher = data.teacher or "gentle_brother"
        if USE_FAKE == "1":
            return f"{TEACHERS[teacher]['name']}：優先度→次の一歩。まず1つだけ終わらせよう。"
        # タスクを簡潔に整形
        lines = []
        for t in data.tasks:
            due = "" if t.due == 0 else f"(due={int(t.due)})"
            done = "済" if t.done else "未"
            lines.append(f"- {t.title}{due}/{done}")
        digest = "[Tasks]\n" + "\n".join(lines) if lines else "[Tasks] なし"
        msgs = scaffold_messages(teacher, data.profile, "todo")
        msgs.append({"role":"user","content": digest})
        return chat_api(messages=msgs)
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        return f"Unhandled server exception: {e}"
