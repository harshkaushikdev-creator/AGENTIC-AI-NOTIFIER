import os, re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import gradio as gr
import resend

load_dotenv()

IST       = ZoneInfo("Asia/Kolkata")
llm       = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0.3)
scheduler = BackgroundScheduler(timezone=IST)
scheduler.start()

TOO_FEW  = 3   # retry if fewer than this
TOO_MANY = 15  # filter if more than this


class State(TypedDict):
    topics:       List[str]
    email:        Optional[str]
    headlines:    List[str]
    sources:      List[dict]
    briefing:     str
    retry_count:  int
    quality_note: str   # shown to user explaining what the agent did


# ── Node 1: Scrape ─────────────────────────────────────────────────────────────
def scrape_news(state: State) -> State:
    import httpx
    headlines = []
    sources   = []
    query     = " ".join(state["topics"])

    # On retry, broaden the query by taking only the first topic keyword
    if state.get("retry_count", 0) > 0:
        query = state["topics"][0].split()[0]  # e.g. "Artificial Intelligence" → "AI"
        print(f"Retrying with broader query: {query}")

    try:
        r = httpx.get("https://gnews.io/api/v4/search", params={
            "q": query, "token": os.environ.get("GNEWS_API_KEY"),
            "lang": "en", "max": 20, "sortby": "publishedAt",
        }, timeout=10)
        for a in r.json().get("articles", []):
            title = a.get("title", "").strip()
            link  = a.get("url", "").strip()
            if title:
                headlines.append(title)
                sources.append({"title": title, "url": link})
    except Exception as e:
        print(f"GNews error: {e}")

    state["headlines"] = headlines
    state["sources"]   = sources
    return state


# ── Node 2: Evaluate quality ───────────────────────────────────────────────────
def evaluate(state: State) -> State:
    count = len(state["headlines"])
    if count < TOO_FEW:
        state["quality_note"] = f"⚠️ Only {count} articles found — retrying with broader query..."
    elif count > TOO_MANY:
        state["quality_note"] = f"📊 {count} articles found — filtering most relevant ones..."
    else:
        state["quality_note"] = f"✅ {count} articles found — generating briefing..."
    print(state["quality_note"])
    return state


# ── Node 3: Filter (when too many articles) ────────────────────────────────────
def filter_articles(state: State) -> State:
    headlines = state["headlines"]
    sources   = state["sources"]
    topics    = state["topics"]

    resp = llm.invoke([
        SystemMessage(content=(
            f"You are a news relevance filter. Topics of interest: {', '.join(topics)}.\n"
            f"From the list below, return ONLY the indices (0-based, comma separated) of the "
            f"top {TOO_FEW} to {TOO_MANY} most relevant and diverse articles. "
            "Return ONLY numbers like: 0,2,5,7,11 — nothing else."
        )),
        HumanMessage(content="\n".join(f"{i}. {h}" for i, h in enumerate(headlines)))
    ])

    try:
        indices = [int(x.strip()) for x in resp.content.strip().split(",") if x.strip().isdigit()]
        indices = [i for i in indices if i < len(headlines)]
        state["headlines"] = [headlines[i] for i in indices]
        state["sources"]   = [sources[i] for i in indices]
        print(f"Filtered to {len(state['headlines'])} articles")
    except Exception as e:
        print(f"Filter error: {e}")

    return state


# ── Node 4: Generate briefing ──────────────────────────────────────────────────
def generate_briefing(state: State) -> State:
    if not state["headlines"]:
        state["briefing"] = "No news found even after retrying. Please try different keywords."
        return state

    bullets = "\n".join(f"- {h}" for h in state["headlines"])
    resp = llm.invoke([
        SystemMessage(content=(
            f"You are a news briefing assistant. Today is {datetime.now(IST).strftime('%B %d, %Y')}.\n"
            "Write a clean briefing from these headlines. Group by sub-topic, add 1 line of context "
            "per item, end with a 2-line Key Takeaway. Keep it under 400 words."
        )),
        HumanMessage(content=f"Topics: {', '.join(state['topics'])}\n\nHeadlines:\n{bullets}")
    ])

    sources_text = "\n\n---\n📚 **Sources:**\n" + "\n".join(
        f"- {s['title']} → {s['url']}" for s in state.get("sources", []) if s.get("url")
    )
    state["briefing"] = resp.content + sources_text
    return state


# ── Routing logic ──────────────────────────────────────────────────────────────
def route_after_evaluate(state: State) -> str:
    count = len(state["headlines"])
    if count < TOO_FEW and state.get("retry_count", 0) < 2:
        return "retry"
    elif count > TOO_MANY:
        return "filter"
    else:
        return "brief"

def increment_retry(state: State) -> State:
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


# ── Build LangGraph ────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(State)

    g.add_node("scrape",   scrape_news)
    g.add_node("evaluate", evaluate)
    g.add_node("retry",    increment_retry)
    g.add_node("filter",   filter_articles)
    g.add_node("brief",    generate_briefing)

    g.set_entry_point("scrape")
    g.add_edge("scrape",   "evaluate")
    g.add_edge("retry",    "scrape")      # retry loops back to scrape
    g.add_edge("filter",   "brief")
    g.add_edge("brief",    END)

    g.add_conditional_edges("evaluate", route_after_evaluate, {
        "retry":  "retry",
        "filter": "filter",
        "brief":  "brief",
    })

    return g.compile()

graph = build_graph()


# ── Email ──────────────────────────────────────────────────────────────────────
def send_email(to: str, subject: str, body: str):
    resend.api_key = os.environ.get("RESEND_API_KEY")
    if not resend.api_key:
        print("Resend API key not set")
        return
    try:
        r = resend.Emails.send({
            "from":    "News Agent <onboarding@resend.dev>",
            "to":      to,
            "subject": subject,
            "text":    body,
        })
        print(f"Email sent: {r}")
    except Exception as e:
        print(f"Email error: {e}")


# ── Scheduled job ──────────────────────────────────────────────────────────────
def run_job(topics: List[str], email: Optional[str]):
    result = graph.invoke({
        "topics": topics, "email": email,
        "headlines": [], "sources": [], "briefing": "",
        "retry_count": 0, "quality_note": ""
    })
    if email:
        send_email(email, f"Daily Briefing: {', '.join(topics)}", result["briefing"])


# ── Parse intent ──────────────────────────────────────────────────────────────
def parse_message(msg: str, prev_email: Optional[str]) -> dict:
    import json
    resp = llm.invoke([
        SystemMessage(content=(
            "Extract from the user message and return ONLY valid JSON with keys:\n"
            '- "topics": list of news topics (strings)\n'
            '- "email": email address or null\n'
            '- "schedule": "now" | "in_minutes" | "at_time" | "daily"\n'
            '- "value": minutes as int if in_minutes, "HH:MM" string if at_time or daily, null if now\n'
            'Use "daily" if user says "every day", "daily", "each day", "every morning/evening" etc.'
        )),
        HumanMessage(content=msg)
    ])
    try:
        raw    = re.sub(r"```json|```", "", resp.content).strip()
        parsed = json.loads(raw)
        parsed["email"] = parsed.get("email") or prev_email
        return parsed
    except:
        return {"topics": ["technology"], "email": prev_email, "schedule": "now", "value": None}


# ── Gradio chat ────────────────────────────────────────────────────────────────
def chat(message: str, history: list) -> str:
    prev_email = None
    for h in history:
        raw = h.get("content", "") if isinstance(h, dict) else h
        if isinstance(raw, list):
            content = " ".join(i.get("text", "") if isinstance(i, dict) else str(i) for i in raw)
        else:
            content = str(raw)
        m = re.search(r"[\w.+\-]+@[\w.\-]+\.\w{2,}", content)
        if m:
            prev_email = m.group(0)
            break

    p        = parse_message(message, prev_email)
    topics   = p.get("topics") or ["technology"]
    email    = p.get("email")
    schedule = p.get("schedule", "now")
    value    = p.get("value")

    email_note = f"\n📬 Will be sent to **{email}**." if email else \
                 "\n💡 Add your email and I'll send it there too."

    if schedule == "now":
        result   = graph.invoke({
            "topics": topics, "email": email,
            "headlines": [], "sources": [], "briefing": "",
            "retry_count": 0, "quality_note": ""
        })
        briefing     = result["briefing"]
        quality_note = result.get("quality_note", "")
        if email:
            send_email(email, f"Briefing: {', '.join(topics)}", briefing)
        return (f"📰 **Briefing: {', '.join(topics)}**{email_note}\n"
                f"_{quality_note}_\n\n---\n\n{briefing}")

    elif schedule == "in_minutes":
        mins     = int(value) if value else 5
        run_time = datetime.now(IST) + timedelta(minutes=mins)
        scheduler.add_job(run_job, "date", run_date=run_time, args=[topics, email])
        return (f"✅ Scheduled! **{', '.join(topics)}** briefing in **{mins} min** "
                f"({run_time.strftime('%I:%M %p')} IST).{email_note}")

    elif schedule == "at_time":
        from dateutil import parser as dp
        t        = dp.parse(str(value)) if value else datetime.now(IST).replace(hour=9, minute=0)
        now      = datetime.now(IST)
        run_time = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        if run_time <= now:
            run_time += timedelta(days=1)
        scheduler.add_job(run_job, "date", run_date=run_time, args=[topics, email])
        return (f"✅ Scheduled! **{', '.join(topics)}** briefing at "
                f"**{run_time.strftime('%I:%M %p')} IST**.{email_note}")

    elif schedule == "daily":
        from dateutil import parser as dp
        t      = dp.parse(str(value)) if value else datetime.now(IST).replace(hour=9, minute=0)
        job_id = f"daily_{email}_{','.join(topics)}"
        scheduler.add_job(
            run_job, "cron",
            hour=t.hour, minute=t.minute,
            timezone=IST,
            args=[topics, email],
            id=job_id,
            replace_existing=True
        )
        return (f"✅ Daily briefing set! You'll get **{', '.join(topics)}** news "
                f"every day at **{t.strftime('%I:%M %p')} IST**.{email_note}\n\n"
                f"💡 To cancel say: *'cancel my daily {', '.join(topics)} briefing'*")

    return "Try: *'Send me AI news now'* or *'cybersecurity news every day at 8 AM to me@gmail.com'*"


gr.ChatInterface(
    fn=chat,
    type="messages",
    title="📰 Autonomous News Briefing Agent",
    description="Tell me what news you want, when, and optionally your email.",
    examples=[
        "Send me AI news right now to harshkaushikdev@gmail.com",
        "Cybersecurity news in 5 minutes to harshkaushikdev@gmail.com",
        "Send me AI news every day at 8 AM to harshkaushikdev@gmail.com",
        "India Tech and startup news daily at 9 PM to harshkaushikdev@gmail.com",
    ],
).launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    show_error=True,
)
