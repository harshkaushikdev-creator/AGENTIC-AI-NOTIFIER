import os, re, json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import gradio as gr
import smtplib
from email.mime.text import MIMEText

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")
llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0.3)

# FIX 2: persistent SQLite jobstore + misfire grace time
scheduler = BackgroundScheduler(
    timezone=IST,
    jobstores={"default": SQLAlchemyJobStore(url="sqlite:///jobs.db")},
    job_defaults={"coalesce": True, "misfire_grace_time": 300}
)
scheduler.start()


class State(TypedDict):
    topics:    List[str]
    email:     Optional[str]
    headlines: List[str]
    sources:   List[dict]
    briefing:  str


def scrape_news(state: State) -> State:
    import httpx
    headlines = []
    sources   = []
    query     = " ".join(state["topics"])
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


def generate_briefing(state: State) -> State:
    if not state["headlines"]:
        state["briefing"] = "No news found for your topics. Try different keywords."
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


def send_email(to: str, subject: str, body: str):
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASS")
    if not gmail_user or not gmail_pass:
        print("Gmail credentials not set")
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = gmail_user
        msg["To"]      = to
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, to, msg.as_string())
        print(f"Email sent to {to}")
    except Exception as e:
        print(f"Email error: {e}")


def build_graph():
    g = StateGraph(State)
    g.add_node("scrape", scrape_news)
    g.add_node("brief",  generate_briefing)
    g.set_entry_point("scrape")
    g.add_edge("scrape", "brief")
    g.add_edge("brief",  END)
    return g.compile()

graph = build_graph()


def run_job(topics: List[str], email: Optional[str]):
    result = graph.invoke({"topics": topics, "email": email, "headlines": [], "sources": [], "briefing": ""})
    if email:
        send_email(email, f"Daily Briefing: {', '.join(topics)}", result["briefing"])


# FIX 1: stronger prompt so LLM correctly classifies "daily" vs "at_time"
def parse_message(msg: str, prev_email: Optional[str]) -> dict:
    resp = llm.invoke([
        SystemMessage(content=(
            "Extract from the user message and return ONLY valid JSON with keys:\n"
            '- "topics": list of news topic strings\n'
            '- "email": email address or null\n'
            '- "schedule": "now" | "in_minutes" | "at_time" | "daily"\n'
            '- "value": integer minutes if in_minutes, "HH:MM" 24hr string if at_time/daily, null if now\n\n'
            'USE "daily" if message has: "every day", "daily", "each day", "everyday", "every morning/evening/night"\n'
            'USE "at_time" ONLY for a single one-time delivery\n'
            'ALWAYS convert to 24hr: "9:20 PM"→"21:20", "8 AM"→"08:00", "9 PM"→"21:00"'
        )),
        HumanMessage(content=msg)
    ])
    try:
        raw    = re.sub(r"```json|```", "", resp.content).strip()
        parsed = json.loads(raw)
        parsed["email"] = parsed.get("email") or prev_email
        print(f"DEBUG parsed: {parsed}")
        return parsed
    except:
        return {"topics": ["technology"], "email": prev_email, "schedule": "now", "value": None}


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
        result   = graph.invoke({"topics": topics, "email": email, "headlines": [], "sources": [], "briefing": ""})
        briefing = result["briefing"]
        if email:
            send_email(email, f"Briefing: {', '.join(topics)}", briefing)
        return f"📰 **Briefing: {', '.join(topics)}**{email_note}\n\n---\n\n{briefing}"

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
        hour   = t.hour
        minute = t.minute
        # FIX 3: safe job_id when email is None
        job_id = f"daily_{email or 'noemail'}_{','.join(sorted(topics))}"
        scheduler.add_job(
            run_job, "cron",
            hour=hour, minute=minute,
            timezone=IST,
            args=[topics, email],
            id=job_id,
            replace_existing=True
        )
        return (f"✅ Daily briefing set! You'll get **{', '.join(topics)}** news "
                f"every day at **{t.strftime('%I:%M %p')} IST**.{email_note}\n\n"
                f"💡 To cancel, say: *'cancel my daily {', '.join(topics)} briefing'*")

    return "Try: *'Send me AI news now'* or *'cybersecurity news every day at 8 AM to me@gmail.com'*"


gr.ChatInterface(
    fn=chat,
    title="📰 Autonomous News Briefing Agent",
    description="Tell me what news you want, when, and optionally your email.",
    examples=[
        "Send me AI news right now to harshkaushikagent@gmail.com",
        "Cybersecurity news in 5 minutes to harshkaushikagent@gmail.com",
        "Send me AI news every day at 8 AM to harshkaushikagent@gmail.com",
        "India Tech and startup news daily at 9 PM to harshkaushikagent@gmail.com",
    ],
).launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=False,
    show_error=True,
)
