import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from collections import deque
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
MAX_SUBLINKS_PER_PAGE = 3

# -------------------------------------------------------
# 🔧 LOAD ENV
# -------------------------------------------------------
load_dotenv()

# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

TAVILY_API_KEY = st.secrets['TAVILY_API_KEY']
FIRECRAWL_API_KEY = st.secrets['FIRECRAWL_API_KEY']
# st.write(TAVILY_API_KEY)
# st.write(FIRECRAWL_API_KEY)

# -------------------------------------------------------
# IMPORT TOOLS
# -------------------------------------------------------
from tavily import TavilyClient
from firecrawl import Firecrawl
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph

import os
# -------------------------------------------------------
# AZURE GPT-4o LLM
# -------------------------------------------------------


llm = AzureChatOpenAI(
                                api_key = st.secrets['API_KEY'],
                                azure_endpoint = st.secrets['AZURE_ENDPOINT'],
                                model = "gpt-4o",
                                api_version=st.secrets['API_VERSION'],
                                temperature = 0.
                                )

# ============================================================
# CHUNK 1 — Imports, Keys, Helper Utilities, Search + Crawl
# ============================================================

import json
import re
from collections import deque
from typing import TypedDict

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tavily import TavilyClient
from firecrawl import Firecrawl
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI


# ============================================================
# 🔑 API KEYS (your existing variable names)
# ============================================================
# These MUST exist in your environment already.
# Example:
# TAVILY_API_KEY = "tvly-xxxxx"
# FIRECRAWL_API_KEY = "fc-xxxxxx"

tavily = TavilyClient(api_key=TAVILY_API_KEY)
firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)

# =====================================================================
# DISC BADGE RENDERER
# =====================================================================
import json
import re
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# DISC BADGE (Humantic Style)
# =========================================================
def linkedin_priority(url: str, intent: str):
    u = url.lower()

    if intent == "person_profile":
        if "/in/" in u:
            return 100
        if "/company/" in u:
            return 5
        if "/posts/" in u or "/feed/" in u or "/pulse/" in u:
            return 2
        return 1

    if intent == "company_profile":
        if "/company/" in u:
            return 100
        if "/in/" in u:
            return 5
        if "/posts/" in u or "/feed/" in u or "/pulse/" in u:
            return 2
        return 1

    return 1



def classify_linkedin_url(url: str):
    """
    Accurately classify LinkedIn URLs as either:
    - person_profile
    - company_profile
    - unknown

    Uses:
    1. URL structural rules (most accurate)
    2. Final fallback → LLM intent reasoning
    """

    u = url.lower()

    # ---------RULE 1: Pure structural classification----------
    # Person profiles ALWAYS have '/in/' pattern globally
    if re.search(r"linkedin\.com\/in\/[^\/]+\/?$", u):
        return "person_profile"

    # Company profiles ALWAYS have /company/<name>
    if "linkedin.com/company/" in u:
        return "company_profile"

    # ---------RULE 2: Check if LinkedIn FOLLOWER/EMPLOYEE metadata exists---------
    # Tavily sometimes returns metadata indicating company
    meta = url  # not ideal until passed raw metadata
    # You will apply this rule in planner node instead.

    # ---------RULE 3: Fallback → let LLM decide---------
    intent_prompt = """
    Decide if this URL belongs to:
    - a person's LinkedIn profile
    - a company's LinkedIn profile
    - or unknown.

    Return strict JSON:
    {"intent": "person_profile" | "company_profile" | "unknown"}
    """

    resp = llm.invoke([
        {"role": "system", "content": intent_prompt},
        {"role": "user", "content": url}
    ])

    try:
        return json.loads(resp.content.strip())["intent"]
    except:
        return "unknown"



def detect_intent_llm(question, extracted_pages, search_urls=None):
    """
    MUCH more accurate intent detection:
    - Look at question
    - Look at URLs
    - Look at raw_content signals
    """

    # Build context for LLM
    context = {
        "question": question,
        "urls": [],
        "samples": []
    }

    # Add URLs (from search step)
    if search_urls:
        for u in search_urls:
            context["urls"].append(u["url"])
            raw = u.get("raw", {})
            snippet = raw.get("raw_content") or raw.get("content") or ""
            context["samples"].append({
                "url": u["url"],
                "snippet": snippet[:500]
            })

    # Add extracted_pages (from crawl step)
    if extracted_pages:
        for p in extracted_pages:
            context["urls"].append(p["url"])
            md = p.get("content") or ""
            context["samples"].append({
                "url": p["url"],
                "snippet": md[:500]
            })

    system_prompt = """
You are an intent classifier for LinkedIn research.

You MUST choose exactly one of:
- "person_profile"
- "company_profile"
- "general_research"

Decision rules:
1. If any LinkedIn URL contains "/in/" → strongly person_profile.
2. If any LinkedIn URL contains "/company/" → strongly company_profile.
3. If raw_content mentions job titles, years, experience, followers → person_profile.
4. If raw_content mentions "employees", "industry", "headquarters", etc. → company_profile.
5. If the question is explicitly asking for a LinkedIn profile of a person → person_profile.
6. If the question is asking for a company profile → company_profile.
7. Otherwise → general_research.

Return STRICT JSON:
{"intent": "..."}    
"""

    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(context)}
    ])

    try:
        data = json.loads(resp.content.strip())
        st.write("content of response is..", resp.content.strip())
        return data.get("intent", "general_research")
    
    except:
        return "general_research"



def render_disc_badge(disc_scores):
    if not disc_scores or not isinstance(disc_scores, dict):
        return ""

    dominant = max(disc_scores, key=disc_scores.get)
    colors = {"D": "#E74C3C", "I": "#F1C40F", "S": "#2ECC71", "C": "#3498DB"}
    color = colors.get(dominant, "#555")

    return f"""
    <div style="
        display:inline-block;
        padding:8px 20px;
        border-radius:30px;
        background:{color};
        color:white;
        font-weight:700;
        font-size:16px;">
        DISC Type: {dominant}
    </div>
    """

# =========================================================
# BIG FIVE RADAR
# =========================================================
def plot_big_five_radar(big_five):
    if not big_five:
        return None

    traits = list(big_five.keys())
    values = list(big_five.values())

    fig = go.Figure(data=go.Scatterpolar(
        theta=traits + [traits[0]],
        r=values + [values[0]],
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False
    )
    return fig

# =========================================================
# DISC WHEEL
# =========================================================
def plot_disc_wheel(disc):
    if not disc:
        return None

    fig = go.Figure(data=[go.Pie(
        labels=list(disc.keys()),
        values=list(disc.values()),
        hole=0.55
    )])
    return fig

# =========================================================
# IMAGE EXTRACTOR
# =========================================================
def extract_profile_image(raw, md):
    for key in ["primary_image", "image_url", "thumbnail", "favicon", "image"]:
        if key in raw and isinstance(raw[key], str) and raw[key].startswith("http"):
            return raw[key]

    m = re.search(r'property="og:image"\s*content="([^"]+)"', md or "")
    if m:
        return m.group(1)

    cdn = re.findall(r'https://media\.licdn\.com[^\s"\']+', md or "")
    if cdn:
        return cdn[0]

    g = re.findall(r'https://encrypted-tbn0\.gstatic\.com[^"\']+', md or "")
    if g:
        return g[0]

    return ""


# ============================================================
# Safe JSON Helper
# ============================================================
def safe_json(llm, messages, max_retries=4):
    """
    Forces LLM to output VALID JSON.
    Retries until success.
    """
    for _ in range(max_retries):
        resp = llm.invoke(messages)
        raw = resp.content.strip()

        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(raw)
        except:
            # Retry prompt
            messages.append({
                "role": "system",
                "content": "Return ONLY valid JSON. No explanations."
            })

    return {}

# ============================================================
# LinkedIn URL Detection
# ============================================================
def is_linkedin(url: str):
    return "linkedin.com" in url.lower()

# ============================================================
# Tavily Search Wrapper
# ============================================================
def search_tool(query):
    return tavily.search(
        query=query,
        max_results=4,
        include_raw_content=True
    ).get("results", [])

# ============================================================
# Firecrawl Wrapper for Non-LinkedIn URLs
# ============================================================
MAX_SUBLINKS_PER_PAGE = 3

def crawl_tool(url):
    try:
        out = firecrawl.scrape(url=url, formats=["markdown", "links", "html"])
        return {
            "markdown": out.markdown or "",
            "html": out.html or "",
            "links": out.links or [],
        }
    except:
        return {"markdown": "", "html": "", "links": []}

# ============================================================
# LLM Decider: Should we crawl this page?
# ============================================================
def llm_crawl_decider(snippet, question):
    decision = safe_json(llm, [
        {
            "role": "system",
            "content": 'Return ONLY {"crawl": true} or {"crawl": false}.'
        },
        {"role": "user", "content": f"QUESTION:\n{question}"},
        {"role": "user", "content": f"PAGE SNIPPET:\n{snippet}"}
    ])
    return decision.get("crawl", False)

# ============================================================
# Image Extractor (LinkedIn profile picture)
# ============================================================
# def extract_profile_image(raw, md):
#     """
#     Priority:
#     1) Tavily metadata (primary_image, thumbnail, etc.)
#     2) OG:image tags
#     3) LinkedIn CDN (media.licdn.com)
#     4) Google CDN fallback
#     """
#     # Tavily fields
#     for key in ["primary_image", "image_url", "thumbnail", "favicon", "image"]:
#         if key in raw and isinstance(raw[key], str) and raw[key].startswith("http"):
#             return raw[key]

#     # OG tags
#     if md:
#         m = re.search(r'property="og:image"\s*content="([^"]+)"', md)
#         if m:
#             return m.group(1)

#     # LinkedIn CDN
#     cdn = re.findall(r'https://media\.licdn\.com[^\s"\']+', md or "")
#     if cdn:
#         return cdn[0]

#     # Google fallback
#     gcache = re.findall(r'https://encrypted-tbn0\.gstatic\.com[^"\']+', md or "")
#     if gcache:
#         return gcache[0]

#     return ""

# ============================================================
# CHUNK 2 — Planner → Search → Crawl → Extract
# ============================================================

MAX_PAGES = 20
MAX_DEPTH = 2
PROMISING_KEYWORDS = ["case", "note", "study", "customer", "client", "research"]

# ------------------------------------------------------------
# PLANNER NODE
# ------------------------------------------------------------
def llm_relevance_score(question, tavily_result, debug=False):
    text = tavily_result.get("raw_content") or tavily_result.get("content") or ""
    url = tavily_result.get("url", "")

    snippet = (text or "")[:800]

    # FIXED: escape all literal braces → {{ }}
    prompt = """
You are a strict relevance scoring engine.

Rate how relevant this page is to the user's question.

Question:
{question}

URL:
{url}

Content Snippet:
{snippet}

Return STRICT JSON ONLY:
{{"score": <number between 0 and 1>}}
""".format(question=question, url=url, snippet=snippet)

    resp = llm.invoke([
        {"role": "system", "content": "Return ONLY JSON. No commentary."},
        {"role": "user", "content": prompt}
    ])

    raw = resp.content.strip().replace("```json", "").replace("```", "")

    try:
        data = json.loads(raw)
        score = float(data.get("score", 0))
    except Exception:
        score = 0.0

    if debug:
        st.write("🔍 LLM Relevance Debug →", url)
        st.json({"score": score, "snippet": snippet[:200]})

    return score




def filter_relevant_results(question, results, threshold=0.30, debug=False):
    """
    - Uses LLM to check relevance
    - Keeps LinkedIn OR non-LinkedIn results if relevant
    - threshold=0.30 works best
    """

    kept = []

    for r in results:
        score = llm_relevance_score(question, r["raw"], debug=debug)

        if debug:
            st.write(f"Score for {r['url']}: {score}")

        if score >= threshold:
            kept.append(r)

    if debug:
        st.success(f"Kept {len(kept)} relevant pages.")

    return kept


def planner_node(state):
    question = state["question"].strip()

    # ============================================================
    # 🔥 1) DIRECT LINKEDIN URL MODE (NO LLM PLANNING)
    # ============================================================
    # if question.startswith("http") and "linkedin.com" in question:
    #     st.markdown("🔗 **Direct LinkedIn URL detected — skipping LLM planning**")

    #     direct_query = f'"{question}"'   # exact match for Tavily

    #     # 👉 Fetch top 2 results from Tavily
    #     results = tavily.search(
    #         query=direct_query,
    #         max_results=2,
    #         include_raw_content=True
    #     ).get("results", [])

    #     st.write("### Tavily Returned the following URLs:")
    #     st.json(results)

    #     if not results:
    #         st.error("Tavily did not return results for this LinkedIn URL.")
    #         return {"plan": {"mode": "direct_linkedin", "pages": []}}

    #     # --------------------------------------------------------
    #     # Extract Tavily content into our unified format
    #     # --------------------------------------------------------
    #     pages = []
    #     for r in results:
    #         pages.append({
    #             "url": r["url"],
    #             "raw": r,
    #             "markdown": r.get("raw_content") or r.get("content") or ""
    #         })

    #     # --------------------------------------------------------
    #     # AUTO-DETECT PERSON vs COMPANY from actual URL
    #     # --------------------------------------------------------
    #     is_company = "linkedin.com/company/" in question.lower()
    #     is_person = "linkedin.com/in/" in question.lower()

    #     mode = (
    #         "company_profile" if is_company else
    #         "person_profile" if is_person else
    #         "unknown_linkedin"
    #     )

    #     # RETURN FINAL PLAN (no LLM, no crawling)
    #     return {
    #         "plan": {
    #             "mode": "direct_linkedin",
    #             "linkedin_mode": mode,   # pass forward
    #             "pages": pages           # <== used by crawl_node
    #         }
    #     }
    if question.startswith("http") and "linkedin.com" in question:
        st.markdown("🔗 **Direct LinkedIn URL detected — skipping planning**")

        direct_query = f'"{question}"'

    # Fetch top 2 Tavily results
        results = tavily.search(
            query=direct_query,
            max_results=2,
            include_raw_content=True
        ).get("results", [])

        if not results:
            return {"plan": {"mode": "direct_linkedin", "pages": []}}

        # Apply classifier to result URLs
        classified_pages = []
        for r in results:
            intent = classify_linkedin_url(r["url"])
            classified_pages.append({
                "url": r["url"],
                "intent": intent,
                "raw": r,
                "markdown": r.get("raw_content") or r.get("content") or ""
            })

        return {
            "plan": {
                "mode": "direct_linkedin",
                "pages": classified_pages
            }
        }

    # ============================================================
    # 🔵 2) NORMAL MODE → USE LLM TO GENERATE SEARCH QUERIES
    # ============================================================
    plan = safe_json(llm, [
    {
        "role": "system",
        "content": """
        You are a planning agent.
        Output STRICT JSON:
        {"search_queries": [...], "extraction_focus": [...], "mode": "auto"}
        """
    },
    {"role": "user", "content": question}
])

    st.markdown("### 🧭 Planner Output")
    st.json(plan)

    queries = plan.get("search_queries", [])

    # ===========================================
    # ⭐ PRIORITIZE /in/ or /company/ in search results
    # ===========================================
    intent = detect_intent_llm(question, [])
    if 'intent' not in st.session_state:
        st.session_state['intent'] = intent
    st.write('I am planner node and intent of question is..', intent)
    boosted_queries = []
    for q in queries:
        if intent == "person_profile":
            boosted_queries.append(q + " linkedin profile /in/")
        elif intent == "company_profile":
            boosted_queries.append(q + " linkedin company /company/")
        else:
            boosted_queries.append(q)

    plan["search_queries"] = boosted_queries
    st.write('Plan is..', plan)

    return {"plan": plan}




# ------------------------------------------------------------
# SEARCH NODE (Tavily)
# ------------------------------------------------------------
def search_node(state):
    plan = state["plan"]
    question = state["question"]

    queries = plan.get("search_queries", [])
    urls = []

    st.markdown("### 🔎 Tavily Search Results")

    for q in queries:
        st.write(f"**Searching:** {q}")
        results = search_tool(q)

        for r in results:
            urls.append({"url": r["url"], "raw": r})
            st.write("•", r["url"])

    # ----------------------------------------
    # 🔥 APPLY LLM RELEVANCE FILTERING HERE
    # ----------------------------------------
    st.markdown("### 🧠 Relevance Filtering (LLM)")
    relevant = filter_relevant_results(question, urls, threshold=0.30, debug=True)

    if not relevant:
        st.error("LLM found NO relevant pages.")
        return {"search_urls": []}

    st.success(f"Using {len(relevant)} relevant pages after filtering.")

    # Return filtered URLs
    return {"search_urls": relevant[:5]}

# ------------------------------------------------------------
# CRAWL NODE (Tavily for LinkedIn + Firecrawl for other sites)
# ------------------------------------------------------------
def crawl_node(state):

    plan = state["plan"]

    # DIRECT MODE — return only the Tavily page
    if plan.get("mode") == "direct_linkedin":
        st.markdown("🔵 **Direct LinkedIn mode — skipping all crawling**")
        return {"crawled_pages": plan.get("pages", [])}
    question = state["question"]
    search_urls = state["search_urls"]

    queue = deque([(u, 0) for u in search_urls])
    visited = set()
    crawled = []

    st.markdown("### 🌐 Crawling Activity")
    progress = st.progress(0)
    count = 0

    while queue and len(crawled) < MAX_PAGES:
        url_obj, depth = queue.popleft()
        url = url_obj["url"]
        raw = url_obj["raw"]

        if url in visited or depth > MAX_DEPTH:
            continue
        visited.add(url)

        # -------------------------------
        # LINKEDIN → use Tavily content only
        # -------------------------------
        if is_linkedin(url):
            st.write(f"🔵 **LinkedIn detected — skipping Firecrawl:** {url}")

            md = raw.get("raw_content") or raw.get("content") or ""
            crawled.append({
                "url": url,
                "markdown": md,
                "raw": raw
            })

            count += 1
            progress.progress(count / MAX_PAGES)
            continue

        # -------------------------------
        # RELEVANCE DECISION
        # -------------------------------
        if raw and isinstance(raw, dict):
            text = raw.get("raw_content") or raw.get("content") or ""
        else:
            text = ""

        snippet = text[:800]

        if not llm_crawl_decider(snippet, question):
            st.write(f"⛔ **Skipped (irrelevant):** {url}")
            continue

        # -------------------------------
        # FIRECRAWL (non-LinkedIn)
        # -------------------------------
        st.write(f"🟢 **Crawling:** {url}")
        crawled_page = crawl_tool(url)

        crawled.append({
            "url": url,
            "markdown": crawled_page["markdown"],
            "raw": raw
        })

        # -------------------------------
        # Add promising sub-links
        # -------------------------------
        sublinks = crawled_page.get("links", [])[:MAX_SUBLINKS_PER_PAGE]

        for sub in sublinks:
            s = sub.lower()
            if any(k in s for k in PROMISING_KEYWORDS):
                queue.append(({"url": sub, "raw": {"content": ""}}, depth + 1))

        count += 1
        progress.progress(count / MAX_PAGES)

    return {"crawled_pages": crawled}


# ------------------------------------------------------------
# EXTRACTOR NODE (Identify: person / company / generic page)
# ------------------------------------------------------------
import re

def extractor_node(state):
    pages = state["crawled_pages"]
    question = state["question"].lower()

    extracted = []
    st.markdown("### 📄 Extracted Pages Summary")

    intent = detect_intent_llm(question, pages)
    st.write('Intent of pages is..', intent)
    st.session_state['intent'] = intent


    # ====================================
    # FIXED URL CLASSIFIERS
    # ====================================
    def is_real_profile(u: str) -> bool:
        return bool(re.search(r"linkedin\.com\/in\/[^\/?#]+\/?$", u))

    def is_real_company(u: str) -> bool:
        return bool(re.search(r"linkedin\.com\/company\/[^\/?#]+\/?$", u))

    # ====================================
    # UPDATED SCORING — USE STRICT RULES
    # ====================================
    def score_url(u: str) -> int:
        u = u.lower()

        if intent == "person_profile":
            if is_real_profile(u):
                return 200   # highest score
            if "linkedin.com/posts/" in u: 
                return 20
            if "linkedin.com/company/" in u:
                return 5
            return 1

        if intent == "company_profile":
            if is_real_company(u):
                return 200
            if is_real_profile(u):
                return 10
            if "linkedin.com/posts/" in u:
                return 5
            return 1

        return 1

    pages_sorted = sorted(pages, key=lambda p: score_url(p["url"]), reverse=True)

    # ====================================
    # EXTRACTION LOOP
    # ====================================
    for p in pages_sorted:
        url = p["url"].lower()
        text = p["markdown"]
        raw = p["raw"]

        extracted.append({
            "url": p["url"],
            "is_profile": is_real_profile(url),
            "is_company": is_real_company(url),
            "is_post": ("linkedin.com/posts/" in url),
            "content": text,
            "raw": raw,
            "image_url": extract_profile_image(raw, text)
        })

        st.write(
            f"• {p['url']} — "
            f"Profile: {is_real_profile(url)}, "
            f"Company: {is_real_company(url)}, "
            f"Post: {'linkedin.com/posts/' in url}, "
            f"Score: {score_url(url)}"
        )

    return {"extracted": extracted}


# ============================================================
# CHUNK 3 — Fact Extractor + Persona Engine + Synthesizer
# ============================================================


# ------------------------------------------------------------
# PERSON FACT EXTRACTOR
# ------------------------------------------------------------
def extract_person_facts(extracted_pages):
    """
    Extracts clean LinkedIn PERSON factual data.
    Only reads what is actually present.
    """
    system_prompt = """
    You are a strict LinkedIn profile parser.
    Read ALL crawled pages and extract ONLY factual information.

    Return STRICT JSON:
    {
      "name": "",
      "headline": "",
      "current_title": "",
      "current_company": "",
      "location": "",
      "followers": "",
      "total_experience_years": "",
      "past_roles": [{"title":"", "company":"", "duration":""}],
      "education": [{"school":"", "degree":"", "years":""}],
      "skills": [],
      "about": "",
      "profile_url": ""
    }

    Rules:
    - NO hallucinations.
    - Use ONLY text from LinkedIn pages.
    - Prefer linkedin.com/in/ pages over posts.
    """

    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(extracted_pages)}
    ])

    return safe_json(llm, [{"role": "user", "content": resp.content}])


# ------------------------------------------------------------
# COMPANY FACT EXTRACTOR
# ------------------------------------------------------------
def extract_company_facts(extracted_pages):
    """
    Clean LinkedIn COMPANY profile extractor.
    """
    system_prompt = """
    Extract.LinkedIn COMPANY facts ONLY.
    Return strict JSON:
    {
        "name": "",
        "industry": "",
        "followers": "",
        "employees": "",
        "headquarters": "",
        "year_of_establishment": "",
        "company_url": ""
    }

    Rules:
    - Must read raw_content from LinkedIn company pages.
    - Do NOT guess. Only extract if present.
    - If multiple values appear, pick the most authoritative one.
    """

    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(extracted_pages)}
    ])

    return safe_json(llm, [{"role": "user", "content": resp.content}])


# ------------------------------------------------------------
# PERSONA ENGINE (Big Five, DISC, Work Style, Cold Call)
# ------------------------------------------------------------
def persona_engine(extracted_pages):
    system_prompt = """
    Produce STRICT JSON persona model:

{
  "big_five": {...},
  "disc": {...},
  "work_style": {...},
  "communication_style": {...},
  "sales_guidance": {...},
  "personalization": {...},
  "cold_call": {
      "what_to_say": "",
      "what_not_to_say": "",
      "script": {
          "0_20_seconds": "",
          "20_40_seconds": "",
          "40_60_seconds": "",
          "60_80_seconds": "",
          "80_100_seconds": "",
          "100_120_seconds": ""
      }
  },
  "buyer_intent": {...},
  "confidence": 0-1
}

Rules:
- Cold call script MUST be a **2-minute script only** (NOT 5 minutes).
- Break into **6 clean segments** of 20 seconds each.
- Each segment must contain **one crisp objective**, not long paragraphs.
- Make the talk-track aligned to this specific buyer’s persona.
- Avoid repetition between segments.
- Always provide actionable guidance; no fluff.
- JSON must be strictly valid.
"""
    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(extracted_pages)}
    ])

    return safe_json(llm, [{"role": "user", "content": resp.content}])


# ------------------------------------------------------------
# FINAL HUMANTIC-STYLE SYNTHESIZER (NO duplication)
# ------------------------------------------------------------
def prioritize_for_summary(extracted, intent):
    """
    Returns extracted pages sorted so that
    PERSON → /in/ first → posts → rest
    COMPANY → /company/ first → posts → rest
    GENERAL → no ordering
    """

    def score(p):
        u = p["url"].lower()

        if intent == "person_profile":
            if "linkedin.com/in/" in u: return 300
            if "linkedin.com/posts/" in u: return 150
            return 10

        if intent == "company_profile":
            if "linkedin.com/company/" in u: return 300
            if "linkedin.com/posts/" in u: return 150
            return 10

        return 1

    return sorted(extracted, key=score, reverse=True)


def synthesis_node(state):
    extracted = state["extracted"]
    question = state["question"]

    # ---------------------------------------------------------
    # 1️⃣ LET LLM DECIDE THE TRUE INTENT
    # ---------------------------------------------------------
    intent = st.session_state['intent']
    st.write('I am in synthesis node and intent is..', st.session_state['intent'])
    # returns: person_profile / company_profile / general_research

    # ---------------------------------------------------------
    # 2️⃣ FLAGS FROM CRAWLER (used only to SUPPORT logic)
    # ---------------------------------------------------------
    has_profile_page = any(x["is_profile"] for x in extracted)
    has_company_page = any(x["is_company"] for x in extracted)

    # =========================================================
    # 🔵 PERSON MODE (when LLM says it is a person)
    # =========================================================
    if intent == "person_profile":

        # Fallback safeguard → if no profile page found but task is person
        if not has_profile_page:
            # keep processing anyway → persona engine still works
            pass

        facts = extract_person_facts(extracted)
        persona = persona_engine(extracted)

        # -------------------------------
        # 2-MIN COLD CALL SCRIPT
        # -------------------------------
        # Replace the 5-minute script with 2-minute structure
        narrative_prompt = """
        Create a HUMANTIC-style narrative report using ONLY the provided facts/persona.

        DO NOT repeat JSON fields. DO NOT hallucinate. 
        KEEP the writing tight, clean, professional.

        SECTIONS REQUIRED:

        1) LinkedIn Profile Summary (factual, crisp)
        2) LinkedIn followers
        2) Professional Overview (5–7 lines)
        3) Behavioural Summary (3–5 lines)
        4) Communication & Influence Style (short)
        5) Personalized Cold Call Guidance (short)
        6) **2-Minute Cold Call Script** → Break into:
           - First 30 seconds (Opener)
           - 30–60 seconds (Value Hook)
           - 60–90 seconds (Relevance Pitch)
           - 90–120 seconds (Close / CTA)
        7) Buyer Intent Signals (short)

        DO NOT mention persona JSON. DO NOT repeat big-five or DISC numbers.
        Just synthesize.
        """

        combined = {"facts": facts, "persona": persona}
        ranked = prioritize_for_summary(extracted, intent)

        final_answer = llm.invoke([
            {"role": "system", "content": narrative_prompt},
            {"role": "user", "content": json.dumps(ranked)}
        ]).content

        img = next((x["image_url"] for x in extracted if x["image_url"]), "")

        return {
            "mode": "person",
            "answer": final_answer,
            "facts": facts,
            "persona": persona,
            "image_url": img,
            "profile_url": facts.get("profile_url", "")
        }


   # ---------------------------------------------------------
    # 🟣 COMPANY MODE  (Bullet-point Summary)
    # ---------------------------------------------------------
    if intent == "company_profile":
        facts = extract_company_facts(extracted)

        narrative_prompt = """
        You are a Company Intelligence Writer.

        Transform the provided COMPANY FACTS into a strictly formatted,
        BULLET-POINT SUMMARY.
        STRICT REQUIREMENTS:
        ⚠ ABSOLUTE NON-NEGOTIABLE RULES ⚠
        - ONLY bullet points (each line MUST start with "- ")
        - NO paragraphs
        - NO continuous text longer than one sentence
        - NO storytelling
        - NO description beyond factual wording
        - Do NOT infer anything not explicitly present
        - If a field is not found → skip it
        - ALWAYS include Followers if present
        - ALWAYS place LinkedIn URL as the LAST bullet

        REQUIRED BULLET FORMAT (example):
        - **Overview:** <one crisp factual line>
        - **Industry:** <value>
        - **Headquarters:** <value>
        - **Employees:** <value>
        - **Followers:** <value>
        - **Founded:** <value>
        - **Key Focus Areas:** <comma-separated list, extracted ONLY from given text>
        - **LinkedIn:** <url>

        DO NOT return anything except bullet points.
        DO NOT write a paragraph summary.
        """

        final_answer = llm.invoke([
            {"role": "system", "content": narrative_prompt},
            {"role": "user", "content": json.dumps(facts)}
        ]).content

        img = next((x["image_url"] for x in extracted if x["image_url"]), "")

        return {
            "mode": "company",
            "answer": final_answer,
            "facts": facts,
            "persona": {},
            "image_url": img,
            "profile_url": facts.get("company_url", "")
        }

    # =========================================================
    # 🟢 GENERAL TOPIC MODE
    # =========================================================
    # Example queries:
    # - "recent developments at Nestle India"
    # - "market updates on EV batteries"
    # - "find me latest benchmarks"
    system_prompt = """
    Summarize clearly and factually using the extracted content. 
    No persona. No LinkedIn formatting. No hallucinations. 
    Just a clean research summary.
    """

    final = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(extracted)}
    ]).content

    return {
        "mode": "generic",
        "answer": final,
        "facts": {},
        "persona": {},
        "image_url": "",
        "profile_url": ""
    }

# =====================================================================
# AGENT GRAPH — DEFINES THE FULL STATE MACHINE
# =====================================================================

class ResearchState(TypedDict):
    question: str
    plan: dict
    search_urls: list
    crawled_pages: list
    extracted: list
    mode: str
    answer: str
    facts: dict
    persona: dict
    image_url: str
    profile_url: str


# -----------------------------
# BUILD GRAPH
# -----------------------------
graph = StateGraph(ResearchState)

# Add nodes (declared in chunks 1–3)
graph.add_node("planner", planner_node)
graph.add_node("search", search_node)
graph.add_node("crawl", crawl_node)
graph.add_node("extract", extractor_node)
graph.add_node("synthesize", synthesis_node)

# Connect nodes
graph.add_edge("planner", "search")
graph.add_edge("search", "crawl")
graph.add_edge("crawl", "extract")
graph.add_edge("extract", "synthesize")

# Entry + Exit
graph.set_entry_point("planner")
graph.set_finish_point("synthesize")

# IMPORTANT: compile graph ONCE
graph = graph.compile()

# =====================================================================
# STREAMLIT UI — HYBRID HUMANTIC LAYOUT (FINAL)
# =====================================================================

st.set_page_config(page_title="LinkedIn GPT", layout="wide")
st.title("🧠 LinkedIN GPT - No crawling of LinkedIN!!!")
st.markdown("Works on publicly available data")

# =====================================================================
# DIRECT LINKEDIN URL HANDLING
# =====================================================================

import re

def normalize_query(user_query: str):
    """
    If user enters a LinkedIn URL:
    → we DO NOT search
    → we DO NOT add keywords
    → we FORCE the system to process ONLY that URL
    """

    q = user_query.strip()

    if q.startswith("http://") or q.startswith("https://"):
        if "linkedin.com/in/" in q or "linkedin.com/company/" in q:
            # important: no search keywords added
            return q  

    return q

query = st.text_area("Enter LinkedIn URL or Search Query:", height=110)
run_btn = st.button("Analyze")

if run_btn and query.strip():

    with st.spinner("Running Deep Multi-Source Analysis…"):
        norm_query = normalize_query(query)
        result = graph.invoke({"question": norm_query})

    # --------------------------------------------------------
    # UNPACK RESULT
    # --------------------------------------------------------
    mode = result.get("mode")
    answer = result.get("answer", "")

    facts = result.get("facts", {}) or {}
    persona = result.get("persona", {}) or {}

    image_url = result.get("image_url", "")
    profile_url = result.get("profile_url", "")

    # ========================================================
    # 🧭 MODE DISPLAY
    # ========================================================
    # mode_label = {
    #     "person": "👤 PERSON PROFILE MODE",
    #     "company": "🏢 COMPANY PROFILE MODE",
    #     "generic": "🌐 GENERAL RESEARCH MODE"
    # }.get(mode, "🌐 GENERAL MODE")

    mode = st.session_state['intent']
    st.markdown(f"### {mode}")

    # ========================================================
    # 🟪 CREATE TABS
    # ========================================================
    tab_summary, tab_persona, tab_coldcall, tab_links, tab_debug = st.tabs(
        ["📄 Summary", "🧠 Persona Dashboard", "📞 Cold Call Intelligence",
         "🔗 Crawled Links", "🐞 Debug JSON"]
    )

    # ========================================================
    # 📄 TAB 1 — SUMMARY
    # ========================================================
    with tab_summary:

        # ---------------------------
        # PERSON MODE UI
        # ---------------------------
        if mode == "person_profile":

            col1, col2 = st.columns([1, 3])

            with col1:
                if image_url:
                    st.image(image_url, width=260)
                else:
                    st.info("No profile image detected.")

            with col2:
                # DISC Badge
                disc_scores = persona.get("disc", {})
                if disc_scores:
                    st.markdown(render_disc_badge(disc_scores),
                                unsafe_allow_html=True)

                # High-level identity block
                st.markdown(f"""
                # **{facts.get("name", "Unknown Name")}**
                ### {facts.get("headline", "")}

                **{facts.get("current_title","")} — {facts.get("current_company","")}**  
                📍 {facts.get("location","unknown")}  
                ⭐ Experience: **{facts.get("total_experience_years","unknown")} years**  
                🌐 Followers: **{facts.get("followers","unknown")}**
                """)

                if profile_url:
                    st.markdown(f"🔗 **LinkedIn:** {profile_url}")

            st.markdown("---")
            st.markdown("### 📝 Humantic-Style Summary")
            st.write(answer)

        # ---------------------------
        # COMPANY MODE UI
        # ---------------------------
        elif mode == "company_profile":
            col1, col2 = st.columns([1, 3])

            with col1:
                if image_url:
                    st.image(image_url, width=250)

            with col2:
                st.markdown(f"""
                # 🏢 **{facts.get("name","Unknown Company")}**
                **Industry:** {facts.get("industry","unknown")}  
                **Followers:** {facts.get("followers","unknown")}  
                **Employees:** {facts.get("employees","unknown")}  
                **Headquarters:** {facts.get("headquarters","unknown")}  
                **Established:** {facts.get("year_of_establishment","unknown")}  
                """)

                if facts.get("company_url"):
                    st.markdown(f"🔗 **LinkedIn:** {facts.get('company_url')}")

            st.markdown("---")
            st.markdown("### 📝 Company Summary")
            st.write(answer)

        # ---------------------------
        # GENERIC MODE UI
        # ---------------------------
        else:
            st.markdown("### 📄 Research Summary")
            st.write(answer)

    # ========================================================
    # 🧠 TAB 2 — PERSONA DASHBOARD
    # ========================================================
    with tab_persona:
        if mode != "person_profile":
            st.info("Persona insights available only in PERSON MODE.")
        else:
            colA, colB = st.columns(2)

            with colA:
                st.markdown("### 🔷 Big Five Radar Chart")
                bigfive = persona.get("big_five", {})
                if bigfive:
                    fig = plot_big_five_radar(bigfive)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Big Five values missing.")

            with colB:
                st.markdown("### 🧭 DISC Wheel")
                disc = persona.get("disc", {})
                if disc:
                    fig = plot_disc_wheel(disc)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("DISC values missing.")

            st.markdown("---")
            st.markdown("### 🧩 Work Style")
            st.json(persona.get("work_style", {}))

            st.markdown("### 🗣 Communication Style")
            st.json(persona.get("communication_style", {}))

            st.markdown("### 💡 Sales Guidance")
            st.json(persona.get("sales_guidance", {}))

            st.markdown("### 🎯 Personalization Insights")
            st.json(persona.get("personalization", {}))

            st.markdown("### 🛒 Buyer Intent Signals")
            st.json(persona.get("buyer_intent", {}))

            st.markdown("### 🔐 Confidence Score")
            conf = persona.get("confidence")
            if conf is not None:
                st.write(f"{conf:.2f}")

    # ========================================================
    # 📞 TAB 3 — COLD CALL INTELLIGENCE
    # ========================================================
    with tab_coldcall:
        if mode != "person_profile":
            st.info("Cold call intelligence is available only for individuals.")
        else:
            cold = persona.get("cold_call", {})

            st.markdown("### 🟢 What to Say")
            st.write(cold.get("what_to_say", ""))

            st.markdown("### 🔴 What NOT to Say")
            st.write(cold.get("what_not_to_say", ""))

            st.markdown("### 📞 Full 5-Minute Cold Call Script")
            st.write(cold.get("script", ""))

    # ========================================================
    # 🔗 TAB 4 — CRAWLED LINKS
    # ========================================================
    with tab_links:
        st.markdown("### 🌍 All Crawled Links")
        crawled_list = [p["url"] for p in result.get("crawled_pages", [])]
        st.write("\n".join(crawled_list) if crawled_list else "No links crawled")

    # ========================================================
    # 🐞 TAB 5 — DEBUG JSON
    # ========================================================
    with tab_debug:
        st.markdown("### 🐞 Full JSON Output")
        st.json(result)



