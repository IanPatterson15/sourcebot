import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass
from openai import OpenAI
from qdrant_client import QdrantClient
import streamlit as st
import base64
import requests
from urllib.parse import urlparse
import re

def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "economics_papers"

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_papers(query, n_results=5):
    query_embedding = get_embedding(query)
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=n_results,
        with_payload=True
    ).points
    ids = [str(h.id) for h in hits]
    metadatas = [h.payload for h in hits]
    documents = [h.payload.get("text", "") for h in hits]
    return {
        "ids": [ids],
        "metadatas": [metadatas],
        "documents": [documents]
    }

def analyze_paper(abstract, query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a research assistant helping undergraduates find sources.
                Given a student's research query and a paper's abstract, provide:
                1. A 2-3 sentence explanation of why this paper is relevant to their argument
                2. 2-3 direct quotes from the abstract that are most relevant to their search

                Format your response exactly like this:
                RELEVANCE: [your explanation here]
                QUOTES:
                - "[quote 1]"
                - "[quote 2]"
                - "[quote 3]"
                """
            },
            {
                "role": "user",
                "content": f"Student is looking for: {query}\n\nAbstract: {abstract}"
            }
        ]
    )
    return response.choices[0].message.content

def parse_analysis(analysis):
    relevance = ""
    quotes = []
    lines = analysis.split("\n")
    in_quotes = False
    for line in lines:
        if line.startswith("RELEVANCE:"):
            relevance = line.replace("RELEVANCE:", "").strip()
        elif line.startswith("QUOTES:"):
            in_quotes = True
        elif in_quotes and line.strip().startswith("-"):
            quote = line.strip()[1:].strip().strip('"')
            if quote:
                quotes.append(quote)
    return relevance, quotes

def format_apa(title, authors, year, doi):
    if authors and authors != "Unknown":
        author_list = [a.strip() for a in authors.split(",")]
        formatted_authors = []
        for author in author_list:
            parts = author.strip().split()
            if len(parts) >= 2:
                last = parts[-1]
                initials = " ".join([p[0] + "." for p in parts[:-1]])
                formatted_authors.append(f"{last}, {initials}")
            else:
                formatted_authors.append(author)
        if len(formatted_authors) == 1:
            author_str = formatted_authors[0]
        elif len(formatted_authors) == 2:
            author_str = " & ".join(formatted_authors)
        else:
            author_str = ", ".join(formatted_authors[:-1]) + ", & " + formatted_authors[-1]
    else:
        author_str = "Unknown Author"

    year_str = f"({year})" if year else "(n.d.)"
    title_str = title if title else "Untitled"
    doi_str = f" {doi}" if doi else ""
    return f"{author_str} {year_str}. {title_str}.{doi_str}"

def format_mla_journal(title, authors, year, journal, volume, issue, pages, doi):
    if authors and authors.strip():
        author_list = [a.strip() for a in authors.split(",")]
        if len(author_list) == 1:
            parts = author_list[0].strip().split()
            if len(parts) >= 2:
                author_str = f"{parts[-1]}, {' '.join(parts[:-1])}."
            else:
                author_str = author_list[0] + "."
        elif len(author_list) == 2:
            p1 = author_list[0].strip().split()
            p2 = author_list[1].strip().split()
            first = f"{p1[-1]}, {' '.join(p1[:-1])}" if len(p1) >= 2 else author_list[0]
            second = f"{' '.join(p2[:-1])} {p2[-1]}" if len(p2) >= 2 else author_list[1]
            author_str = f"{first}, and {second}."
        else:
            p1 = author_list[0].strip().split()
            first = f"{p1[-1]}, {' '.join(p1[:-1])}" if len(p1) >= 2 else author_list[0]
            author_str = f"{first}, et al."
    else:
        author_str = ""

    title_str = f'"{title.strip()}"' if title else '"Untitled."'
    journal_str = f"*{journal.strip()}*" if journal else "*Unknown Journal*"

    vol_issue = ""
    if volume and issue:
        vol_issue = f", vol. {volume}, no. {issue}"
    elif volume:
        vol_issue = f", vol. {volume}"

    year_str = f", {year}" if year else ""
    pages_str = f", pp. {pages}" if pages else ""
    doi_str = f", {doi}" if doi else ""

    parts = [p for p in [author_str, f"{title_str}.", f"{journal_str}{vol_issue}{year_str}{pages_str}{doi_str}."] if p]
    return " ".join(parts)

def format_mla_website(title, authors, site_name, url, access_date, publish_date):
    if authors and authors.strip():
        author_list = [a.strip() for a in authors.split(",")]
        if len(author_list) == 1:
            parts = author_list[0].strip().split()
            author_str = f"{parts[-1]}, {' '.join(parts[:-1])}." if len(parts) >= 2 else author_list[0] + "."
        elif len(author_list) == 2:
            p1 = author_list[0].strip().split()
            p2 = author_list[1].strip().split()
            first = f"{p1[-1]}, {' '.join(p1[:-1])}" if len(p1) >= 2 else author_list[0]
            second = f"{' '.join(p2[:-1])} {p2[-1]}" if len(p2) >= 2 else author_list[1]
            author_str = f"{first}, and {second}."
        else:
            p1 = author_list[0].strip().split()
            first = f"{p1[-1]}, {' '.join(p1[:-1])}" if len(p1) >= 2 else author_list[0]
            author_str = f"{first}, et al."
    else:
        author_str = ""

    title_str = f'"{title.strip()}."' if title else '"Untitled."'
    site_str = f"*{site_name.strip()}*" if site_name else ""
    pub_str = f"{publish_date}," if publish_date else ""
    url_str = f"{url}." if url else ""
    access_str = f"Accessed {access_date}." if access_date else ""

    parts = [p for p in [author_str, title_str, site_str + ("," if site_str else ""), pub_str, url_str, access_str] if p and p not in [",", ""]]
    return " ".join(parts)

def fetch_metadata_from_doi(doi):
    doi_clean = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi.strip())
    try:
        url = f"https://api.crossref.org/works/{doi_clean}"
        headers = {"User-Agent": "SourceBot/1.0 (mailto:sourcebot@school.edu)"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return None, f"DOI not found (status {resp.status_code}). Please fill in fields manually."

        data = resp.json().get("message", {})
        title_list = data.get("title", [])
        title = title_list[0] if title_list else ""
        authors_raw = data.get("author", [])
        author_names = []
        for a in authors_raw:
            given = a.get("given", "")
            family = a.get("family", "")
            if given and family:
                author_names.append(f"{given} {family}")
            elif family:
                author_names.append(family)
        authors = ", ".join(author_names)
        year = ""
        date_parts = data.get("published", {}).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = str(date_parts[0][0])
        container = data.get("container-title", [])
        journal = container[0] if container else ""
        volume = data.get("volume", "")
        issue = data.get("issue", "")
        pages = data.get("page", "")
        return {
            "title": title, "authors": authors, "year": year,
            "doi": f"https://doi.org/{doi_clean}", "journal": journal,
            "volume": volume, "issue": issue, "pages": pages, "source_type": "journal"
        }, None
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please fill in fields manually."
    except Exception as e:
        return None, f"Could not fetch metadata: {str(e)}"

def fetch_metadata_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SourceBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return None, f"Could not access URL (status {resp.status_code}). Please fill in fields manually."

        html = resp.text

        def get_meta(name=None, prop=None):
            if prop:
                match = re.search(rf'<meta[^>]+property=["\']og:{prop}["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                if not match:
                    match = re.search(rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:{prop}["\']', html, re.IGNORECASE)
            else:
                match = re.search(rf'<meta[^>]+name=["\']{name}["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                if not match:
                    match = re.search(rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']{name}["\']', html, re.IGNORECASE)
            return match.group(1).strip() if match else ""

        title = get_meta(prop="title")
        if not title:
            title = get_meta(name="title")
        if not title:
            t_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
            title = t_match.group(1).strip() if t_match else ""

        author = get_meta(name="author")
        if not author:
            author = get_meta(name="article:author")

        site_name = get_meta(prop="site_name")
        if not site_name:
            parsed = urlparse(url)
            site_name = parsed.netloc.replace("www.", "").split(".")[0].capitalize()

        publish_date = get_meta(name="article:published_time")
        if not publish_date:
            publish_date = get_meta(name="pubdate")
        if not publish_date:
            publish_date = get_meta(prop="published_time")
        if publish_date:
            date_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', publish_date)
            if date_match:
                months = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "June", "July", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
                y, m, d = date_match.groups()
                publish_date = f"{int(d)} {months[int(m)-1]} {y}"

        return {
            "title": title, "authors": author, "site_name": site_name,
            "url": url, "publish_date": publish_date, "source_type": "website"
        }, None
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please fill in fields manually."
    except Exception as e:
        return None, f"Could not fetch metadata: {str(e)}"

def detect_and_fetch(raw_input):
    raw = raw_input.strip()
    is_doi = (
        re.match(r'^10\.\d{4,}/', raw) or
        re.match(r'^https?://(dx\.)?doi\.org/10\.', raw)
    )
    if is_doi:
        return fetch_metadata_from_doi(raw)
    elif raw.startswith("http://") or raw.startswith("https://"):
        return fetch_metadata_from_url(raw)
    else:
        return None, "Please enter a valid URL (starting with https://) or a DOI (starting with 10.)."


# ---- SESSION STATE ----
if "page" not in st.session_state:
    st.session_state.page = "home"
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "cite_result" not in st.session_state:
    st.session_state.cite_result = None
if "cite_generated" not in st.session_state:
    st.session_state.cite_generated = False
if "cite_prefill" not in st.session_state:
    st.session_state.cite_prefill = {}
if "cite_fetch_error" not in st.session_state:
    st.session_state.cite_fetch_error = None
if "cite_fetch_success" not in st.session_state:
    st.session_state.cite_fetch_success = False

# ---- PAGE CONFIG ----
st.set_page_config(page_title="SourceBot", page_icon="·", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
#  DESIGN SYSTEM CSS  — Sophisticated Dark / Academic Premium
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400;1,500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

/* ── TOKENS ─────────────────────────────────────────────── */
:root {
    --bg:         #0b0d13;
    --surface:    #111420;
    --surface2:   #161924;
    --border:     rgba(255,255,255,0.06);
    --border2:    rgba(255,255,255,0.11);
    --accent:     #c4a35a;
    --accent-dim: rgba(196,163,90,0.12);
    --accent-glow:rgba(196,163,90,0.18);
    --blue:       #8fa4e8;
    --blue-dim:   rgba(143,164,232,0.08);
    --text:       #e6e1d6;
    --muted:      #74788a;
    --dim:        #3d4050;
}

/* ── RESET / BASE ───────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.main, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
}

/* Subtle paper grain */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
}

[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
[data-testid="block-container"], .block-container {
    padding: 0 !important; margin-top: 0 !important; max-width: 100% !important;
}
[data-testid="stVerticalBlock"] { gap: 0 !important; }
section.main > div { padding-top: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }

/* ── NAVBAR ─────────────────────────────────────────────── */
.top-navbar {
    position: fixed; top: 0; left: 0; right: 0; height: 60px;
    background: rgba(11,13,19,0.9);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 3rem;
    z-index: 1000;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-sizing: border-box;
}

.navbar-brand {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.1rem; font-weight: 500;
    color: var(--text) !important;
    text-decoration: none !important;
    letter-spacing: 0.01em;
}
.navbar-brand:hover { color: var(--text) !important; text-decoration: none !important; }

.navbar-brand-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
}

.navbar-links {
    display: flex; gap: 2px; align-items: center;
}

.navbar-links a {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem;
    font-weight: 400;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted) !important;
    text-decoration: none !important;
    padding: 6px 16px;
    border-radius: 2px;
    transition: color 0.2s, background 0.2s;
    white-space: nowrap;
}
.navbar-links a:hover {
    color: var(--text) !important;
    background: rgba(255,255,255,0.04);
}
.navbar-links a.active {
    color: var(--text) !important;
    background: rgba(255,255,255,0.07);
}

.navbar-spacer { height: 60px; }

/* ── FORM ELEMENTS ──────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important; font-weight: 300 !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: var(--dim) !important; opacity: 1 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px rgba(196,163,90,0.15) !important;
    outline: none !important;
}

.stTextInput label, .stTextArea label, .stSelectbox label,
[data-testid="stSlider"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important; font-weight: 400 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: var(--dim) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── BUTTONS ────────────────────────────────────────────── */
.btn-primary .stButton > button,
.search-btn .stButton > button,
.cite-btn .stButton > button {
    background: var(--accent) !important;
    color: #0b0d13 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important; font-weight: 500 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.btn-primary .stButton > button:hover,
.search-btn .stButton > button:hover,
.cite-btn .stButton > button:hover {
    opacity: 0.86 !important; transform: translateY(-1px) !important;
}

.btn-secondary .stButton > button,
.cite-clear-btn .stButton > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important; font-weight: 400 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    padding: 9px 20px !important;
    transition: all 0.2s !important;
}
.btn-secondary .stButton > button:hover,
.cite-clear-btn .stButton > button:hover {
    color: var(--text) !important;
    border-color: rgba(255,255,255,0.22) !important;
    background: rgba(255,255,255,0.03) !important;
}

/* ── SLIDER ─────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ── ALERTS ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    border-left-width: 2px !important;
}

/* ── RESULT CARD ────────────────────────────────────────── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    padding: 28px 32px 22px;
    margin-bottom: 0;
    position: relative;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: var(--border2); border-left-color: var(--accent); }

[data-testid="stCode"] {
    margin-top: 0 !important; margin-bottom: 20px !important;
    border-radius: 0 !important;
    border: 1px solid var(--border) !important;
    border-top: none !important; border-left: 2px solid transparent !important;
    background: rgba(0,0,0,0.25) !important;
}
[data-testid="stCode"] code {
    font-size: 0.8rem !important;
    color: var(--muted) !important;
    font-style: italic !important;
    font-family: 'DM Mono', monospace !important;
}

.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.08rem; font-weight: 500;
    color: var(--text); margin-bottom: 4px; line-height: 1.35;
    letter-spacing: -0.01em;
}
.result-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.08em;
    color: var(--muted); margin-bottom: 18px;
}
.result-section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.14em;
    color: var(--accent); margin-bottom: 6px; margin-top: 16px;
    display: flex; align-items: center; gap: 8px;
}
.result-section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border2), transparent); max-width: 40px;
}
.result-relevance {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem; line-height: 1.7;
    color: rgba(230,225,214,0.75); font-weight: 300;
}
.result-quote {
    font-style: italic;
    color: var(--muted);
    border-left: 2px solid var(--accent);
    padding: 8px 14px;
    margin: 6px 0;
    font-size: 0.85rem; line-height: 1.6;
    background: var(--accent-dim);
    font-family: 'DM Sans', sans-serif;
}
.result-link a {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--accent) !important; text-decoration: none !important;
    transition: color 0.15s;
}
.result-link a:hover { color: var(--text) !important; }

/* ── HISTORY PANEL ──────────────────────────────────────── */
.history-panel {
    background: var(--surface);
    border-right: 1px solid var(--border);
    min-height: calc(100vh - 60px);
    padding: 28px 16px;
    box-sizing: border-box;
}
.history-panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.16em;
    color: var(--accent); margin-bottom: 16px;
    padding-bottom: 12px; border-bottom: 1px solid var(--border);
}
.history-panel-empty {
    font-size: 0.78rem; color: var(--dim);
    font-style: italic; line-height: 1.6;
}
.history-item {
    font-size: 0.75rem; color: var(--muted);
    padding: 8px 10px; margin-bottom: 4px;
    border: 1px solid transparent;
    line-height: 1.45; word-break: break-word;
    transition: all 0.18s; cursor: default;
    border-left: 1px solid var(--border);
}
.history-item:hover {
    border-left-color: var(--accent);
    color: var(--text);
    background: rgba(255,255,255,0.02);
}

/* ── LANDING PAGE ───────────────────────────────────────── */
.landing-wrapper {
    min-height: calc(100vh - 100px);
    display: flex; flex-direction: row;
    align-items: center; justify-content: space-between;
    padding: 0 5rem 0 5rem;
    box-sizing: border-box; position: relative; overflow: hidden;
}

.landing-wrapper::before {
    content: '';
    position: absolute; top: -200px; right: -100px;
    width: 600px; height: 600px;
    background: radial-gradient(ellipse, rgba(143,164,232,0.04) 0%, transparent 70%);
    pointer-events: none;
}

.landing-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 24px;
    display: flex; align-items: center; gap: 12px;
}
.landing-eyebrow::before {
    content: ''; display: block; width: 28px; height: 1px; background: var(--accent);
}
.landing-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3.4rem, 5.5vw, 5.2rem);
    font-weight: 400; line-height: 1.05;
    letter-spacing: -0.025em; color: var(--text);
    margin-bottom: 22px;
}
.landing-title em {
    font-style: italic; color: var(--accent);
}
.landing-tagline {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.98rem; font-weight: 300; line-height: 1.7;
    color: var(--muted); margin-bottom: 44px; max-width: 400px;
}
.landing-actions {
    display: flex; gap: 14px; flex-wrap: wrap;
}
.landing-btn-primary {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--accent); color: #0b0d13 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 13px 28px; text-decoration: none !important;
    transition: opacity 0.2s, transform 0.15s;
}
.landing-btn-primary:hover {
    opacity: 0.85; transform: translateY(-1px);
    color: #0b0d13 !important; text-decoration: none !important;
}
.landing-btn-secondary {
    display: inline-flex; align-items: center; gap: 8px;
    background: transparent; color: var(--text) !important;
    border: 1px solid var(--border2);
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem; font-weight: 400;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 12px 28px; text-decoration: none !important;
    transition: border-color 0.2s, background 0.2s, transform 0.15s;
}
.landing-btn-secondary:hover {
    border-color: rgba(255,255,255,0.22);
    background: rgba(255,255,255,0.03);
    transform: translateY(-1px);
    color: var(--text) !important; text-decoration: none !important;
}
.landing-stat-bar {
    display: flex; gap: 0; margin-top: 56px;
    border-top: 1px solid var(--border); padding-top: 28px;
    width: 100%;
}
.landing-stat {
    flex: 1; text-align: center; padding: 0;
    border-right: 1px solid var(--border);
}
.landing-stat:last-child { border-right: none; }
.landing-stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem; font-weight: 500;
    color: var(--text); letter-spacing: -0.02em;
}
.landing-stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--dim); margin-top: 4px;
}

/* ── PAGE HEADERS ───────────────────────────────────────── */
.page-header-wrap {
    padding: 52px 48px 40px;
    border-bottom: 1px solid var(--border);
    position: relative;
}
.page-header-wrap::after {
    content: ''; position: absolute;
    bottom: -1px; left: 48px;
    width: 48px; height: 1px;
    background: var(--accent);
}
.page-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
}
.page-tag::before { content: ''; width: 18px; height: 1px; background: var(--accent); }
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem; font-weight: 400;
    letter-spacing: -0.02em; color: var(--text);
    margin-bottom: 8px; line-height: 1.1;
}
.page-desc {
    font-size: 0.86rem; color: var(--muted);
    font-weight: 300; font-style: italic;
}

/* Legacy aliases for old class names in templates */
.search-question { font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 400; letter-spacing: -0.02em; color: var(--text); line-height: 1.1; padding: 52px 24px 0; }
.cite-header { font-family: 'Playfair Display', serif; font-size: 2.5rem; font-weight: 400; letter-spacing: -0.02em; color: var(--text); line-height: 1.1; padding: 52px 0 0; }
.cite-subheader { font-size: 0.86rem; color: var(--muted); font-weight: 300; font-style: italic; margin-bottom: 28px; }
.cite-section-label { font-family: 'DM Mono', monospace; font-size: 0.78rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); margin-bottom: 14px; margin-top: 16px; display: block; }

/* ── CITE RESULT ────────────────────────────────────────── */
.cite-result-box {
    background: var(--accent-dim);
    border: 1px solid rgba(196,163,90,0.2);
    border-left: 2px solid var(--accent);
    padding: 24px 28px;
    margin-top: 20px; margin-bottom: 8px;
}
.cite-result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 12px;
}
.cite-result-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem; color: var(--text); line-height: 1.75; font-weight: 300;
}

/* ── FOOTER ─────────────────────────────────────────────── */
.footer {
    border-top: 1px solid var(--border);
    padding: 20px 48px;
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 0;
}
.footer-brand {
    font-family: 'Playfair Display', serif;
    font-size: 0.85rem; color: var(--dim);
}
.footer-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--dim);
}
.footer-fixed {
    position: fixed; bottom: 0; left: 0; right: 0;
    border-top: 1px solid var(--border);
    padding: 12px 48px;
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(11,13,19,0.92);
    backdrop-filter: blur(12px); z-index: 999;
}

/* ── FETCH BUTTON ALIGNMENT ─────────────────────────────── */
.fetch-btn-col { display: flex; align-items: flex-end; padding-bottom: 1px; }
.fetch-btn-col .stButton { width: 100%; }
.fetch-btn-col .stButton > button { width: 100%; height: 42px; }

/* ── FADE ANIMATIONS ────────────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in          { animation: fadeUp 0.5s ease forwards; }
.fade-in-delay-1  { animation: fadeUp 0.5s ease 0.08s both; }
.fade-in-delay-2  { animation: fadeUp 0.5s ease 0.16s both; }
.fade-in-delay-3  { animation: fadeUp 0.5s ease 0.26s both; }
</style>
""", unsafe_allow_html=True)

# ---- HANDLE QUERY PARAM NAVIGATION ----
go_param = st.query_params.get("go")
if go_param in ["home", "search", "cite"]:
    st.session_state.page = go_param
    st.query_params.clear()
    st.rerun()

# ---- TOP NAVBAR ----
page = st.session_state.page
home_class   = "active" if page == "home"   else ""
search_class = "active" if page == "search" else ""
cite_class   = "active" if page == "cite"   else ""

st.markdown(f"""
<div class="top-navbar">
    <a class="navbar-brand" href="?go=home" target="_self">
        <span class="navbar-brand-dot"></span>
        SourceBot
    </a>
    <div class="navbar-links">
        <a href="?go=home"   target="_self" class="{home_class}">Home</a>
        <a href="?go=search" target="_self" class="{search_class}">Quick Search</a>
        <a href="?go=cite"   target="_self" class="{cite_class}">Generate Citations</a>
    </div>
</div>
<div class="navbar-spacer"></div>
""", unsafe_allow_html=True)


# ==============================
# HOME PAGE
# ==============================
if st.session_state.page == "home":

    # Footer injected via CSS ::after to avoid Streamlit HTML sanitization
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]::before {
        content: 'An Ian Patterson Production';
        position: fixed;
        bottom: 0; left: 0; right: 0;
        z-index: 998;
        text-align: center;
        padding: 14px 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #3d4050;
        background: rgba(11,13,19,0.95);
        border-top: 1px solid rgba(255,255,255,0.06);
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

    graph_svg = """<svg viewBox="0 0 480 420" xmlns="http://www.w3.org/2000/svg" style="width:580px;height:520px;opacity:0.82;transform:translateX(40px);display:block;">
<defs><style>
.nr{fill:none;stroke:#c4a35a;stroke-width:1.2}
.nd{fill:#c4a35a}
.ns{fill:rgba(196,163,90,0.45)}
.eg{stroke:rgba(196,163,90,0.18);stroke-width:1;fill:none}
.eb{stroke:rgba(196,163,90,0.38);stroke-width:1.2;fill:none}
@keyframes dI{from{stroke-dashoffset:1}to{stroke-dashoffset:0}}
@keyframes fN{from{opacity:0;transform:scale(0.4)}to{opacity:1;transform:scale(1)}}
.e1{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 1.2s ease 0.2s forwards}
.e2{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 1.0s ease 0.4s forwards}
.e3{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 1.1s ease 0.3s forwards}
.e4{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 0.9s ease 0.6s forwards}
.e5{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 1.0s ease 0.5s forwards}
.e6{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 1.2s ease 0.7s forwards}
.e7{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 0.8s ease 0.8s forwards}
.e8{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 1.1s ease 0.9s forwards}
.e9{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 1.0s ease 1.0s forwards}
.e10{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 0.9s ease 1.1s forwards}
.e11{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 1.0s ease 1.2s forwards}
.e12{stroke-dasharray:1;stroke-dashoffset:1;animation:dI 0.8s ease 1.3s forwards}
.n1{opacity:0;transform-origin:240px 200px;animation:fN 0.5s ease 0.1s forwards}
.n2{opacity:0;transform-origin:140px 120px;animation:fN 0.5s ease 0.3s forwards}
.n3{opacity:0;transform-origin:340px 110px;animation:fN 0.5s ease 0.4s forwards}
.n4{opacity:0;transform-origin:120px 280px;animation:fN 0.5s ease 0.5s forwards}
.n5{opacity:0;transform-origin:360px 290px;animation:fN 0.5s ease 0.6s forwards}
.n6{opacity:0;transform-origin:240px 340px;animation:fN 0.5s ease 0.7s forwards}
.n7{opacity:0;transform-origin:70px 180px;animation:fN 0.5s ease 0.8s forwards}
.n8{opacity:0;transform-origin:410px 190px;animation:fN 0.5s ease 0.9s forwards}
.n9{opacity:0;transform-origin:200px 55px;animation:fN 0.5s ease 1.0s forwards}
.n10{opacity:0;transform-origin:290px 360px;animation:fN 0.5s ease 1.1s forwards}
.n11{opacity:0;transform-origin:170px 360px;animation:fN 0.5s ease 1.2s forwards}
.n12{opacity:0;transform-origin:380px 60px;animation:fN 0.5s ease 1.3s forwards}
</style>
<filter id="gl"><feGaussianBlur stdDeviation="2.5" result="cb"/><feMerge><feMergeNode in="cb"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<line class="eg e1" x1="240" y1="200" x2="140" y2="120" pathLength="1"/>
<line class="eg e2" x1="240" y1="200" x2="340" y2="110" pathLength="1"/>
<line class="eg e3" x1="240" y1="200" x2="120" y2="280" pathLength="1"/>
<line class="eg e4" x1="240" y1="200" x2="360" y2="290" pathLength="1"/>
<line class="eg e5" x1="240" y1="200" x2="240" y2="340" pathLength="1"/>
<line class="eb e6" x1="140" y1="120" x2="340" y2="110" pathLength="1"/>
<line class="eg e7" x1="140" y1="120" x2="70" y2="180" pathLength="1"/>
<line class="eg e8" x1="140" y1="120" x2="200" y2="55" pathLength="1"/>
<line class="eg e9" x1="340" y1="110" x2="410" y2="190" pathLength="1"/>
<line class="eg e10" x1="340" y1="110" x2="380" y2="60" pathLength="1"/>
<line class="eg e11" x1="120" y1="280" x2="240" y2="340" pathLength="1"/>
<line class="eg e12" x1="360" y1="290" x2="240" y2="340" pathLength="1"/>
<line class="eg e7" x1="70" y1="180" x2="120" y2="280" pathLength="1"/>
<line class="eg e9" x1="410" y1="190" x2="360" y2="290" pathLength="1"/>
<line class="eg e12" x1="240" y1="340" x2="290" y2="360" pathLength="1"/>
<line class="eg e11" x1="240" y1="340" x2="170" y2="360" pathLength="1"/>
<g class="n1" filter="url(#gl)"><circle cx="240" cy="200" r="14" class="nr"/><circle cx="240" cy="200" r="5" class="nd"/></g>
<g class="n2" filter="url(#gl)"><circle cx="140" cy="120" r="10" class="nr"/><circle cx="140" cy="120" r="4" class="nd"/></g>
<g class="n3" filter="url(#gl)"><circle cx="340" cy="110" r="10" class="nr"/><circle cx="340" cy="110" r="4" class="nd"/></g>
<g class="n4"><circle cx="120" cy="280" r="8" class="nr"/><circle cx="120" cy="280" r="3" class="nd"/></g>
<g class="n5"><circle cx="360" cy="290" r="8" class="nr"/><circle cx="360" cy="290" r="3" class="nd"/></g>
<g class="n6"><circle cx="240" cy="340" r="9" class="nr"/><circle cx="240" cy="340" r="3.5" class="nd"/></g>
<g class="n7"><circle cx="70" cy="180" r="5" class="nr"/><circle cx="70" cy="180" r="2" class="ns"/></g>
<g class="n8"><circle cx="410" cy="190" r="5" class="nr"/><circle cx="410" cy="190" r="2" class="ns"/></g>
<g class="n9"><circle cx="200" cy="55" r="5" class="nr"/><circle cx="200" cy="55" r="2" class="ns"/></g>
<g class="n10"><circle cx="290" cy="360" r="4" class="nr"/><circle cx="290" cy="360" r="1.5" class="ns"/></g>
<g class="n11"><circle cx="170" cy="360" r="4" class="nr"/><circle cx="170" cy="360" r="1.5" class="ns"/></g>
<g class="n12"><circle cx="380" cy="60" r="4" class="nr"/><circle cx="380" cy="60" r="1.5" class="ns"/></g>
</svg>"""

    st.markdown("""
    <div class="landing-wrapper">
        <div style="flex:1;max-width:520px;position:relative;z-index:1;">
            <div class="landing-eyebrow fade-in">AI Research Assistant</div>
            <h1 class="landing-title fade-in-delay-1">Research with<br><em>precision</em><br>and clarity.</h1>
            <p class="landing-tagline fade-in-delay-2">SourceBot surfaces peer-reviewed literature and generates publication-ready citations — so you can focus on the argument that matters.</p>
            <div class="landing-actions fade-in-delay-2">
                <a href="?go=search" target="_self" class="landing-btn-primary">Search Papers</a>
                <a href="?go=cite" target="_self" class="landing-btn-secondary">Cite a Source</a>
            </div>
            <div class="landing-stat-bar fade-in-delay-3">
                <div class="landing-stat" style="flex:1;text-align:center;border-right:1px solid var(--border);">
                    <div class="landing-stat-num">100,000+</div>
                    <div class="landing-stat-label">Papers Indexed</div>
                </div>
                <div class="landing-stat" style="flex:1;text-align:center;">
                    <div class="landing-stat-num">APA &middot; MLA</div>
                    <div class="landing-stat-label">Citation Formats</div>
                </div>
            </div>
        </div>
        <div style="flex:1;display:flex;justify-content:flex-end;align-items:center;overflow:visible;">
    """ + graph_svg + """
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================
# SEARCH PAGE
# ==============================
elif st.session_state.page == "search":

    hist_col, main_col = st.columns([1, 5])

    with hist_col:
        if st.session_state.search_history:
            history_items_html = "".join([
                f'<div class="history-item">{q}</div>'
                for q in reversed(st.session_state.search_history)
            ])
        else:
            history_items_html = '<div class="history-panel-empty">Your searches will appear here as you explore.</div>'

        st.markdown(f"""
        <div class="history-panel">
            <div class="history-panel-title">Recent Searches</div>
            {history_items_html}
        </div>
        """, unsafe_allow_html=True)

    with main_col:
        st.markdown("""
        <div style="padding: 52px 24px 0;">
            <div class="page-tag fade-in">Quick Search</div>
            <h1 class="landing-title fade-in-delay-1" style="font-size:2.4rem; padding:0; margin-bottom:10px;">What are you looking for?</h1>
            <p style="font-family:'DM Sans',sans-serif; font-size:0.86rem; font-weight:300; color:var(--muted); font-style:italic; margin-bottom:28px;">
                Describe your research argument or question — the more specific, the better.
            </p>
        </div>
        """, unsafe_allow_html=True)

        query = st.text_area(
            label="",
            placeholder="e.g. An argument that raising the minimum wage does not significantly increase unemployment...",
            height=110,
        )

        st.markdown("<div style='margin-top: 1.6rem;'></div>", unsafe_allow_html=True)

        col_btn, _ = st.columns([1.5, 5])
        with col_btn:
            st.markdown('<div class="search-btn">', unsafe_allow_html=True)
            search_clicked = st.button("Search Papers")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        if search_clicked:
            if query:
                if query not in st.session_state.search_history:
                    st.session_state.search_history.append(query)
                st.session_state.current_query = query
                st.rerun()
            else:
                st.warning("Please enter a search query.")

        if st.session_state.current_query:
            active_query = st.session_state.current_query

            with st.spinner("Searching through sources..."):
                results = search_papers(active_query, 5)

            st.markdown("<br>", unsafe_allow_html=True)

            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                abstract = results["documents"][0][i]

                analysis = analyze_paper(abstract, active_query)
                relevance, quotes = parse_analysis(analysis)

                doi = metadata.get("doi", "")
                link = f"https://doi.org/{doi.replace('https://doi.org/', '')}" if doi else None
                authors = metadata.get("authors", "Unknown")
                year = metadata.get("year", "")
                title = metadata.get("title", "Untitled")

                apa_citation = format_apa(title, authors, year, doi)

                quotes_html = "".join([f'<div class="result-quote">"{q}"</div>' for q in quotes])
                link_html = (
                    f'<div class="result-link"><a href="{link}" target="_blank">↗ Read the full paper</a></div>'
                    if link else
                    '<div class="result-meta" style="margin-top:8px;">No link available</div>'
                )

                st.markdown(f"""
                <div class="result-card fade-in">
                    <div class="result-title">{title}</div>
                    <div class="result-meta">{authors}&nbsp;&nbsp;·&nbsp;&nbsp;{year}</div>
                    <div class="result-section-label">Why it&#39;s relevant</div>
                    <div class="result-relevance">{relevance}</div>
                    <div class="result-section-label">Relevant quotes</div>
                    {quotes_html}
                    <div class="result-section-label" style="margin-top:18px;">Source</div>
                    {link_html}
                    <div class="result-section-label" style="margin-top:18px;">APA Citation</div>
                    <div style="
                        background: rgba(0,0,0,0.22);
                        border-top: 1px solid var(--border);
                        margin: 10px -32px -22px -32px;
                        padding: 16px 32px 16px;
                        position: relative;
                    ">
                        <div id="cit-{i}" style="
                            font-family: 'DM Mono', monospace;
                            font-size: 0.78rem;
                            color: var(--muted);
                            font-style: italic;
                            line-height: 1.65;
                            word-break: break-word;
                            padding-right: 100px;
                        ">{apa_citation}</div>
                        <button id="btn-{i}" onclick="
                            var txt = document.getElementById('cit-{i}').innerText;
                            navigator.clipboard.writeText(txt).then(function() {{
                                var b = document.getElementById('btn-{i}');
                                b.innerText = '✓ Copied';
                                b.style.color = '#c4a35a';
                                b.style.borderColor = '#c4a35a';
                                setTimeout(function() {{
                                    b.innerText = 'Copy';
                                    b.style.color = '';
                                    b.style.borderColor = '';
                                }}, 2000);
                            }});
                        " style="
                            position: absolute;
                            top: 16px; right: 32px;
                            background: transparent;
                            border: 1px solid rgba(255,255,255,0.11);
                            color: #3d4050;
                            padding: 4px 12px;
                            font-family: 'DM Mono', monospace;
                            font-size: 0.6rem;
                            letter-spacing: 0.1em;
                            text-transform: uppercase;
                            cursor: pointer;
                            transition: all 0.2s;
                        ">Copy</button>
                    </div>
                </div>
                <div style="height: 28px;"></div>
                """, unsafe_allow_html=True)

    try:
        paper_count = qdrant.count(collection_name=COLLECTION_NAME).count
        count_str = f"{paper_count:,} papers indexed"
    except:
        count_str = ""

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]::before {
        content: 'An Ian Patterson Production';
        position: fixed;
        bottom: 0; left: 0; right: 0;
        z-index: 998;
        text-align: center;
        padding: 14px 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #3d4050;
        background: rgba(11,13,19,0.95);
        border-top: 1px solid rgba(255,255,255,0.06);
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================
# CITE PAGE
# ==============================
elif st.session_state.page == "cite":

    _, main_col, _ = st.columns([0.5, 6, 0.5])
    pf = st.session_state.cite_prefill

    with main_col:
        st.markdown("""
        <div class="cite-header fade-in">Generate Citations</div>
        <div class="cite-subheader">Paste a link or DOI to auto-fill, or enter details manually.</div>
        """, unsafe_allow_html=True)

        # ---- AUTO-FETCH SECTION ----
        st.markdown('<div class="cite-section-label">Auto-Fill from URL or DOI</div>', unsafe_allow_html=True)
        fetch_col, btn_fetch_col = st.columns([5, 1])
        with fetch_col:
            fetch_input = st.text_input(
                label="",
                placeholder="e.g. https://doi.org/10.1257/aer.20180975  or  https://www.brookings.edu/articles/...",
                key="fetch_input_field"
            )
        with btn_fetch_col:
            # Invisible spacer that exactly matches Streamlit's empty label height
            st.markdown('<div style="height:46px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="cite-btn">', unsafe_allow_html=True)
            fetch_clicked = st.button("Fetch")
            st.markdown('</div>', unsafe_allow_html=True)

        if fetch_clicked:
            if fetch_input.strip():
                with st.spinner("Fetching metadata..."):
                    metadata, error = detect_and_fetch(fetch_input.strip())
                if error:
                    st.session_state.cite_fetch_error = error
                    st.session_state.cite_fetch_success = False
                    st.session_state.cite_prefill = {}
                else:
                    st.session_state.cite_prefill = metadata
                    st.session_state.cite_fetch_error = None
                    st.session_state.cite_fetch_success = True
                    st.session_state.cite_generated = False
                    st.session_state.cite_result = None
                st.rerun()
            else:
                st.warning("Please enter a URL or DOI to fetch.")

        if st.session_state.cite_fetch_error:
            st.error(st.session_state.cite_fetch_error)

        if st.session_state.cite_fetch_success:
            st.success("✓ Metadata fetched — fields have been pre-filled below. Review and adjust as needed.")

        st.markdown("<div style='margin-top:0.8rem; border-top:1px solid var(--border); padding-top:1.2rem;'></div>", unsafe_allow_html=True)

        # ---- FORMAT + SOURCE TYPE ----
        col_fmt, col_src = st.columns(2)
        with col_fmt:
            cite_format = st.selectbox("Citation Format", ["APA", "MLA"])

        prefill_source = pf.get("source_type", "journal")
        default_src_idx = 1 if prefill_source == "website" else 0

        with col_src:
            if cite_format == "MLA":
                source_type = st.selectbox("Source Type", ["Journal Article", "Website"], index=default_src_idx)
            else:
                source_type = "Journal Article"
                st.selectbox("Source Type", ["Journal Article"], disabled=True)

        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

        # ---- SHARED FIELDS ----
        st.markdown('<div class="cite-section-label">Source Details</div>', unsafe_allow_html=True)

        title_input   = st.text_input("Title",   value=pf.get("title", ""),   placeholder="e.g. The Effects of Minimum Wage on Employment")
        authors_input = st.text_input("Authors", value=pf.get("authors", ""), placeholder="e.g. John Smith, Jane Doe  (comma-separated, First Last format)")

        # ---- CONDITIONAL FIELDS ----
        if source_type == "Journal Article":
            col1, col2 = st.columns(2)
            with col1:
                year_input = st.text_input("Year", value=pf.get("year", ""), placeholder="e.g. 2021")
            with col2:
                doi_input = st.text_input("DOI", value=pf.get("doi", ""), placeholder="e.g. 10.1000/xyz123")

            if cite_format == "MLA":
                st.markdown('<div class="cite-section-label">Journal Details</div>', unsafe_allow_html=True)
                journal_input = st.text_input("Journal Name", value=pf.get("journal", ""), placeholder="e.g. Journal of Economic Perspectives")
                col3, col4, col5 = st.columns(3)
                with col3:
                    volume_input = st.text_input("Volume", value=pf.get("volume", ""), placeholder="e.g. 35")
                with col4:
                    issue_input  = st.text_input("Issue",  value=pf.get("issue", ""),  placeholder="e.g. 2")
                with col5:
                    pages_input  = st.text_input("Pages",  value=pf.get("pages", ""),  placeholder="e.g. 45–67")
            else:
                journal_input = volume_input = issue_input = pages_input = ""

            site_name_input = url_input = access_date_input = publish_date_input = ""

        elif source_type == "Website":
            col1, col2 = st.columns(2)
            with col1:
                publish_date_input = st.text_input("Publish Date", value=pf.get("publish_date", ""), placeholder="e.g. 15 Mar. 2022")
            with col2:
                access_date_input = st.text_input("Access Date", placeholder="e.g. 1 Feb. 2025")

            site_name_input = st.text_input("Website/Organization Name", value=pf.get("site_name", ""), placeholder="e.g. The Brookings Institution")
            url_input       = st.text_input("URL", value=pf.get("url", ""), placeholder="e.g. https://www.brookings.edu/articles/...")

            year_input = doi_input = journal_input = volume_input = issue_input = pages_input = ""

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        btn_col1, btn_col2, _ = st.columns([1.4, 1, 4])
        with btn_col1:
            st.markdown('<div class="cite-btn">', unsafe_allow_html=True)
            generate_clicked = st.button("Generate Citation")
            st.markdown('</div>', unsafe_allow_html=True)
        with btn_col2:
            st.markdown('<div class="cite-clear-btn">', unsafe_allow_html=True)
            clear_clicked = st.button("Clear")
            st.markdown('</div>', unsafe_allow_html=True)

        if clear_clicked:
            st.session_state.cite_result = None
            st.session_state.cite_generated = False
            st.session_state.cite_prefill = {}
            st.session_state.cite_fetch_error = None
            st.session_state.cite_fetch_success = False
            st.rerun()

        if generate_clicked:
            if not title_input.strip():
                st.warning("Please enter a title to generate a citation.")
            else:
                if cite_format == "APA":
                    result = format_apa(title_input, authors_input, year_input, doi_input)
                elif cite_format == "MLA" and source_type == "Journal Article":
                    result = format_mla_journal(
                        title_input, authors_input, year_input,
                        journal_input, volume_input, issue_input, pages_input, doi_input
                    )
                elif cite_format == "MLA" and source_type == "Website":
                    result = format_mla_website(
                        title_input, authors_input, site_name_input,
                        url_input, access_date_input, publish_date_input
                    )
                else:
                    result = ""

                st.session_state.cite_result = result
                st.session_state.cite_generated = True
                st.rerun()

        # ---- RESULT ----
        if st.session_state.cite_generated and st.session_state.cite_result:
            st.markdown(f"""
            <div class="cite-result-box fade-in">
                <div class="cite-result-label">{cite_format} Citation</div>
                <div class="cite-result-text">{st.session_state.cite_result}</div>
            </div>
            """, unsafe_allow_html=True)

            escaped = st.session_state.cite_result.replace("\\", "\\\\").replace("`", "\\`")
            st.markdown(f"""
            <div style="margin-top:0.7rem;">
                <button onclick="navigator.clipboard.writeText(`{escaped}`).then(() => {{
                    this.innerText = '✓ Copied';
                    this.style.color = 'var(--accent)';
                    this.style.borderColor = 'var(--accent)';
                    setTimeout(() => {{
                        this.innerText = 'Copy to Clipboard';
                        this.style.color = '';
                        this.style.borderColor = '';
                    }}, 2000);
                }})" style="
                    background: transparent;
                    border: 1px solid var(--border2);
                    color: var(--muted);
                    padding: 7px 18px;
                    font-family: 'DM Mono', monospace;
                    font-size: 0.62rem;
                    letter-spacing: 0.1em;
                    text-transform: uppercase;
                    cursor: pointer;
                    transition: all 0.2s;
                ">Copy to Clipboard</button>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]::before {
        content: 'An Ian Patterson Production';
        position: fixed;
        bottom: 0; left: 0; right: 0;
        z-index: 998;
        text-align: center;
        padding: 14px 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #3d4050;
        background: rgba(11,13,19,0.95);
        border-top: 1px solid rgba(255,255,255,0.06);
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)