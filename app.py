import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
import streamlit as st
import base64
load_dotenv()

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

# ---- SESSION STATE ----
if "page" not in st.session_state:
    st.session_state.page = "home"
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = None

# ---- PAGE CONFIG ----
st.set_page_config(page_title="SourceBot", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

# ---- CUSTOM CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #ffffff;
}

.main { background-color: #0a0a0f; padding-top: 0 !important; }

[data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(75, 45, 127, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(75, 45, 127, 0.1) 0%, transparent 50%);
}

[data-testid="stHeader"] { background-color: #0a0a0f; height: 0 !important; }

[data-testid="block-container"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

[data-testid="stVerticalBlock"] { gap: 0 !important; }

.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

section.main > div { padding-top: 0 !important; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(1px 1px at 10% 15%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 25% 40%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 40% 10%, rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 60%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 25%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 80% 70%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 45%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 15% 75%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 35% 85%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 60% 90%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 5%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 5% 50%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 45% 30%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 88% 15%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 20% 95%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 65% 55%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 20%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 50% 75%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 95% 35%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 8% 88%, rgba(255,255,255,0.4) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ---- HISTORY PANEL ---- */
.history-panel {
    background-color: #0d0d14;
    border-right: 1px solid rgba(255,255,255,0.06);
    min-height: 100vh;
    padding: 1.5rem 0.8rem;
    box-sizing: border-box;
}

.history-panel-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #a78bde;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.history-panel-empty {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.2);
    font-style: italic;
}

.history-item {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.45);
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    cursor: pointer;
    line-height: 1.4;
    word-break: break-word;
}

.history-item:hover { color: rgba(255,255,255,0.9); }

/* ---- HOME BUTTON top-right ---- */
.home-btn-container {
    position: fixed;
    top: 1rem;
    right: 1.5rem;
    z-index: 1000;
}

.home-btn-container a {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    text-decoration: none;
    background-color: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 0.6rem 1rem;
}

.home-btn-container a:hover { color: rgba(255,255,255,0.9); }

/* ---- SEARCH BUTTON ---- */
.search-btn .stButton > button {
    background-color: #ffffff !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 50px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 2rem !important;
    width: auto !important;
}

.search-btn .stButton > button:hover {
    background-color: #e0d8f0 !important;
    transform: scale(1.02) !important;
}

/* ---- LANDING PAGE ---- */
.landing-wrapper {
    height: 75vh;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    padding: 0 4rem;
    box-sizing: border-box;
}

.landing-title {
    font-size: 5rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.0;
    margin-bottom: 0.3rem;
    letter-spacing: -2px;
}

.landing-tagline {
    font-size: 1rem;
    font-weight: 300;
    color: rgba(255,255,255,0.6);
    font-style: italic;
    margin-bottom: 2rem;
    letter-spacing: 0.3px;
}

/* ---- SEARCH PAGE ---- */
.search-question {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    letter-spacing: -1px;
    padding: 2.5rem 2rem 0 1rem;
}

/* ---- TEXT AREA ---- */
.stTextArea textarea {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    background-color: rgba(255,255,255,0.07);
    color: #ffffff;
    padding: 1rem;
}

.stTextArea textarea::placeholder { color: rgba(255,255,255,0.35) !important; opacity: 1 !important; }

.stTextArea textarea:focus {
    border-color: rgba(255,255,255,0.5) !important;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.2) !important;
    outline: none !important;
    background-color: rgba(255,255,255,0.1) !important;
}

/* ---- SLIDER ---- */
[data-testid="stSlider"] label { color: rgba(255,255,255,0.7) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #ffffff !important;
    border-color: #ffffff !important;
}

/* ---- RESULT CARDS ---- */
.result-card {
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-left: 4px solid #7B5AB0;
    border-radius: 8px 8px 0 0;
    padding: 1.5rem;
    margin-bottom: 0;
}

[data-testid="stCode"] {
    margin-top: 0 !important;
    margin-bottom: 1.2rem !important;
    border-radius: 0 0 8px 8px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-top: none !important;
    border-left: 4px solid #7B5AB0 !important;
}

[data-testid="stCode"] code {
    font-size: 0.82rem !important;
    color: rgba(255,255,255,0.6) !important;
    font-style: italic !important;
}

.result-title { font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 0.3rem; }
.result-meta { font-size: 0.82rem; color: rgba(255,255,255,0.4); font-style: italic; margin-bottom: 1rem; }

.result-section-label {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #a78bde;
    margin-bottom: 0.4rem;
    margin-top: 1rem;
}

.result-relevance { font-size: 0.93rem; line-height: 1.65; color: rgba(255,255,255,0.8); }

.result-quote {
    font-style: italic;
    color: rgba(255,255,255,0.6);
    border-left: 2px solid rgba(167,139,222,0.5);
    padding-left: 0.8rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.5;
}

.result-link a { color: rgba(255,255,255,0.6); text-decoration: none; font-size: 0.88rem; font-weight: 600; }
.result-link a:hover { color: #ffffff; }

/* ---- FOOTER ---- */
.footer {
    text-align: center;
    padding: 1rem 0;
    margin-top: 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    font-size: 0.78rem;
    color: rgba(255,255,255,0.25);
    letter-spacing: 0.5px;
}

.footer-fixed {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    text-align: center;
    padding: 1rem 0;
    border-top: 1px solid rgba(255,255,255,0.08);
    font-size: 0.78rem;
    color: rgba(255,255,255,0.25);
    letter-spacing: 0.5px;
    background-color: #0a0a0f;
    z-index: 999;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn 0.6s ease forwards; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
function fixSlider() {
    const tracks = document.querySelectorAll('[data-baseweb="slider"] div');
    tracks.forEach(el => {
        const bg = window.getComputedStyle(el).backgroundColor;
        if (bg === 'rgb(255, 75, 75)' || bg === 'rgb(255, 99, 71)' || bg === 'rgb(255, 76, 76)') {
            el.style.backgroundColor = '#ffffff';
        }
    });
}
setInterval(fixSlider, 300);
</script>
""", unsafe_allow_html=True)

# ==============================
# HOME PAGE
# ==============================
if st.session_state.page == "home":

    if st.query_params.get("go") == "search":
        st.session_state.page = "search"
        st.query_params.clear()
        st.rerun()

    with open("robot.png", "rb") as f:
        robot_data = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div class="landing-wrapper fade-in">
        <div style="flex: 1;">
            <div class="landing-title">SourceBot</div>
            <div class="landing-tagline">Search for what you <em>need</em>.</div>
            <div style="margin-top: 1.8rem;">
                <a href="?go=search" target="_self" style="
                    display: inline-block;
                    background-color: #ffffff;
                    color: #0a0a0f;
                    border-radius: 50px;
                    font-family: 'Inter', sans-serif;
                    font-size: 0.95rem;
                    font-weight: 600;
                    padding: 0.7rem 2rem;
                    text-decoration: none;
                ">Get Started</a>
            </div>
        </div>
        <div style="flex: 1; display: flex; justify-content: flex-end; align-items: center; padding-right: 2rem;">
            <img src="data:image/png;base64,{robot_data}" style="height: 420px; object-fit: contain; filter: drop-shadow(0 0 40px rgba(123, 90, 176, 0.4));">
        </div>
    </div>
    <div class="footer">
        <div style="font-size: 0.7rem; color: rgba(255,255,255,0.2); margin-bottom: 0.3rem; letter-spacing: 1px; text-transform: uppercase;">Search from over {qdrant.count(collection_name=COLLECTION_NAME).count:,} papers</div>
        An Ian Patterson Production
    </div>
    """, unsafe_allow_html=True)

# ==============================
# SEARCH PAGE
# ==============================
elif st.session_state.page == "search":

    # Fixed home button top-right
    st.markdown("""
    <div class="home-btn-container">
        <a href="?go=home" target="_self">Home</a>
    </div>
    """, unsafe_allow_html=True)

    # Handle home navigation
    if st.query_params.get("go") == "home":
        st.session_state.page = "home"
        st.query_params.clear()
        st.rerun()

    # Two column layout
    hist_col, main_col = st.columns([1, 5])

    with hist_col:
        if st.session_state.search_history:
            history_items_html = "".join([
                f'<div class="history-item">• {q}</div>'
                for q in reversed(st.session_state.search_history)
            ])
        else:
            history_items_html = '<div class="history-panel-empty">Your searches will appear here.</div>'

        st.markdown(f"""
        <div class="history-panel">
            <div class="history-panel-title">Search History</div>
            {history_items_html}
        </div>
        """, unsafe_allow_html=True)

    with main_col:

        st.markdown("""
        <div class="search-question fade-in">What are you looking for today?</div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)

        query = st.text_area(
            label="",
            placeholder="e.g. An argument that raising the minimum wage does not significantly increase unemployment...",
            height=100,
        )

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="search-btn">', unsafe_allow_html=True)
        search_clicked = st.button("Search")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        # When search is clicked, save query and rerun so history updates first
        if search_clicked:
            if query:
                if query not in st.session_state.search_history:
                    st.session_state.search_history.append(query)
                st.session_state.current_query = query
                st.rerun()
            else:
                st.warning("Please enter a search query.")

        # After rerun, run the actual search using the saved query
        if st.session_state.current_query:
            active_query = st.session_state.current_query

            with st.spinner("Searching through sources..."):
                results = search_papers(active_query, n_results)

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
                apa_safe = apa_citation.replace("`", "").replace("\\", "")

                quotes_html = "".join([f'<div class="result-quote">"{q}"</div>' for q in quotes])
                link_html = f'<div class="result-link"><a href="{link}" target="_blank">→ Read the full paper</a></div>' if link else '<div class="result-meta">No link available</div>'

                st.markdown(f"""
                <div class="result-card fade-in">
                    <div class="result-title">{title}</div>
                    <div class="result-meta">{authors} &nbsp;·&nbsp; {year}</div>
                    <div class="result-section-label">Why it's relevant</div>
                    <div class="result-relevance">{relevance}</div>
                    <div class="result-section-label">Relevant quotes</div>
                    {quotes_html}
                    <div class="result-section-label" style="margin-top:1rem;">Source</div>
                    {link_html}
                    <div class="result-section-label" style="margin-top:1rem;">APA Citation</div>
                </div>
                """, unsafe_allow_html=True)
                st.code(apa_safe, language=None)

    paper_count = qdrant.count(collection_name=COLLECTION_NAME).count
    st.markdown(f"""
    <div class="footer-fixed">
        <div style="font-size: 0.7rem; color: rgba(255,255,255,0.2); margin-bottom: 0.3rem; letter-spacing: 1px; text-transform: uppercase;">Search from over {paper_count:,} papers</div>
        An Ian Patterson Production
    </div>
    """, unsafe_allow_html=True)