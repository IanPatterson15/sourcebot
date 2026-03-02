import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

TOPICS = [
    # ---- ECONOMICS & BUSINESS (expanded) ----
    "inflation monetary policy",
    "fiscal policy government spending",
    "economic growth determinants",
    "unemployment labor market",
    "central banking Federal Reserve",
    "quantitative easing unconventional monetary policy",
    "government debt sustainability",
    "recession recovery economics",
    "behavioral economics nudge",
    "prospect theory loss aversion",
    "bounded rationality decision making",
    "market structure competition antitrust",
    "game theory strategic interaction",
    "information asymmetry adverse selection",
    "moral hazard incentives",
    "principal agent problem",
    "auction theory mechanism design",
    "price discrimination markets",
    "network effects platform economics",
    "two sided markets platforms",
    "gig economy labor platforms",
    "corporate finance capital structure",
    "dividend policy firms",
    "mergers acquisitions corporate",
    "venture capital startup financing",
    "private equity leveraged buyouts",
    "initial public offerings IPO",
    "asset pricing models",
    "stock market anomalies",
    "financial crisis contagion",
    "banking regulation systemic risk",
    "shadow banking financial intermediation",
    "cryptocurrency blockchain economics",
    "fintech digital finance",
    "ESG sustainable investing",
    "executive compensation incentives",
    "corporate governance board",
    "shareholder activism",
    "international trade comparative advantage",
    "trade policy tariffs protectionism",
    "foreign direct investment multinationals",
    "global value chains supply chains",
    "exchange rate macroeconomics",
    "balance of payments current account",
    "economic development poverty trap",
    "foreign aid effectiveness development",
    "microfinance poverty reduction",
    "institutional quality corruption development",
    "human capital returns education",
    "income inequality redistribution",
    "intergenerational mobility poverty",
    "minimum wage employment effects",
    "immigration labor market effects",
    "housing market affordability policy",
    "healthcare economics insurance",
    "pension retirement economics",
    "environmental economics carbon tax",
    "climate change economic impact",
    "energy economics transition",
    "innovation diffusion technology",
    "artificial intelligence productivity economics",
    "automation future of work",
    "entrepreneurship firm creation",
    "small business economics",
    "industrial policy government intervention",
    "public goods provision",
    "externalities market failure",
    "auction procurement public sector",
    "tax evasion compliance economics",
    "political economy redistribution",
    "rent seeking lobbying economics",
    "US economy macroeconomics history",
    "Canadian economy policy",
    "Mexican economy development trade",
    "North American integration USMCA",
    "emerging markets capital flows",
    "China economy development",
    "European economy monetary union",
    "global imbalances savings investment",
    "secular stagnation productivity",
    "demographic change aging economy",
    "urbanization economic development",
    "rural development agriculture economics",
    "food security economics",
    "water economics resource",
    "transportation economics infrastructure",
    "digital economy e-commerce",
    "platform economy sharing",
    "data privacy economics regulation",
    "media economics advertising",
    "sports economics",
    "cultural economics arts",
    "real estate economics housing",
    "mortgage market securitization",
    "insurance economics risk",
    "commodities futures markets",
    "sovereign debt crisis",
    "currency crisis exchange rate",
    "financial inclusion microfinance",
    "remittances migration economics",
    "tourism economics development",
    "pharmaceutical economics drug pricing",
    "education economics school choice",
    "crime economics deterrence",
    "military spending defense economics",

    # ---- SOCIAL SCIENCES (expanded) ----
    "social inequality stratification",
    "racial discrimination labor market",
    "gender wage gap discrimination",
    "social mobility intergenerational",
    "poverty measurement welfare",
    "social capital trust networks",
    "organizational behavior management",
    "leadership effectiveness management",
    "team performance collaboration",
    "workplace motivation productivity",
    "burnout stress workplace",
    "diversity inclusion workplace",
    "consumer behavior marketing",
    "brand loyalty marketing",
    "advertising effectiveness consumer",
    "social media consumer behavior",
    "public opinion political behavior",
    "voting behavior elections",
    "political polarization ideology",
    "democracy institutions governance",
    "corruption public sector",
    "civil society NGO development",
    "immigration policy integration",
    "refugee displacement economics",
    "aging population social policy",
    "family policy child care",
    "education policy reform",
    "health policy reform",
    "mental health economics policy",
    "drug policy substance abuse",
    "criminal justice reform",
    "urban policy housing",
    "rural community development",
    "environmental justice inequality",
    "climate change adaptation policy",
    "social movements activism",
    "trust institutions social",
    "religion economics society",
    "media influence public opinion",
    "misinformation disinformation",
    "social norms behavior economics",
    "identity politics economics",
    "populism political economy",
    "nationalism globalization politics",
    "international development aid",
    "conflict war economics",
    "peace building post conflict",
    "human rights economics",

    # ---- NATURAL SCIENCES ----
    # Public Health
    "public health epidemiology",
    "infectious disease epidemiology",
    "COVID-19 pandemic health economics",
    "mental health interventions",
    "obesity health economics",
    "cancer screening prevention economics",
    "vaccination policy public health",
    "health behavior intervention",
    "nutrition health outcomes",
    "alcohol tobacco health economics",
    "opioid crisis substance abuse",
    "global health inequality",
    "maternal child health developing countries",
    "aging health outcomes",
    "disability economics health",

    # Environmental Science
    "climate change mitigation policy",
    "carbon emissions reduction",
    "renewable energy transition economics",
    "biodiversity conservation economics",
    "deforestation land use economics",
    "ocean pollution economics",
    "air quality health economics",
    "water scarcity economics",
    "sustainable agriculture food",
    "circular economy waste",
    "green technology innovation",
    "natural disaster economics",
    "environmental regulation compliance",
    "ecosystem services valuation",
    "urban environment sustainability",

    # Neuroscience and Psychology
    "neuroscience decision making",
    "cognitive psychology memory learning",
    "behavioral psychology intervention",
    "child development psychology",
    "adolescent psychology behavior",
    "aging cognitive decline",
    "mental health treatment outcomes",
    "addiction neuroscience behavior",
    "stress cortisol health",
    "sleep health productivity",
    "exercise mental health",
    "nutrition brain cognitive function",
    "social neuroscience empathy",
    "emotion regulation psychology",
    "mindfulness meditation psychology",

    # Biology and Medicine
    "genetics health outcomes economics",
    "epigenetics environment health",
    "microbiome health economics",
    "infectious disease economics",
    "antibiotic resistance economics",
    "telemedicine digital health",
    "medical technology innovation",
    "clinical trial methodology",
    "evidence based medicine policy",
    "precision medicine economics",

    # Physics and complexity
    "complexity economics systems",
    "network science economics",
    "agent based modeling economics",
    "econophysics financial markets",
    "statistical mechanics social systems",
]

PAPERS_PER_TOPIC = 200
BATCH_SIZE = 50
CHECKPOINT_FILE = "master_checkpoint.json"
NEW_PAPERS_FILE = "new_papers.json"
MERGED_FILE = "merged_papers.json"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed_topics": [], "papers": {}}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    positions = {}
    for word, locs in inverted_index.items():
        for pos in locs:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions.keys()))


def fetch_with_backoff(url, params, max_retries=6):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                wait = (2 ** retries) * 10
                print(f"  Rate limited. Waiting {wait}s before retry {retries + 1}/{max_retries}...")
                time.sleep(wait)
                retries += 1
                continue
            if response.status_code != 200:
                print(f"  Unexpected status {response.status_code}. Retrying in 10s...")
                time.sleep(10)
                retries += 1
                continue
            return response.json()
        except Exception as e:
            print(f"  Error: {e}. Retrying in 10s...")
            time.sleep(10)
            retries += 1
    print(f"  Giving up after {max_retries} retries.")
    return None


def fetch_topic_paginated(query, max_results=PAPERS_PER_TOPIC):
    print(f"  Fetching: {query}")
    url = "https://api.openalex.org/works"
    all_papers = {}
    cursor = "*"

    while len(all_papers) < max_results:
        params = {
            "search": query,
            "filter": "open_access.is_oa:true",
            "per-page": BATCH_SIZE,
            "cursor": cursor,
            "select": "id,title,abstract_inverted_index,authorships,publication_year,doi"
        }

        data = fetch_with_backoff(url, params)
        if data is None:
            break

        results = data.get("results", [])
        if not results:
            break

        for work in results:
            abstract = reconstruct_abstract(work.get("abstract_inverted_index", {}))
            if not abstract:
                continue

            authors = [a["author"]["display_name"] for a in work.get("authorships", [])[:3]]
            paper = {
                "id": work.get("id"),
                "title": work.get("title"),
                "abstract": abstract,
                "authors": authors,
                "year": work.get("publication_year"),
                "doi": work.get("doi")
            }
            all_papers[paper["id"]] = paper

        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break

        cursor = next_cursor
        time.sleep(3)

    print(f"  Got {len(all_papers)} papers")
    return list(all_papers.values())


def merge_with_existing(new_papers):
    existing = {}
    if os.path.exists(MERGED_FILE):
        print(f"Loading existing merged database...")
        with open(MERGED_FILE, "r") as f:
            existing_list = json.load(f)
        for p in existing_list:
            existing[p["id"]] = p
        print(f"  Existing papers: {len(existing)}")

    added = 0
    for paper in new_papers:
        if paper["id"] not in existing:
            existing[paper["id"]] = paper
            added += 1

    merged = list(existing.values())
    with open(MERGED_FILE, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"  Added {added} new unique papers")
    print(f"  Total in merged database: {len(merged)}")
    return merged


if __name__ == "__main__":
    checkpoint = load_checkpoint()
    completed_topics = set(checkpoint["completed_topics"])
    session_papers = checkpoint["papers"]

    print(f"Master Fetch Script")
    print(f"Total topics: {len(TOPICS)}")
    print(f"Already completed: {len(completed_topics)}")
    print(f"Papers this session so far: {len(session_papers)}\n")

    for i, topic in enumerate(TOPICS):
        if topic in completed_topics:
            print(f"Skipping: {topic}")
            continue

        print(f"[{i+1}/{len(TOPICS)}]")
        papers = fetch_topic_paginated(topic)

        for paper in papers:
            session_papers[paper["id"]] = paper

        checkpoint["completed_topics"].append(topic)
        checkpoint["papers"] = session_papers
        save_checkpoint(checkpoint)

        print(f"  Session total: {len(session_papers)}\n")
        time.sleep(5)

    print(f"\nFetch complete! Session papers: {len(session_papers)}")

    new_papers_list = list(session_papers.values())
    with open(NEW_PAPERS_FILE, "w") as f:
        json.dump(new_papers_list, f, indent=2)
    print(f"Saved to {NEW_PAPERS_FILE}")

    print(f"\nMerging with existing database...")
    merge_with_existing(new_papers_list)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    print("\nDone! Now run embed_new_papers.py to embed the new papers.")