import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

TOPICS = [
    # Finance (60 topics)
    "corporate finance",
    "stock market returns",
    "asset pricing models",
    "financial crisis banking",
    "investment strategies",
    "portfolio management",
    "venture capital startups",
    "private equity leveraged buyouts",
    "mergers and acquisitions",
    "corporate governance",
    "dividend policy firms",
    "capital structure debt equity",
    "initial public offerings IPO",
    "hedge funds performance",
    "mutual funds performance",
    "financial derivatives options",
    "futures markets commodities",
    "cryptocurrency bitcoin blockchain",
    "banking regulation Basel",
    "shadow banking financial intermediation",
    "credit risk default",
    "interest rate risk bonds",
    "foreign exchange markets",
    "financial markets volatility",
    "stock market bubbles crashes",
    "behavioral finance investor bias",
    "market efficiency hypothesis",
    "insider trading securities",
    "financial fraud accounting",
    "corporate social responsibility finance",
    "ESG investing sustainable finance",
    "real estate finance housing investment",
    "mortgage markets securitization",
    "pension funds retirement finance",
    "insurance economics risk",
    "microfinance developing countries",
    "fintech digital payments",
    "central bank digital currency",
    "quantitative finance algorithmic trading",
    "firm valuation discounted cash flow",
    "earnings management accounting",
    "financial reporting transparency",
    "auditing corporate accountability",
    "executive compensation CEO pay",
    "shareholder activism",
    "board of directors governance",
    "family firms ownership structure",
    "small business finance",
    "crowdfunding entrepreneurship finance",
    "bank lending credit markets",
    "systemic risk financial stability",
    "too big to fail banks",
    "financial regulation policy",
    "sovereign debt crisis",
    "bond markets yield curve",
    "equity premium puzzle",
    "factor investing value momentum",
    "high frequency trading market microstructure",
    "financial contagion spillovers",
    "capital flows emerging markets finance",

    # Macroeconomics (40 topics)
    "inflation causes consequences",
    "monetary policy interest rates",
    "fiscal policy government spending",
    "business cycles economic fluctuations",
    "economic growth determinants",
    "unemployment causes solutions",
    "central banking Federal Reserve",
    "quantitative easing unconventional monetary policy",
    "government debt sustainability",
    "recession recovery economics",
    "GDP measurement national accounts",
    "stagflation inflation unemployment",
    "deflation liquidity trap",
    "exchange rate macroeconomics",
    "balance of payments current account",
    "macroeconomic forecasting",
    "DSGE models macroeconomics",
    "new Keynesian economics",
    "austerity fiscal consolidation",
    "stimulus spending multiplier",
    "inflation targeting central banks",
    "money supply velocity inflation",
    "housing market macroeconomics",
    "commodity prices macroeconomics",
    "oil prices economic impact",
    "financial accelerator credit cycles",
    "global imbalances savings glut",
    "secular stagnation low growth",
    "productivity slowdown economics",
    "demographic change aging economy",
    "COVID-19 economic impact recovery",
    "supply chain disruption inflation",
    "energy economics macroeconomics",
    "climate change macroeconomic impact",
    "inequality macroeconomic effects",
    "US Federal Reserve policy history",
    "Canadian monetary policy Bank of Canada",
    "North American economic integration",
    "Mexico macroeconomic policy",
    "post pandemic inflation economics",

    # Microeconomics and Business Strategy (30 topics)
    "market power monopoly antitrust",
    "oligopoly firm competition strategy",
    "game theory strategic interaction",
    "pricing strategy firms",
    "price discrimination markets",
    "network effects platform economics",
    "two sided markets platforms",
    "gig economy labor platforms",
    "Amazon Google platform competition",
    "innovation economics research development",
    "patent protection intellectual property",
    "entrepreneurship firm creation",
    "firm productivity efficiency",
    "supply chain management economics",
    "vertical integration outsourcing",
    "franchise business model",
    "retail economics consumer markets",
    "advertising economics markets",
    "information economics signaling",
    "auction theory mechanism design",
    "public choice economics government",
    "regulation economics utilities",
    "health economics hospital competition",
    "education economics returns schooling",
    "labor economics firm wages",
    "search and matching labor markets",
    "industrial organization economics",
    "entry barriers market structure",
    "disruptive innovation technology economics",
    "artificial intelligence economics firms",

    # Trade and Globalization (25 topics)
    "international trade patterns",
    "trade policy tariffs effects",
    "NAFTA USMCA trade agreement",
    "US China trade war",
    "comparative advantage specialization",
    "trade deficits surpluses",
    "globalization labor markets",
    "offshoring outsourcing jobs",
    "foreign direct investment multinationals",
    "global value chains trade",
    "trade liberalization developing countries",
    "protectionism economic nationalism",
    "World Trade Organization",
    "regional trade agreements",
    "export led growth",
    "import substitution industrialization",
    "trade and inequality",
    "trade and environment",
    "supply chain resilience reshoring",
    "North American supply chains",
    "US trade policy history",
    "Canada trade diversification",
    "Mexico manufacturing exports",
    "exchange rate trade competitiveness",
    "trade finance global payments",

    # Policy (25 topics)
    "tax policy income redistribution",
    "minimum wage employment effects",
    "healthcare economics policy reform",
    "housing policy affordability",
    "immigration economics labor market",
    "immigration policy United States",
    "environmental policy carbon tax",
    "climate policy economics",
    "antitrust policy big tech",
    "social security reform",
    "welfare programs poverty reduction",
    "universal basic income",
    "drug policy economics",
    "criminal justice economics",
    "education policy school choice",
    "infrastructure investment economics",
    "industrial policy government intervention",
    "inequality policy redistribution",
    "gender pay gap policy",
    "racial economic inequality policy",
    "urban policy cities economics",
    "rural economics regional development",
    "food policy agriculture economics",
    "technology policy innovation",
    "data privacy economics regulation",

    # Development Economics (15 topics)
    "economic development poverty",
    "foreign aid effectiveness",
    "microfinance poverty reduction",
    "human capital development",
    "institutional quality governance development",
    "corruption economic development",
    "China economic development",
    "India economic development",
    "Africa economic growth",
    "Latin America economic development",
    "Mexico economic development",
    "inequality economic mobility",
    "sustainable development goals",
    "urbanization economic development",
    "technology adoption developing countries",

    # Behavioral and Innovation Economics (15 topics)
    "behavioral economics nudge theory",
    "prospect theory loss aversion",
    "bounded rationality decision making",
    "cognitive biases economic decisions",
    "present bias time inconsistency",
    "social preferences fairness economics",
    "happiness economics wellbeing",
    "experimental economics lab field",
    "neuroeconomics brain decision making",
    "innovation diffusion technology adoption",
    "entrepreneurship innovation ecosystems",
    "artificial intelligence labor market",
    "automation jobs future of work",
    "platform economy sharing economy",
    "digital economy e-commerce"
]

CHECKPOINT_FILE = "fetch_checkpoint.json"
OUTPUT_FILE = "papers.json"

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

def fetch_papers_for_topic(query, num_papers=200):
    print(f"  Fetching: {query}")
    
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "filter": "open_access.is_oa:true",
        "per-page": num_papers,
        "select": "id,title,abstract_inverted_index,authorships,publication_year,doi,concepts"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
    except Exception as e:
        print(f"  Error fetching {query}: {e}")
        return []
    
    papers = []
    for work in data.get("results", []):
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
        papers.append(paper)
    
    print(f"  Got {len(papers)} papers")
    return papers

if __name__ == "__main__":
    checkpoint = load_checkpoint()
    completed_topics = set(checkpoint["completed_topics"])
    all_papers = checkpoint["papers"]
    
    print(f"Starting fetch. {len(completed_topics)} topics already completed.")
    print(f"Papers collected so far: {len(all_papers)}")
    print(f"Total topics to fetch: {len(TOPICS)}\n")
    
    for i, topic in enumerate(TOPICS):
        if topic in completed_topics:
            print(f"Skipping (already done): {topic}")
            continue
        
        print(f"[{i+1}/{len(TOPICS)}]")
        papers = fetch_papers_for_topic(topic)
        
        for paper in papers:
            all_papers[paper["id"]] = paper
        
        checkpoint["completed_topics"].append(topic)
        checkpoint["papers"] = all_papers
        save_checkpoint(checkpoint)
        
        print(f"  Running total unique papers: {len(all_papers)}\n")
        time.sleep(1)
    
    unique_papers = list(all_papers.values())
    print(f"\nFetch complete! Total unique papers: {len(unique_papers)}")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique_papers, f, indent=2)
    
    print(f"Saved to {OUTPUT_FILE}")
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    print("Checkpoint file cleaned up.")