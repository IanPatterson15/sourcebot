import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

TOPICS = [
    # Sociology
    "social inequality class",
    "racial inequality sociology",
    "social mobility stratification",
    "organizational sociology",
    "social movements collective action",
    "gender inequality workplace",
    "poverty social exclusion",
    "social capital networks",
    "culture consumption sociology",
    "urbanization social change",
    "immigration social integration",
    "family structure sociology",
    "crime deviance sociology",
    "education inequality sociology",
    "health inequality social determinants",

    # Psychology
    "consumer psychology decision making",
    "organizational behavior psychology",
    "leadership psychology management",
    "motivation work psychology",
    "cognitive biases judgment",
    "social psychology influence",
    "personality traits workplace",
    "stress burnout workplace psychology",
    "group dynamics teamwork psychology",
    "negotiation psychology",
    "trust psychology organizations",
    "emotion regulation workplace",
    "identity psychology social",
    "persuasion attitude change psychology",
    "well-being happiness psychology",

    # Political Science
    "public policy political economy",
    "regulation political science",
    "democracy institutions governance",
    "political economy inequality",
    "lobbying interest groups policy",
    "electoral politics voting behavior",
    "bureaucracy public administration",
    "international relations political economy",
    "state capacity governance",
    "political ideology polarization",
    "corruption governance political science",
    "federalism decentralization policy",
    "environmental policy politics",
    "welfare state political economy",
    "trade policy political economy",

    # Demographics
    "population aging demographics",
    "migration demographics economics",
    "urbanization demographics",
    "fertility population economics",
    "labor force demographics",
    "immigration demographics United States",
    "demographic change economic impact",
    "population health demographics",
    "aging workforce demographics",
    "generational differences demographics",

    # Law and Economics
    "property rights economics law",
    "contract law economics",
    "corporate law governance",
    "antitrust law economics",
    "regulation law economics",
    "intellectual property law economics",
    "labor law employment economics",
    "environmental law economics",
    "financial regulation law",
    "liability law economics",
    "bankruptcy law economics",
    "tax law economics",

    # History
    "economic history United States",
    "business history corporate",
    "financial crisis history",
    "industrial revolution economic history",
    "trade history globalization",
    "labor history unions",
    "monetary history central banking",
    "inequality history long run",
    "development economic history",
    "North American economic history",
    "Canadian economic history",
    "Great Depression economic history",
    "postwar economic history",
    "technology history economic impact",
    "colonialism economic history"
]

CHECKPOINT_FILE = "social_fetch_checkpoint.json"
OUTPUT_FILE = "social_papers.json"

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

    print(f"Starting social sciences fetch.")
    print(f"Topics to fetch: {len(TOPICS)}")
    print(f"Already completed: {len(completed_topics)}")
    print(f"Papers so far: {len(all_papers)}\n")

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

        print(f"  Running total: {len(all_papers)}\n")
        time.sleep(1)

    unique_papers = list(all_papers.values())
    print(f"\nFetch complete! Total unique social science papers: {len(unique_papers)}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique_papers, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    print("Done!")
