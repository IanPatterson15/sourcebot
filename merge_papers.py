import json
import os

BUSINESS_FILE = "papers.json"
SOCIAL_FILE = "social_papers.json"
OUTPUT_FILE = "merged_papers.json"

def merge_papers():
    print("Loading business/economics papers...")
    with open(BUSINESS_FILE, "r") as f:
        business_papers = json.load(f)
    print(f"  Loaded {len(business_papers)} business papers")

    print("Loading social science papers...")
    with open(SOCIAL_FILE, "r") as f:
        social_papers = json.load(f)
    print(f"  Loaded {len(social_papers)} social science papers")

    # Merge and deduplicate by ID
    all_papers = {}
    for paper in business_papers:
        all_papers[paper["id"]] = paper
    for paper in social_papers:
        all_papers[paper["id"]] = paper

    unique_papers = list(all_papers.values())
    print(f"\nTotal unique papers after merge: {len(unique_papers)}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique_papers, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_papers()