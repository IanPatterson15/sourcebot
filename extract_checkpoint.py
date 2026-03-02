import json

print("Loading checkpoint...")
with open("master_checkpoint.json", "r") as f:
    checkpoint = json.load(f)

papers = list(checkpoint["papers"].values())
print(f"Found {len(papers)} papers in checkpoint")

with open("new_papers.json", "w") as f:
    json.dump(papers, f, indent=2)

print("Saved to new_papers.json")