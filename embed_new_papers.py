import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="economics_papers")

CHECKPOINT_FILE = "embed_checkpoint.json"

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed_ids": []}

def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

def embed_new_papers():
    print("Loading merged papers...")
    with open("merged_papers.json", "r") as f:
        all_papers = json.load(f)
    print(f"  Total papers in merged file: {len(all_papers)}")

    print("Checking what's already in ChromaDB...")
    existing = collection.get(include=[])
    existing_ids = set(existing["ids"])
    print(f"  Already embedded: {len(existing_ids)} papers")

    new_papers = [p for p in all_papers if p["id"] not in existing_ids]
    print(f"  New papers to embed: {len(new_papers)}\n")

    if not new_papers:
        print("Nothing new to embed — you're all up to date!")
        return

    checkpoint = load_checkpoint()
    completed_ids = set(checkpoint["completed_ids"])

    new_papers = [p for p in new_papers if p["id"] not in completed_ids]
    print(f"  Remaining after checkpoint: {len(new_papers)}\n")

    for i, paper in enumerate(new_papers):
        text_to_embed = f"{paper['title']}. {paper['abstract']}"
        embedding = get_embedding(text_to_embed)

        authors_str = ", ".join(paper.get("authors", []))
        collection.add(
            ids=[paper["id"]],
            embeddings=[embedding],
            documents=[paper["abstract"]],
            metadatas=[{
                "title": paper["title"] or "",
                "authors": authors_str,
                "year": str(paper.get("year", "")),
                "doi": paper.get("doi") or ""
            }]
        )

        checkpoint["completed_ids"].append(paper["id"])

        if (i + 1) % 10 == 0:
            save_checkpoint(checkpoint)
            print(f"Embedded {i + 1}/{len(new_papers)} new papers")

    save_checkpoint(checkpoint)
    print(f"\nDone! All new papers embedded.")
    print(f"Total papers now in ChromaDB: {len(existing_ids) + len(new_papers)}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

if __name__ == "__main__":
    embed_new_papers()