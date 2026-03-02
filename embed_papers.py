import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="economics_papers")

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def embed_papers():
    with open("papers.json", "r") as f:
        papers = json.load(f)
    
    print(f"Embedding {len(papers)} papers...")
    
    for i, paper in enumerate(papers):
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
        
        if (i + 1) % 10 == 0:
            print(f"Embedded {i + 1}/{len(papers)} papers")
    
    print("Done! All papers embedded and stored.")

if __name__ == "__main__":
    embed_papers()