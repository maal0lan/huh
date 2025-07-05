import json
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
with open("rag_data.json", "r") as f:
    chunks = json.load(f)
def get_embedding(text):
    return model.encode(text).tolist()
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
def semantic_search(query, top_k=3):
    query_vec = get_embedding(query)
    scored_chunks = []
    for chunk in chunks:
        sim = cosine_similarity(query_vec, chunk["embedding"])
        scored_chunks.append((sim, chunk["text"]))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]
if __name__ == "__main__":
    query = input("Enter your query: ")
    results = semantic_search(query, top_k=3)
    print("\n Top results:\n")
    for score, text in results:
        print(f"Score: {score:.4f}")
        print(f"Text: {text}\n")
