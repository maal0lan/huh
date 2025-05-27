from sentence_transformers import SentenceTransformer
import pdfplumber
import json
import os
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)


model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def get_embedding(text):
    return model.encode(text).tolist()


def process_pdf(pdf_path):
    print("extracting bro wait up.....")
    text = extract_text(pdf_path)
    print(f"GOT {len(text)} characters")

    print("Chunking text...")
    chunks = chunk_text(text)

    data = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        data.append({
            "text": chunk,
            "embedding": emb
        })

    print("Saving JSON file...")
    with open("rag_data.json", "w") as f:
        json.dump(data, f)

    print("First 300 chars:\n", text[:300])
    print("Chunks:", chunks[:2])
    print("Done spoiling everything.")

# RUN IT UPPPP -bu humana kind idk spelling nigaa
if __name__ == "__main__":
    pdf_path = "Unit-5.pdf"  # Update this to your local path
    process_pdf(pdf_path)
