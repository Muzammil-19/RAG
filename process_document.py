import fitz  # PyMuPDF for PDFs
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tiktoken

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to load text from files
def load_text(file_path):
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return "\n".join([page.get_text("text") for page in doc])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

# Function to chunk text
def chunk_text(text, chunk_size=512, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    words = enc.encode(text)
    chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size - overlap)]
    return ["".join(enc.decode(chunk)) for chunk in chunks]

# Function to generate embeddings and store in FAISS
def create_faiss_index(documents):
    vectors = [embedding_model.encode(chunk) for doc in documents for chunk in chunk_text(load_text(doc))]
    vectors = np.array(vectors, dtype=np.float32)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, "embeddings/faiss_index")
    print("FAISS index saved.")

if __name__ == "__main__":
    files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith((".pdf", ".txt", ".docx"))]
    create_faiss_index(files)
