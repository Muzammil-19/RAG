from fastapi import FastAPI, File, UploadFile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os

app = FastAPI()

# Load FAISS index and embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/faiss_index")

# Load Llama Model
llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

# Function to get top-k similar chunks
def search_faiss(query, top_k=3):
    query_vector = embedding_model.encode(query).astype(np.float32)
    distances, indices = index.search(np.array([query_vector]), top_k)
    
    # Load stored documents
    files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith((".pdf", ".txt", ".docx"))]
    texts = [open(f, "r", encoding="utf-8").read() for f in files]

    retrieved_texts = [texts[i] for i in indices[0] if i < len(texts)]
    return "\n".join(retrieved_texts)

@app.get("/")
def home():
    return {"message": "Welcome to the RAG Chatbot API"}

@app.post("/query/")
def query_llm(query: str):
    context = search_faiss(query)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm(prompt)
    return {"answer": response["choices"][0]["text"]}

@app.post("/upload/")
async def upload_file(file: UploadFile):
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"message": f"File '{file.filename}' uploaded successfully."}
