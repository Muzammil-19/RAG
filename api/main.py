from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os
import time
import tiktoken

app = FastAPI()

# Load FAISS index and embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = None  # Placeholder for FAISS index

# Load Llama Model with limited context window
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", 
    n_ctx=2048, 
    n_threads=4,  # Adjust based on your CPU cores
    n_batch=128
)

# Load tokenizer for token truncation
enc = tiktoken.get_encoding("cl100k_base")

def truncate_text(text, max_tokens=400):
    tokens = enc.encode(text)
    truncated_tokens = tokens[:max_tokens]
    return enc.decode(truncated_tokens)

# Load stored document chunks
def load_chunks():
    with open("embeddings/chunk_map.txt", "r", encoding="utf-8") as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}  # Convert to dict

chunk_dict = load_chunks()

def rebuild_faiss():
    global index, chunk_dict  # Ensure global update

    # Load document chunks again
    chunk_dict = load_chunks()

    # Generate embeddings for chunks
    chunk_texts = list(chunk_dict.values())
    chunk_embeddings = embedding_model.encode(chunk_texts).astype(np.float32)

    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    faiss_index.add(chunk_embeddings)

    # Save updated index
    faiss.write_index(faiss_index, "embeddings/faiss_index")

    # Reload FAISS index
    index = faiss.read_index("embeddings/faiss_index")
    print("FAISS index reloaded.")



def search_faiss(query, top_k=10):
    query_vector = embedding_model.encode(query).astype(np.float32)
    distances, indices = index.search(np.array([query_vector]), top_k)

    # **Retrieve valid indices**
    valid_indices = [i for i in indices[0] if 0 <= i < len(chunk_dict)]

    # **Fetch retrieved chunks**
    retrieved_texts = [chunk_dict[i].strip() for i in valid_indices]

    # **Debugging Info**
    print("Retrieved indices:", indices[0])
    print("Valid Retrieved Indices:", valid_indices)
    print("Final Retrieved Context:\n", "\n\n".join(retrieved_texts[:5]))

    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant information found."



@app.get("/")
def home():
    return {"message": "Welcome to the RAG Chatbot API"}

@app.post("/query/")
async def query_llm(query: str):
    rebuild_faiss()  # Ensure FAISS is updated

    context = search_faiss(query)

    if context == "No relevant information found.":
        return {"answer": "I don't have enough information to answer this."}

    prompt = f"""
            You are a highly intelligent AI assistant. Your goal is to provide clear, concise, and meaningful answers using the given context.

            ### Context:
            {context}

            ### Question:
            {query}

            ### Instructions:
            - Use the provided context to answer the question accurately.
            - If the context is insufficient, say, "I don't have enough information to answer this."
            - Keep the response **concise, relevant, and well-structured**.
            - Avoid unnecessary repetition or incomplete sentences.

            ### Answer:
            """

    start_time = time.time()
    response = llm(prompt, max_tokens=128,  stream=True)

    full_response = ""
    for chunk in response:
        full_response += chunk["choices"][0]["text"]
        print(full_response)
    end_time = time.time()

    print("Llama Processing Time:", end_time - start_time)
    # print({"answer": response["choices"][0]["text"].strip()})
    # return {"answer": response["choices"][0]["text"].strip()}
    print("response time:",end_time-start_time)
    return {"answer":full_response}


@app.post("/upload/")
async def upload_file(file: UploadFile):
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    rebuild_faiss()  # Ensure FAISS is updated
    return {"message": f"File '{file.filename}' uploaded successfully."}

