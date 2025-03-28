from fastapi import FastAPI, File, UploadFile, Depends, Query, HTTPException, BackgroundTasks
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os
import time
import tiktoken
from sqlalchemy import create_engine, Column, String, Text, select, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import uuid
from typing import List
from pydantic import BaseModel
from sqlalchemy.sql import func  # Import func
# import threading
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")  # Define IST timezone


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Configuration
DATABASE_URL = "sqlite:///./conversation.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# Define ConversationHistory Model
class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    vendor_name = Column(String(100), nullable=False)
    user_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST))  # Store time in IST



Base.metadata.create_all(bind=engine)

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load FAISS index & embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = None

# Load Llama Model
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    n_batch=128
)

# Tokenizer for truncation
enc = tiktoken.get_encoding("cl100k_base")

def truncate_text(text, max_tokens=400):
    tokens = enc.encode(text)
    truncated_tokens = tokens[:max_tokens]
    return enc.decode(truncated_tokens)

# Load stored document chunks
def load_chunks():
    try:
        with open("embeddings/chunk_map.txt", "r", encoding="utf-8") as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    except FileNotFoundError:
        return {}

chunk_dict = load_chunks()

# Rebuild FAISS index
def rebuild_faiss():
    global index, chunk_dict
    chunk_dict = load_chunks()

    if not chunk_dict:
        print("No chunks found.")
        return

    chunk_texts = list(chunk_dict.values())
    chunk_embeddings = embedding_model.encode(chunk_texts).astype(np.float32)

    faiss_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    faiss_index.add(chunk_embeddings)
    faiss.write_index(faiss_index, "embeddings/faiss_index")

    index = faiss.read_index("embeddings/faiss_index")
    print("FAISS index reloaded.")

# Search FAISS
def search_faiss(query, top_k=10):
    if index is None:
        return "No relevant information found."

    query_vector = embedding_model.encode(query).astype(np.float32)
    distances, indices = index.search(np.array([query_vector]), top_k)

    valid_indices = [i for i in indices[0] if 0 <= i < len(chunk_dict)]
    retrieved_texts = [chunk_dict[i].strip() for i in valid_indices]
    print("Final Retrieved Context:\n", "\n\n".join(retrieved_texts))


    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant information found."

# API Models
class QueryRequest(BaseModel):
    vendor_name: str
    user_type: str
    query: str

@app.get("/")
def home():
    return {"message": "Welcome to the RAG Chatbot API"}

@app.post("/query/")
async def query_llm(request: QueryRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    rebuild_faiss()  # Ensure FAISS is updated

    vendor_name = request.vendor_name
    user_type = request.user_type
    query = request.query

    # Save user query immediately
    conversation = ConversationHistory(vendor_name=vendor_name, user_type=user_type, message=query)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    # Run LLM processing in the background
    background_tasks.add_task(process_llm_response, vendor_name, query, db)

    return {"message": "Processing your query, please wait..."}

def process_llm_response(vendor_name: str, query: str, db: Session):
    """Handles LLM processing and stores response in DB."""
    context = search_faiss(query)

    # Fetch the last 5 messages from conversation history
    recent_messages = db.execute(
        select(ConversationHistory)
        .where(ConversationHistory.vendor_name == vendor_name)
        .order_by(ConversationHistory.timestamp.desc())  # Get latest messages first
        .limit(5)
    ).scalars().all()

    # Format previous messages as context
    chat_history = "\n".join([f"{msg.user_type}: {msg.message}" for msg in reversed(recent_messages)])

    if context == "No relevant information found.":
        bot_response = "I don't have enough information to answer this."
    else:
        prompt = f"""
            You are **Smarty**, a highly intelligent and conversational AI assistant. Your task is to provide **concise, accurate, and helpful responses** based on the provided context. 

            ### Rules:
            1. **Introduce yourself as "Smarty"** if asked your name.
            2. **Never ask for information already present in the context.**
            3. **If job start and end dates are available, automatically calculate the duration.**
            4. **If the answer is derivable from the context, provide it confidently—never say "I don't have enough details."**
            5. **Use recent conversation history to improve responses and maintain context.**
            6. **For greetings (e.g., "hi", "hello"), respond naturally instead of retrieving factual information.**
            7. **Avoid repetitive answers; respond naturally based on previous messages.**
            8. **If context is irrelevant to the query, provide a general helpful response.**
            9. **don't use based on the context or context gives, type of things, answer like a human in a conversation**

            ---

            ### **Recent Conversation History:**
            {chat_history}

            ### **Context:**
            {context}

            ### **User's Query:**
            {query}

            ### **Smarty's Answer:**
            """



        start_time = time.time()
        response = llm(prompt, max_tokens=128)
        end_time = time.time()

        print("Llama Processing Time:", end_time - start_time)
        bot_response = response["choices"][0]["text"].strip()
        print("response:", bot_response)

    # Store bot response in DB
    chatbot_conversation = ConversationHistory(vendor_name=vendor_name, user_type="chatbot", message=bot_response)
    db.add(chatbot_conversation)
    db.commit()
    db.refresh(chatbot_conversation)

@app.post("/upload/")
async def upload_file(file: UploadFile):
    file_location = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    rebuild_faiss()
    return {"message": f"File '{file.filename}' uploaded successfully."}

@app.get("/conversation-history/",include_in_schema=False)
async def get_conversation_history(
    vendor_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    # Get total message count for the vendor
    total_messages = db.execute(
        select(func.count()).where(ConversationHistory.vendor_name == vendor_name)
    ).scalar() or 0  

    total_pages = max(1, -(-total_messages // page_size))  # Calculate total pages (ceil division)

    if page > total_pages:
        raise HTTPException(status_code=404, detail="Page not found")

    # Fetch the last 'page_size' messages, then order them in ascending order
    results = (
        db.execute(
            select(ConversationHistory)
            .where(ConversationHistory.vendor_name == vendor_name)
            .order_by(ConversationHistory.timestamp.desc())  # Newest messages first
            .offset((page - 1) * page_size)  # Correct offset for pagination
            .limit(page_size)
        )
        .scalars()
        .all()
    )

    # Reverse the results to show in ascending order
    results.reverse()

    return {
        "page": page,
        "total_pages": total_pages,
        "total_messages": total_messages,
        "messages": results,
    }
