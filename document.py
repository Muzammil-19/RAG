from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # For PDFs
import os

def load_document(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def chunk_text(text, chunk_size=512, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    file_path = "resume_01_25.pdf"  # Replace with your PDF file path
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found!")
    else:
        text = load_document(file_path)
        print(f"Extracted {len(text)} characters from the document.")
        
        chunks = chunk_text(text)
        print(f"Generated {len(chunks)} text chunks.")

        # Print first 3 chunks as an example
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:\n{chunk}\n{'-'*40}")
