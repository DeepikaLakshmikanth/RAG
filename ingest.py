"""
ingest.py — PDF Ingestion Pipeline for RAG System
Extracts text from PDFs, chunks it, embeds with Nomic Embed (Ollama), and stores in ChromaDB.
"""

import os
import sys
import fitz  # PyMuPDF
import chromadb
import ollama
import json
import hashlib
from pathlib import Path
import pandas as pd


# ─── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
EMBEDDING_MODEL = "nomic-embed-text"


# ─── PDF Text Extraction ────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF file, page by page.
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append({
                    "page": page_num + 1,
                    "text": text.strip()
                })
        doc.close()
        print(f"  [OK] Extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
    except Exception as e:
        print(f"  [ERROR] Error extracting {pdf_path}: {e}")
    return pages


# ─── Excel Text Extraction ──────────────────────────────────────────────────────
def extract_text_from_excel(excel_path: str) -> list[dict]:
    """
    Extract text from an Excel file, sheet by sheet, row by row.
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...] (using sheet/row as 'page')
    """
    pages = []
    try:
        # Read all sheets into a dictionary of DataFrames
        xls = pd.read_excel(excel_path, sheet_name=None)
        page_num = 1
        for sheet_name, df in xls.items():
            # Convert each row to a string, joining columns with spaces
            for index, row in df.iterrows():
                row_text = " ".join([str(val) for val in row.values if pd.notna(val)])
                if row_text.strip():
                    pages.append({
                        "page": page_num, # We can use page_num as an abstract index
                        "text": f"[Sheet: {sheet_name}, Row: {index+2}]\n{row_text.strip()}"
                    })
                    page_num += 1
        print(f"  [OK] Extracted {len(pages)} rows/chunks from {os.path.basename(excel_path)}")
    except Exception as e:
        print(f"  [ERROR] Error extracting {excel_path}: {e}")
    return pages


# ─── Text Chunking ──────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, respecting paragraph boundaries where possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to break at a paragraph or sentence boundary
        if end < len(text):
            # Try paragraph break first
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Try sentence break
                for sep in [". ", ".\n", "! ", "? "]:
                    sent_break = text.rfind(sep, start, end)
                    if sent_break > start + chunk_size // 2:
                        end = sent_break + len(sep)
                        break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, accounting for overlap
        start = end - overlap if end < len(text) else end

    return chunks


# ─── Generate Embeddings ─────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a text string using Ollama's nomic-embed-text model.
    """
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


# ─── Main Ingestion Pipeline ────────────────────────────────────────────────────
def ingest_documents():
    """
    Main pipeline: scan documents → extract → chunk → embed → store in ChromaDB.
    Returns ingestion stats.
    """
    # Check data directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[DIR] Created data directory: {DATA_DIR}")
        print("   Please add documents to the data/ folder and run again.")
        return {"status": "no_docs", "message": "Data directory created. Add documents and re-run."}

    # Find supported files
    supported_exts = (".pdf", ".xls", ".xlsx", ".csv")
    doc_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(supported_exts)]
    if not doc_files:
        print("[WARN] No supported files found in the data/ folder.")
        return {"status": "no_docs", "message": "No supported files found in data/ folder."}

    print(f"\n[INFO] Found {len(doc_files)} file(s): {', '.join(doc_files)}\n")

    # Initialize ChromaDB
    print("[DB] Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete existing collection if it exists (fresh ingest)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("   Cleared existing collection.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "RAG documents with Nomic Embed embeddings"}
    )

    all_chunks_data = []
    total_chunks = 0

    for doc_file in doc_files:
        doc_path = os.path.join(DATA_DIR, doc_file)
        print(f"\n[PROC] Processing: {doc_file}")

        # Step 1: Extract text
        if doc_file.lower().endswith(".pdf"):
            pages = extract_text_from_pdf(doc_path)
        elif doc_file.lower().endswith((".xls", ".xlsx")):
            pages = extract_text_from_excel(doc_path)
        elif doc_file.lower().endswith(".csv"):
            try:
                # Handle CSV using pandas similarly
                df = pd.read_csv(doc_path)
                pages = []
                for index, row in df.iterrows():
                    row_text = " ".join([str(val) for val in row.values if pd.notna(val)])
                    if row_text.strip():
                        pages.append({
                            "page": index + 1,
                            "text": f"[Row: {index+2}]\n{row_text.strip()}"
                        })
                print(f"  [OK] Extracted {len(pages)} rows from {os.path.basename(doc_path)}")
            except Exception as e:
                print(f"  [ERROR] Error extracting {doc_path}: {e}")
                pages = []
        else:
            pages = []
            
        if not pages:
            continue

        # Step 2: Chunk each page
        for page_data in pages:
            page_num = page_data["page"]
            page_text = page_data["text"]
            chunks = chunk_text(page_text)

            print(f"  [CHUNK] Page {page_num}: {len(chunks)} chunks")

            for chunk_idx, chunk_text_content in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{doc_file}_{page_num}_{chunk_idx}_{chunk_text_content[:50]}".encode()
                ).hexdigest()

                # Step 3: Generate embedding
                try:
                    embedding = get_embedding(chunk_text_content)
                except Exception as e:
                    print(f"  [ERROR] Embedding error for chunk {chunk_idx}: {e}")
                    continue

                # Step 4: Store in ChromaDB
                metadata = {
                    "source": doc_file,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk_text_content),
                    "total_page_chunks": len(chunks)
                }

                collection.add(
                    ids=[chunk_id],
                    documents=[chunk_text_content],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )

                all_chunks_data.append({
                    "id": chunk_id,
                    "text": chunk_text_content,
                    "metadata": metadata
                })

                total_chunks += 1

    # Save chunk data for the frontend
    chunks_json_path = os.path.join(os.path.dirname(__file__), "chunks_data.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks_data, f, indent=2, ensure_ascii=False)

    stats = {
        "status": "success",
        "total_pdfs": len(doc_files),
        "total_chunks": total_chunks,
        "pdf_files": doc_files,
        "chroma_db_path": CHROMA_DB_DIR,
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }

    print(f"\n{'='*60}")
    print(f"[DONE] Ingestion Complete!")
    print(f"   Documents processed: {len(doc_files)}")
    print(f"   Total chunks:   {total_chunks}")
    print(f"   ChromaDB path:  {CHROMA_DB_DIR}")
    print(f"   Collection:     {COLLECTION_NAME}")
    print(f"{'='*60}\n")

    return stats


if __name__ == "__main__":
    print("[START] Starting Document Ingestion Pipeline...")
    print(f"   Data directory: {DATA_DIR}")
    print(f"   ChromaDB path:  {CHROMA_DB_DIR}")
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    print(f"   Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars\n")
    
    stats = ingest_documents()
    print(json.dumps(stats, indent=2))
