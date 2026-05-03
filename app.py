"""
app.py — Flask Backend for RAG Dashboard
Serves the HTML dashboard and provides API endpoints for querying, chunk viewing, and database stats.
"""

import os
import json
import chromadb
import ollama
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ─── Configuration ──────────────────────────────────────────────────────────────
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "nomic-embed-text"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Fast, capable model on Groq

# ─── Initialize Clients ─────────────────────────────────────────────────────────
groq_client = None
chroma_client = None
collection = None


def init_chroma():
    """Initialize ChromaDB client and collection."""
    global chroma_client, collection
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"✅ ChromaDB loaded: {collection.count()} chunks in collection")
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")


def get_groq_client():
    """Get Groq client, initializing it dynamically if needed."""
    global groq_client
    if groq_client is not None:
        return groq_client
        
    # Force reload of .env
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY")
    
    if api_key and api_key != "your_groq_api_key_here":
        groq_client = Groq(api_key=api_key)
        print("✅ Groq client initialized dynamically")
        return groq_client
    else:
        print("⚠️  GROQ_API_KEY not set in .env file")
        return None


def get_embedding(text: str) -> list[float]:
    """Generate embedding using Ollama nomic-embed-text."""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


# ─── Routes ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML dashboard."""
    return render_template("index.html")


@app.route("/api/documents", methods=["GET"])
def get_documents():
    """List all ingested PDFs and their metadata."""
    if collection is None:
        return jsonify({"error": "ChromaDB not initialized"}), 500

    try:
        # Get all documents with metadata
        results = collection.get(include=["metadatas", "documents"])

        # Group by source PDF
        documents = {}
        for i, meta in enumerate(results["metadatas"]):
            source = meta.get("source", "Unknown")
            if source not in documents:
                documents[source] = {
                    "name": source,
                    "total_chunks": 0,
                    "pages": set()
                }
            documents[source]["total_chunks"] += 1
            documents[source]["pages"].add(meta.get("page", 0))

        # Convert sets to sorted lists
        doc_list = []
        for doc in documents.values():
            doc["pages"] = sorted(list(doc["pages"]))
            doc["total_pages"] = len(doc["pages"])
            doc_list.append(doc)

        return jsonify({"documents": doc_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/<path:filename>", methods=["DELETE"])
def delete_document(filename):
    """Delete a document and its chunks from ChromaDB and the file system."""
    if collection is None:
        return jsonify({"error": "ChromaDB not initialized"}), 500

    try:
        # Delete from ChromaDB
        collection.delete(where={"source": filename})
        
        # Delete from filesystem
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return jsonify({"success": True, "message": f"Deleted {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload a new document to the data folder."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        file.save(os.path.join(DATA_DIR, filename))
        return jsonify({"success": True, "message": f"Uploaded {filename}"})


@app.route("/api/chunks", methods=["GET"])
def get_chunks():
    """Get all chunks with their metadata."""
    if collection is None:
        return jsonify({"error": "ChromaDB not initialized"}), 500

    try:
        source_filter = request.args.get("source", None)

        if source_filter:
            results = collection.get(
                where={"source": source_filter},
                include=["documents", "metadatas"]
            )
        else:
            results = collection.get(include=["documents", "metadatas"])

        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })

        # Sort by page and chunk_index
        chunks.sort(key=lambda x: (
            x["metadata"].get("page", 0),
            x["metadata"].get("chunk_index", 0)
        ))

        return jsonify({"chunks": chunks, "total": len(chunks)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query_rag():
    """
    RAG Query: embed question → retrieve chunks → call Groq → return answer.
    """
    if collection is None:
        return jsonify({"error": "ChromaDB not initialized"}), 500

    data = request.json
    question = data.get("question", "").strip()
    top_k = data.get("top_k", 5)

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        # Step 1: Embed the question
        query_embedding = get_embedding(question)

        # Step 2: Retrieve top-K similar chunks from ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns L2 distances; convert to similarity score
            distance = results["distances"][0][i]
            similarity = 1 / (1 + distance)  # Convert distance to similarity (0-1)

            retrieved_chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "similarity_score": round(similarity, 4)
            })

        # Step 3: Build context for the LLM
        context = "\n\n---\n\n".join([
            f"[Source: {c['metadata'].get('source', 'N/A')}, Page {c['metadata'].get('page', '?')}, "
            f"Chunk {c['metadata'].get('chunk_index', '?')}]\n{c['text']}"
            for c in retrieved_chunks
        ])

        prompt = f"""You are a helpful assistant that answers questions based on the provided context from product documentation.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise and specific
- Reference the source page numbers when relevant

ANSWER:"""

        # Step 4: Call Groq API
        answer = "Groq API key not configured. Please add your API key to the .env file."
        model_used = GROQ_MODEL

        client = get_groq_client()
        if client:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful product documentation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    model=GROQ_MODEL,
                    temperature=0.3,
                    max_tokens=1024
                )
                answer = chat_completion.choices[0].message.content
                model_used = chat_completion.model
            except Exception as e:
                answer = f"Groq API error: {str(e)}"

        return jsonify({
            "question": question,
            "answer": answer,
            "model": model_used,
            "retrieved_chunks": retrieved_chunks,
            "total_retrieved": len(retrieved_chunks),
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/database", methods=["GET"])
def get_database_info():
    """Return ChromaDB collection stats."""
    if collection is None:
        return jsonify({"error": "ChromaDB not initialized"}), 500

    try:
        count = collection.count()

        # Get sample data to determine embedding dimensions
        sample = None
        if count > 0:
            sample = collection.get(
                limit=1,
                include=["embeddings", "metadatas"]
            )

        embedding_dim = 0
        if sample and sample["embeddings"]:
            embedding_dim = len(sample["embeddings"][0])

        # Get all metadata to compute stats
        all_data = collection.get(include=["metadatas"])
        sources = set()
        pages = set()
        total_chars = 0

        for meta in all_data["metadatas"]:
            sources.add(meta.get("source", ""))
            pages.add(f"{meta.get('source', '')}_{meta.get('page', 0)}")
            total_chars += meta.get("chunk_size", 0)

        return jsonify({
            "collection_name": COLLECTION_NAME,
            "total_chunks": count,
            "total_documents": len(sources),
            "total_pages": len(pages),
            "embedding_dimensions": embedding_dim,
            "embedding_model": EMBEDDING_MODEL,
            "chroma_db_path": CHROMA_DB_DIR,
            "avg_chunk_size": round(total_chars / count) if count > 0 else 0,
            "total_characters": total_chars
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest", methods=["POST"])
def trigger_ingest():
    """Trigger re-ingestion of documents from the data folder."""
    try:
        from ingest import ingest_documents
        stats = ingest_documents()

        # Reinitialize ChromaDB connection
        init_chroma()

        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Start Server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting RAG Dashboard Server...")
    init_chroma()
    print(f"\n🌐 Dashboard: http://localhost:5000\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
