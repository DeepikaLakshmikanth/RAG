# Local RAG Dashboard

A complete, local-first Retrieval-Augmented Generation (RAG) system with a modern glassmorphism web dashboard. This project allows you to ingest documents (.pdf, .xls, .xlsx, .csv), embed them using local models, store them in a persistent vector database, and query them interactively with an LLM.

## Features

- **Document Ingestion**: Supports uploading and parsing PDF and Excel/CSV files row-by-row.
- **Local Embeddings**: Uses **Ollama** and `nomic-embed-text` to generate embeddings locally, ensuring your documents remain private.
- **Persistent Storage**: Utilizes **ChromaDB** for fast, local vector storage.
- **Groq LLM Engine**: Connects to the Groq API for blazing-fast inference using Llama 3 models.
- **Interactive UI**: A beautiful, responsive glassmorphism dashboard that allows you to manage documents, visualize the RAG query flow (including retrieved chunks and similarity scores), and track database statistics in real-time.

## Prerequisites

1. **Python 3.10+**
2. **Ollama**: Download and install [Ollama](https://ollama.com/), then pull the embedding model:
   ```bash
   ollama run nomic-embed-text
   ```
3. **Groq API Key**: You'll need a free API key from [Groq](https://console.groq.com/keys).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DeepikaLakshmikanth/RAG.git
   cd RAG
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`.
3. Use the **📤 Upload Document** button in the top right to upload your files. They will be automatically processed and ingested.
4. Ask questions in the center console to see the RAG pipeline in action!

## Architecture

- `app.py`: The Flask backend serving API routes and the dashboard frontend.
- `ingest.py`: The ingestion pipeline handles extracting text, chunking, and ChromaDB insertion.
- `templates/index.html`: A single-page application built with raw HTML/CSS/JS for maximum performance and customization.
- `data/`: Where uploaded documents are stored.
- `chroma_db/`: Local persistent vector database storage.
