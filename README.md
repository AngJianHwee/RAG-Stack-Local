# Streamlit RAG with Ollama and Pinecone Local

This project demonstrates a simple Retrieval-Augmented Generation (RAG) setup using Streamlit for the UI (running natively), Ollama for local text embeddings (running in Docker), and Pinecone Local as a vector database (running in Docker).

## Prerequisites

Before you begin, ensure you have Docker installed and running on your system.

## Setup and Running the Application

Follow these steps to get the Streamlit RAG application up and running:

### 1. Start Ollama Service

First, run the Ollama Docker container and pull the `all-minilm:33m` model. This model will be used for generating text embeddings.

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull all-minilm:33m
docker exec -d ollama ollama serve
```

Verify Ollama is running and the model is available (optional):
```bash
# Try embedding
curl -X POST \
    http://localhost:11434/api/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model": "all-minilm:33m", "prompt": "Hello world"}'
```

### 2. Start Pinecone Local Service

Next, run the Pinecone Local Docker container. This will serve as your local vector database.

```bash
docker pull ghcr.io/pinecone-io/pinecone-local:latest
docker rm -f pinecone-local || true # Remove if already exists
docker run -d \
    --name pinecone-local \
    -e PORT=5081 \
    -e PINECONE_HOST=localhost \
    -p 5081-6000:5081-6000 \
    --platform linux/amd64 \
    ghcr.io/pinecone-io/pinecone-local:latest
```

### 3. Run the Streamlit Application Natively

First, ensure you have the Python dependencies installed:

```bash
pip install -r requirements.txt
```

Then, run the Streamlit application:

```bash
streamlit run app.py
```

### 4. Access the Streamlit Application

Open your web browser and navigate to:

[http://localhost:8501](http://localhost:8501)

You can now interact with the Streamlit application to:
*   Enter text, get its embedding from Ollama, and store it in the Pinecone Local vector database.
*   Enter a query text, get its embedding, and retrieve similar stored texts from Pinecone Local.

## Project Structure

*   `requirements.txt`: Lists the Python dependencies for the Streamlit application, including `langchain`.
*   `app.py`: The main Streamlit application script, now with enhanced text chunking options.

## New Features

### Advanced Text Chunking

The application now includes advanced text chunking options using `langchain.text_splitter.RecursiveCharacterTextSplitter`. You can adjust the `Chunk Size` and `Chunk Overlap` using sliders in the sidebar to optimize how your documents are split before embedding and storage. This allows for more granular control over the RAG process, potentially improving retrieval quality.
