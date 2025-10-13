# Streamlit RAG with Ollama and Pinecone Local

This project demonstrates a simple Retrieval-Augmented Generation (RAG) setup using Streamlit for the UI, Ollama for local text embeddings, and Pinecone Local as a vector database. All components are designed to run within Docker containers.

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

### 3. Build the Streamlit Application Docker Image

Navigate to the root directory of this project (where `Dockerfile`, `app.py`, and `requirements.txt` are located) and build the Streamlit app's Docker image. Note that `requirements.txt` now includes `pinecone[grpc]` for the gRPC client.

```bash
docker build -t streamlit-rag-app .
```

### 4. Run the Streamlit Application Docker Container

Run the Streamlit container, ensuring it's on the same Docker network as Ollama and Pinecone Local so they can communicate. The Streamlit app connects to Pinecone Local using the service name `pinecone-local` within the Docker network. By default, Docker containers run on the `bridge` network if not specified.

```bash
docker run -d -p 8501:8501 --name streamlit-app --network bridge streamlit-rag-app
```
*Note: If you configured Ollama or Pinecone Local to run on a custom Docker network, replace `bridge` with the name of that network.*

### 5. Access the Streamlit Application

Open your web browser and navigate to:

[http://localhost:8501](http://localhost:8501)

You can now interact with the Streamlit application to:
*   Enter text, get its embedding from Ollama, and store it in the Pinecone Local vector database.
*   Enter a query text, get its embedding, and retrieve similar stored texts from Pinecone Local.

## Project Structure

*   `requirements.txt`: Lists the Python dependencies for the Streamlit application.
*   `app.py`: The main Streamlit application script.
*   `Dockerfile`: Defines the Docker image for the Streamlit application.
