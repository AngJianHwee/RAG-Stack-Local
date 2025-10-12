# SOP-RAG: Streamlit, Ollama, and Pinecone for Local RAG

This project provides a robust local development environment for a Retrieval-Augmented Generation (RAG) pipeline using Streamlit for the UI, Ollama for local text embeddings, and Pinecone Local as a vector database. All components are containerized and orchestrated using Docker Compose for a seamless setup.

## Features

- **Streamlit Application**: An interactive web interface to demonstrate RAG functionalities.
- **Automated Ollama Setup**: The required embedding model is automatically pulled on startup.
- **Pinecone Local**: A local, Docker-based Pinecone instance for vector storage and retrieval.
- **Robust Docker Compose**: Services are configured with health checks, restart policies, and an initialization service for a stable and reliable startup sequence.

## Project Structure

```
.
├── app.py                  # Streamlit application logic
├── Dockerfile.streamlit    # Dockerfile for the Streamlit app and init service
├── requirements.txt        # Python dependencies for the Streamlit app
├── docker-compose.yml      # Docker Compose configuration for all services
├── init-ollama.sh          # Initialization script to pull the Ollama model
├── ... (documentation files)
```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed on your system:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually comes with Docker Desktop)

### Installation and Setup

1.  **Build and Run with Docker Compose:**
    Navigate to the root directory of this project in your terminal and run the following command:

    ```bash
    docker compose up --build -d
    ```
    This command will:
    - Build the Docker image for the Streamlit and initialization services.
    - Start all services (`ollama`, `pinecone-local`, `init-ollama`, `streamlit`) in the correct order.
    - The `init-ollama` service will wait for Ollama to be ready and then automatically pull the `all-minilm:33m` model.
    - The Streamlit application will start only after the model is pulled and Pinecone is healthy.

2.  **Access the Streamlit Application:**
    Once all services are up and running (this might take a few minutes, especially on the first run while the model is downloading), open your web browser and go to:

    [http://localhost:8501](http://localhost:8501)

## Usage

The Streamlit application provides an interface to:

-   **Generate Embeddings**: Input text and get its vector embedding using the configured Ollama model.
-   **Create Pinecone Index**: Initialize a local Pinecone index.
-   **Upsert and Query**: Add documents to the Pinecone index and perform similarity searches.

Follow the instructions on the Streamlit UI to interact with the RAG pipeline.

## Troubleshooting

-   **Services not starting**: Check the Docker logs for each service. The `init-ollama` service log is particularly useful for debugging model pulling issues.
    ```bash
    docker compose logs -f init-ollama
    docker compose logs -f streamlit
    ```
-   **"Ollama is not responding" error**: This can happen if the Ollama container takes a long time to start. The `init-ollama` script will retry for a few minutes. If it persists, try increasing the `MAX_ATTEMPTS` in `init-ollama.sh`.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is open-source and available under the MIT License.
