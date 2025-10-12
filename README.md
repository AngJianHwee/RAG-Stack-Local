# RAG-Stack-Local: Streamlit, Ollama, and Pinecone for Local RAG

This project provides a local development environment for a Retrieval-Augmented Generation (RAG) pipeline using Streamlit for the UI, Ollama for local text embeddings, and Pinecone Local as a vector database. All components are containerized using Docker Compose for easy setup and management.

## Features

- **Streamlit Application**: An interactive web interface to demonstrate RAG functionalities.
- **Ollama Integration**: Generate text embeddings locally using various Ollama models (e.g., `all-minilm:33m`).
- **Pinecone Local**: A local, Docker-based Pinecone instance for vector storage and retrieval, perfect for rapid prototyping.
- **Docker Compose**: Simplifies the setup and orchestration of all services.

## Project Structure

```
.
├── app.py                  # Streamlit application logic
├── Dockerfile.streamlit    # Dockerfile for the Streamlit app
├── requirements.txt        # Python dependencies for the Streamlit app
├── docker-compose.yml      # Docker Compose configuration
├── Generating Local Text Embeddings with Ollama via Docker and Python Requests # Documentation for Ollama setup
└── Local Pinecone Vector Database- Docker-Based Setup and Python Integration for Fast Prototyping.NOTES # Documentation for Pinecone Local setup
```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed on your system:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually comes with Docker Desktop)

### Installation and Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # If this was a git repository, you would clone it here.
    # For this task, assume you are in the project directory.
    ```

2.  **Build and Run with Docker Compose:**
    Navigate to the root directory of this project in your terminal and run the following command:

    ```bash
    docker compose up --build -d
    ```
    The `-d` flag runs the services in detached mode.

3.  **Pull Ollama Model:**
    The `ollama` service is configured to run, but you might need to manually pull the embedding model. You can do this by executing a command inside the running Ollama container:

    ```bash
    docker exec -it ollama ollama pull all-minilm:33m
    ```
    *Note: The `app.py` is configured to use `all-minilm:33m` by default. You can change this in the Streamlit UI.*

4.  **Access the Streamlit Application:**
    Once all services are up and running (this might take a few minutes for Ollama and Pinecone to become healthy), open your web browser and go to:

    [http://localhost:8501](http://localhost:8501)

## Usage

The Streamlit application provides an interface to:

-   **Generate Embeddings**: Input text and get its vector embedding using the configured Ollama model.
-   **Create Pinecone Index**: Initialize a local Pinecone index.
-   **Upsert and Query**: Add documents to the Pinecone index and perform similarity searches.

Follow the instructions on the Streamlit UI to interact with the RAG pipeline.

## Troubleshooting

-   **Services not starting**: Check the Docker logs for each service:
    ```bash
    docker compose logs ollama
    docker compose logs pinecone-local
    docker compose logs streamlit
    ```
-   **Ollama connection issues**: Ensure the Ollama model (`all-minilm:33m`) has been pulled successfully.
-   **Pinecone connection issues**: Verify that the Pinecone Local container is running and healthy.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is open-source and available under the MIT License.
