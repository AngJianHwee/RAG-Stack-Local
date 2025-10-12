# SOP-RAG: Streamlit, Ollama (Native), and Pinecone for Local RAG

This project provides a local development environment for a Retrieval-Augmented Generation (RAG) pipeline using Streamlit for the UI and Pinecone Local as a vector database. Ollama is expected to be installed and running natively on your host machine to leverage its full capabilities, including GPU access.

## Features

- **Streamlit Application**: An interactive web interface to demonstrate RAG functionalities.
- **Native Ollama Integration**: Connects to a locally running Ollama instance for text embeddings, allowing direct GPU utilization.
- **Pinecone Local**: A local, Docker-based Pinecone instance for vector storage and retrieval.
- **Docker Compose**: Simplifies the setup and orchestration of the Streamlit app and Pinecone Local.

## Project Structure

```
.
├── app.py                  # Streamlit application logic
├── Dockerfile.streamlit    # Dockerfile for the Streamlit app (Ubuntu-based)
├── requirements.txt        # Python dependencies for the Streamlit app
├── docker-compose.yml      # Docker Compose configuration for Pinecone Local and Streamlit
├── ... (documentation files)
```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed on your system:

-   **Docker** and **Docker Compose**: [Install Docker](https://docs.docker.com/get-docker/)
-   **Ollama**: [Install Ollama natively on your host machine](https://ollama.com/download). Make sure it's running and accessible on port `11434`.
    -   On Ubuntu run `curl -fsSL https://ollama.com/install.sh | sh` to install.
    -   After installing Ollama, pull the required model (e.g., `all-minilm:33m`):
        ```bash
        ollama pull all-minilm:33m
        ```

### Installation and Setup

1.  **Start Ollama Natively:**
    Ensure your Ollama instance is running on your host machine. You can usually start it by simply running `ollama serve` in your terminal after installation.

2.  **Configure OLLAMA_HOST (Important for Linux Users):**
    The `docker-compose.yml` file is configured to use `OLLAMA_HOST=http://host.docker.internal:11434`.
    -   **macOS / Windows (Docker Desktop)**: `host.docker.internal` will automatically resolve to your host machine's IP address. No further action is usually needed.
    -   **Linux**: `host.docker.internal` might not work out-of-the-box. You may need to:
        -   Find your host machine's IP address (e.g., `ip a` or `ifconfig`).
        -   Replace `host.docker.internal` in `docker-compose.yml` with your host's IP address (e.g., `http://192.168.1.100:11434`).
        -   Alternatively, you can try using the Docker bridge IP, often `http://172.17.0.1:11434`.

3.  **Build and Run with Docker Compose:**
    Navigate to the root directory of this project in your terminal and run the following command:

    ```bash
    docker compose up --build -d
    ```
    This command will:
    -   Build the Docker image for the Streamlit application.
    -   Start the `pinecone-local` and `streamlit` services.
    -   The Streamlit application will connect to your native Ollama instance.

4.  **Access the Streamlit Application:**
    Once the services are up and running, open your web browser and go to:

    [http://localhost:8501](http://localhost:8501)

## Usage

The Streamlit application provides an interface to:

-   **Generate Embeddings**: Input text and get its vector embedding using the configured Ollama model (from your native Ollama instance).
-   **Create Pinecone Index**: Initialize a local Pinecone index.
-   **Upsert and Query**: Add documents to the Pinecone index and perform similarity searches.

Follow the instructions on the Streamlit UI to interact with the RAG pipeline.

## Troubleshooting

-   **"Could not connect to Ollama" error in Streamlit**:
    -   Ensure Ollama is running natively on your host machine (`ollama serve`).
    -   Verify that the `all-minilm:33m` model has been pulled (`ollama pull all-minilm:33m`).
    -   Check your `OLLAMA_HOST` configuration in `docker-compose.yml`, especially if you are on Linux.
    -   Ensure no firewall is blocking port `11434` on your host machine.
-   **Pinecone connection issues**: Check the Docker logs for the `pinecone-local` service:
    ```bash
    docker compose logs -f pinecone-local
    ```
-   **Streamlit application issues**: Check the Docker logs for the `streamlit` service:
    ```bash
    docker compose logs -f streamlit
    ```

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is open-source and available under the MIT License.
