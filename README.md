# Streamlit RAG with Ollama and Pinecone Local

This project demonstrates a Retrieval-Augmented Generation (RAG) system using Streamlit for the UI, Ollama for local embeddings, and Pinecone Local for vector storage. It now includes user authentication and session management to ensure data isolation.

## Features

*   **User Authentication:** Login and registration system to manage user access.
*   **Session Management:** Documents and queries are associated with logged-in users, ensuring data privacy.
*   **Text Embedding:** Utilizes Ollama (specifically `all-minilm:33m`) to generate embeddings for text.
*   **Vector Storage:** Stores text embeddings in a local Pinecone instance.
*   **Semantic Search:** Retrieves similar text chunks based on a query using Pinecone.
*   **Advanced Text Chunking:** Configurable `Chunk Size` and `Chunk Overlap` using `langchain.text_splitter.RecursiveCharacterTextSplitter`.
*   **Admin Page:** A dedicated page for users to view, filter, paginate, and manage their stored embeddings. Features include:
    *   Viewing embeddings with their ID, text content, original text ID, and insert date.
    *   **Configurable Filtering:** Filter embeddings by "Text Content", "ID", "Original Text ID", or "Insert Date" using a dropdown and a search input. Filters are applied explicitly via an "Apply Filters & Pagination" button.
    *   **Advanced Pagination:** Browse through embeddings with a configurable number of items per page (selected via a dropdown), "Previous" and "Next" buttons, and direct page number buttons for quick navigation.
    *   **Enhanced Deletion:** Batch deletion of selected embeddings with confirmation buttons at both the top and bottom of the list. Individual deletion of embeddings directly next to each entry.
    *   **Interactive Styling:** Improved visual styling for embedding entries, with selected cards changing color for clear identification.

## Project Structure

*   `app.py`: The main Streamlit application, handling UI, session management, and orchestrating calls to utility functions.
*   `utils.py`: Contains utility functions for user management (loading/saving users, password hashing) and Ollama embedding generation.
*   `pinecone_utils.py`: Encapsulates Pinecone initialization and interaction logic for both RAG embeddings and user credentials.
*   `requirements.txt`: Lists Python dependencies.
*   `README.md`: This documentation.

## Setup and Installation

### 1. Prerequisites

*   **Docker:** Ensure Docker is installed and running on your system.
*   **Ollama:** Download and install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull the `all-minilm:33m` model: `ollama pull all-minilm:33m`

### 2. Run Pinecone Local with Docker

Start the Pinecone Local service using Docker:

```bash
docker run -e PINECONE_API_KEY="pclocal" -p 5081:5081 pinecone/pinecone-local
```

### 3. Clone the Repository

```bash
git clone https://github.com/AngJianHwee/RAG-Stack-Local.git
cd RAG-Stack-Local
```

### 4. Install Python Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

## Configuration

This project uses environment variables for sensitive information and configurable parameters, loaded from a `.env` file.

1.  **Create `.env` file:** In the root directory of the project, create a file named `.env`.
2.  **Add variables:** Populate the `.env` file with the following key-value pairs:

    ```
    PINECONE_API_KEY="pclocal"
    PINECONE_HOST="http://localhost:5081"
    RAG_INDEX_NAME="rag-index"
    USER_INDEX_NAME="user-index"
    DIMENSION=384
    OLLAMA_EMBEDDING_URL="http://localhost:11434/api/embeddings"
    ```
    *   `PINECONE_API_KEY`: Your Pinecone API key (for local, "pclocal" is sufficient).
    *   `PINECONE_HOST`: The host for your Pinecone instance (e.g., "http://localhost:5081" for local).
    *   `RAG_INDEX_NAME`: The name of the Pinecone index for RAG embeddings.
    *   `USER_INDEX_NAME`: The name of the Pinecone index for user credentials.
    *   `DIMENSION`: The dimension of the embeddings (e.g., 384 for `all-minilm:33m`).
    *   `OLLAMA_EMBEDDING_URL`: The URL for the Ollama embedding service.

## Testing

Unit tests are provided for `utils.py` and `pinecone_utils.py` to ensure the core logic functions as expected.

### 1. Install Test Dependencies

Ensure `pytest` and `pytest-mock` are installed (they are included in `requirements.txt`):

```bash
pip install -r requirements.txt
```

### 2. Run Tests

Navigate to the project root directory and run `pytest`:

```bash
pytest
```

## Usage

1.  **Access the Application:** Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
2.  **Register/Login:**
    *   On the login page, you can register a new user by providing a username and password.
    *   After registration, log in with your new credentials.
3.  **Store Embeddings:**
    *   Once logged in, enter text into the "Enter text to embed and store:" text area.
    *   Adjust "Chunk Size" and "Chunk Overlap" using the sidebar sliders if desired.
    *   Click "Store Embedding" to process the text, generate embeddings, and store them in Pinecone, associated with your user ID.
4.  **Admin Page:**
    *   Click the "Admin Page" button in the sidebar.
    *   On this page, you can view all embeddings associated with your user ID, ordered by their insert date (newest first).
    *   **Filtering and Pagination:**
        *   Select a "Filter by" criterion (e.g., "Text Content", "ID") from the dropdown and enter a "Search term".
        *   Choose the number of "Embeddings per page" from the dropdown.
        *   Click the "Apply Filters & Pagination" button to refresh the displayed embeddings based on your selections.
    *   **Navigation:** Navigate through pages using the "Prev" and "Next" buttons, or click directly on page numbers for fast skipping.
    *   **Viewing Embeddings:** Each embedding is displayed in a card-like format, showing its ID, text, original text ID, and insert date. Selected embedding cards will change color.
    *   **Deletion:**
        *   You can delete individual embeddings using the "Delete" button next to each entry.
        *   To perform a batch deletion, select multiple embeddings using the checkboxes (selected cards will change color) and then click "Delete Selected Embeddings" button.
5.  **Retrieve Similar Text:**
    *   Enter a query into the "Enter query text to find similar entries:" text area.
    *   Click "Retrieve Similar" to find and display text chunks from *your* stored documents that are semantically similar to the query.
6.  **Logout:** Click the "Logout" button in the sidebar to end your session.

## Important Notes

*   Ensure Ollama and Pinecone Local are running before starting the Streamlit application.
*   All configuration parameters are loaded from the `.env` file. Make sure it's correctly set up.
*   User credentials (username, hashed password, user ID) are now stored in a dedicated Pinecone index (`user-index`). For a production environment, a more robust database solution with advanced security features would be recommended.
*   All stored RAG embeddings are filtered by the logged-in `user_id`, ensuring that users only interact with their own data.
