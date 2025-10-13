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
    *   Filtering embeddings by text content, ID, or insert date.
    *   Pagination to browse through large sets of embeddings.
    *   Batch deletion of selected embeddings with confirmation buttons at both the top and bottom of the list.
    *   Individual deletion of embeddings directly next to each entry.
    *   Improved visual styling for embedding entries.

## Project Structure

*   `app.py`: The main Streamlit application, handling UI, session management, and orchestrating calls to utility functions.
*   `utils.py`: Contains utility functions for user management (loading/saving users, password hashing) and Ollama embedding generation.
*   `pinecone_utils.py`: Encapsulates Pinecone initialization and interaction logic.
*   `requirements.txt`: Lists Python dependencies.
*   `users.json`: Stores user credentials (username, hashed password, user ID).
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
    *   Use the "Search by text content or ID" input to filter your embeddings by text, ID, or insert date.
    *   Adjust the "Embeddings per page" slider to control the number of entries displayed.
    *   Navigate through pages using the "Previous Page" and "Next Page" buttons.
    *   Each embedding is displayed in a card-like format, showing its ID, text, original text ID, and insert date.
    *   You can delete individual embeddings using the "Delete" button next to each entry.
    *   To perform a batch deletion, select multiple embeddings using the checkboxes and then click either the "Delete Selected Embeddings (Top)" or "Delete Selected Embeddings (Bottom)" button.
5.  **Retrieve Similar Text:**
    *   Enter a query into the "Enter query text to find similar entries:" text area.
    *   Click "Retrieve Similar" to find and display text chunks from *your* stored documents that are semantically similar to the query.
6.  **Logout:** Click the "Logout" button in the sidebar to end your session.

## Important Notes

*   Ensure Ollama and Pinecone Local are running before starting the Streamlit application.
*   The `users.json` file stores user credentials. For a production environment, a more robust database solution would be recommended.
*   All stored embeddings are filtered by the logged-in `user_id`, ensuring that users only interact with their own data.
