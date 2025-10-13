import streamlit as st
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import load_users, save_users, hash_password, check_password, get_ollama_embedding
from pinecone_utils import initialize_pinecone, get_user_embeddings, delete_embeddings

# Initialize Pinecone
index = initialize_pinecone()

if "selected_embeddings" not in st.session_state:
    st.session_state["selected_embeddings"] = []

# --- Streamlit Pages ---
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()
        user_found = False
        for user in users:
            if user["username"] == username and check_password(password, user["password"]):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["user_id"] = user["user_id"] # Store user_id in session state
                user_found = True
                st.rerun()
        if not user_found:
            st.error("Invalid username or password")

    st.subheader("Register New User")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password", key="new_password")
    
    if st.button("Register"):
        if new_username and new_password:
            users = load_users()
            if any(user["username"] == new_username for user in users):
                st.error("Username already exists.")
            else:
                hashed_pw = hash_password(new_password)
                new_user_id = str(len(users) + 1) # Simple incremental ID
                users.append({"username": new_username, "password": hashed_pw, "user_id": new_user_id})
                save_users(users)
                st.success("User registered successfully! Please log in.")
        else:
            st.error("Please enter both username and password for registration.")

def set_page(page_name):
    st.session_state["page"] = page_name

def main_page():
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    
    if st.sidebar.button("Home"):
        set_page("main")
    if st.sidebar.button("Admin Page"):
        set_page("admin")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["user_id"] = None
        st.session_state["page"] = "main" # Reset page on logout
        st.rerun()

    st.title("Streamlit RAG with Ollama and Pinecone Local")

    if index is None:
        st.error("Pinecone index not initialized. Please check the connection.")
        return

    # Chunking options
    st.sidebar.header("Chunking Options")
    chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=50)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=chunk_size - 1, value=50, step=10)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    user_text = st.text_area("Enter text to embed and store:", height=150)

    if st.button("Store Embedding"):
        if user_text:
            chunks = text_splitter.split_text(user_text)
            st.info(f"Text split into {len(chunks)} chunks.")

            for i, chunk in enumerate(chunks):
                with st.spinner(f"Getting embedding for chunk {i+1}/{len(chunks)} from Ollama..."):
                    embedding = get_ollama_embedding(chunk)
                
                if embedding:
                    unique_id = f"{st.session_state['user_id']}-{str(time.time())}-{i}" # Unique ID for each chunk, prefixed with user_id
                    try:
                        index.upsert(vectors=[{"id": unique_id, "values": embedding, "metadata": {"text": chunk, "original_text_id": str(time.time()), "user_id": st.session_state["user_id"]}}])
                        st.success(f"Chunk {i+1} embedded and stored in Pinecone with ID: {unique_id}")
                    except Exception as e:
                        st.error(f"Error storing embedding for chunk {i+1} in Pinecone: {e}")
                else:
                    st.error(f"Could not get embedding for chunk {i+1}. Skipping.")
        else:
            st.warning("Please enter some text to store.")

    st.subheader("Retrieve Similar Text")
    query_text = st.text_area("Enter query text to find similar entries:", height=100)

    if st.button("Retrieve Similar"):
        if query_text:
            with st.spinner("Getting query embedding from Ollama..."):
                query_embedding = get_ollama_embedding(query_text)
            
            if query_embedding:
                try:
                    results = index.query(
                        vector=query_embedding,
                        top_k=5,
                        include_metadata=True,
                        filter={"user_id": st.session_state["user_id"]} # Filter by user ID
                    )
                    st.write("Similar entries found:")
                    for match in results.matches:
                        st.write(f"- **Score:** {match.score:.2f}, **Text:** {match.metadata['text']}")
                except Exception as e:
                    st.error(f"Error retrieving similar embeddings from Pinecone: {e}")
        else:
            st.warning("Please enter some query text.")

def admin_page():
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Home"):
        set_page("main")
    if st.sidebar.button("Admin Page"):
        set_page("admin")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["user_id"] = None
        st.session_state["page"] = "main" # Reset page on logout
        st.rerun()

    st.title("Admin Page - Your Embeddings")

    if index is None:
        st.error("Pinecone index not initialized. Please check the connection.")
        return

    user_id = st.session_state["user_id"]
    
    st.subheader("Filter Embeddings")
    search_term = st.text_input("Search by text content or ID:")

    st.subheader("Your Stored Embeddings")

    embeddings = get_user_embeddings(index, user_id)
    
    if not embeddings:
        st.info("No embeddings found for your account.")
        return

    # Filter embeddings based on search_term
    filtered_embeddings = []
    if search_term:
        search_term_lower = search_term.lower()
        for match in embeddings:
            text_content = match.metadata.get("text", "").lower()
            original_text_id = match.metadata.get("original_text_id", "").lower()
            if search_term_lower in text_content or search_term_lower in original_text_id or search_term_lower in match.id.lower():
                filtered_embeddings.append(match)
    else:
        filtered_embeddings = embeddings

    if not filtered_embeddings:
        st.info("No embeddings match your search criteria.")
        return

    # Display embeddings with checkboxes for deletion
    st.write(f"Displaying {len(filtered_embeddings)} of {len(embeddings)} embeddings.")
    
    # Create a form for batch deletion
    with st.form("delete_embeddings_form"):
        selected_for_deletion = []
        for i, match in enumerate(filtered_embeddings):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                checkbox_key = f"checkbox_{match.id}"
                if st.checkbox("", key=checkbox_key):
                    selected_for_deletion.append(match.id)
            with col2:
                st.markdown(f"**ID:** `{match.id}`")
                st.markdown(f"**Text:** {match.metadata.get('text', 'N/A')}")
                st.markdown(f"**Original Text ID:** `{match.metadata.get('original_text_id', 'N/A')}`")
                st.markdown("---")
        
        st.session_state["selected_embeddings"] = selected_for_deletion

        delete_button = st.form_submit_button("Delete Selected Embeddings")

        if delete_button:
            if st.session_state["selected_embeddings"]:
                with st.spinner("Deleting selected embeddings..."):
                    if delete_embeddings(index, st.session_state["selected_embeddings"], user_id):
                        st.success("Selected embeddings deleted successfully!")
                        st.session_state["selected_embeddings"] = [] # Clear selection
                        st.rerun() # Rerun to refresh the list
                    else:
                        st.error("Failed to delete embeddings.")
            else:
                st.warning("No embeddings selected for deletion.")

# --- Main App Logic ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["user_id"] = None
    st.session_state["page"] = "main" # Initialize page state

if st.session_state["logged_in"]:
    if st.session_state["page"] == "main":
        main_page()
    elif st.session_state["page"] == "admin":
        admin_page() # This function will be defined next
else:
    login_page()
