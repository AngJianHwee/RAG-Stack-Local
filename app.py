import streamlit as st
import time
from datetime import datetime # Import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import hash_password, check_password, get_ollama_embedding, add_user, get_user_by_username
from pinecone_utils import initialize_pinecone_rag_index, get_user_embeddings, delete_embeddings

# Initialize Pinecone RAG Index
rag_index = initialize_pinecone_rag_index()
# The user_index is initialized in utils.py

if "selected_embeddings" not in st.session_state:
    st.session_state["selected_embeddings"] = []

# --- Streamlit Pages ---
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user_data = get_user_by_username(username)
        if user_data and check_password(password, user_data["password"]):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = user_data["user_id"] # Store user_id in session state
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.subheader("Register New User")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password", key="new_password")
    
    if st.button("Register"):
        if new_username and new_password:
            if add_user(new_username, new_password):
                st.success("User registered successfully! Logging in...")
                # Auto-login the newly registered user
                user_data = get_user_by_username(new_username)
                if user_data:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = new_username
                    st.session_state["user_id"] = user_data["user_id"]
                    st.rerun()
                else:
                    st.error("Error retrieving user data after registration. Please try logging in manually.")
            # Error messages are handled within add_user
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

    if rag_index is None:
        st.error("Pinecone RAG index not initialized. Please check the connection.")
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
                    current_time = datetime.now().isoformat() # Get current time for insert date
                    try:
                        rag_index.upsert(vectors=[{"id": unique_id, "values": embedding, "metadata": {"text": chunk, "original_text_id": str(time.time()), "user_id": st.session_state["user_id"], "insert_date": current_time}}])
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
                    results = rag_index.query(
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

    if rag_index is None:
        st.error("Pinecone RAG index not initialized. Please check the connection.")
        return

    user_id = st.session_state["user_id"]
    
    st.subheader("Manage Your Stored Embeddings")

    embeddings = get_user_embeddings(rag_index, user_id)
    
    if not embeddings:
        st.info("No embeddings found for your account.")
        return

    # Sort embeddings by insert_date (newest first)
    embeddings.sort(key=lambda x: x.metadata.get("insert_date", ""), reverse=True)

    # Initialize session state for filters and pagination if not present
    if "filter_criteria" not in st.session_state:
        st.session_state["filter_criteria"] = "Text Content"
    if "search_term" not in st.session_state:
        st.session_state["search_term"] = ""
    if "items_per_page" not in st.session_state:
        st.session_state["items_per_page"] = 10
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 1
    
    # Initialize session state for messages
    if "delete_message" not in st.session_state:
        st.session_state["delete_message"] = {"type": None, "content": None}

    # Display persistent message if available
    if st.session_state["delete_message"]["type"] == "success":
        st.success(st.session_state["delete_message"]["content"])
        st.session_state["delete_message"] = {"type": None, "content": None} # Clear after display
    elif st.session_state["delete_message"]["type"] == "error":
        st.error(st.session_state["delete_message"]["content"])
        st.session_state["delete_message"] = {"type": None, "content": None} # Clear after display
    elif st.session_state["delete_message"]["type"] == "warning":
        st.warning(st.session_state["delete_message"]["content"])
        st.session_state["delete_message"] = {"type": None, "content": None} # Clear after display

    # Filter and pagination controls within a form
    with st.form("filter_pagination_form"):
        col_filter_type, col_search_term = st.columns([0.4, 0.6])
        with col_filter_type:
            st.session_state["filter_criteria"] = st.selectbox(
                "Filter by",
                options=["Text Content", "ID", "Original Text ID", "Insert Date"],
                key="filter_criteria_select"
            )
        with col_search_term:
            st.session_state["search_term"] = st.text_input(
                "Search term",
                value=st.session_state["search_term"],
                key="search_term_input"
            )
        
        st.session_state["items_per_page"] = st.selectbox(
            "Embeddings per page",
            options=[5, 10, 20, 50],
            index=[5, 10, 20, 50].index(st.session_state["items_per_page"]),
            key="items_per_page_select"
        )

        update_button = st.form_submit_button("Apply Filters & Pagination")

    # Apply filters and pagination only when update_button is clicked or on initial load
    if update_button:
        st.session_state["current_page"] = 1 # Reset page on filter/pagination change
        # Filtering logic
        filtered_embeddings = []
        search_term_lower = st.session_state["search_term"].lower()
        for match in embeddings:
            metadata_value = ""
            if st.session_state["filter_criteria"] == "Text Content":
                metadata_value = match.metadata.get("text", "").lower()
            elif st.session_state["filter_criteria"] == "ID":
                metadata_value = match.id.lower()
            elif st.session_state["filter_criteria"] == "Original Text ID":
                metadata_value = match.metadata.get("original_text_id", "").lower()
            elif st.session_state["filter_criteria"] == "Insert Date":
                metadata_value = match.metadata.get("insert_date", "").lower()
            
            if search_term_lower in metadata_value:
                filtered_embeddings.append(match)
        st.session_state["filtered_embeddings"] = filtered_embeddings
    elif "filtered_embeddings" not in st.session_state or st.session_state["delete_triggered"]: # Re-evaluate if delete was triggered
        st.session_state["filtered_embeddings"] = embeddings # Initial load or after delete
        st.session_state["delete_triggered"] = False # Reset flag
    
    filtered_embeddings = st.session_state["filtered_embeddings"]

    if not filtered_embeddings:
        st.info("No embeddings match your search criteria.")
        return

    total_pages = (len(filtered_embeddings) + st.session_state["items_per_page"] - 1) // st.session_state["items_per_page"]
    
    # Ensure current_page is valid
    if st.session_state["current_page"] > total_pages and total_pages > 0:
        st.session_state["current_page"] = total_pages
    elif st.session_state["current_page"] == 0 and total_pages > 0:
        st.session_state["current_page"] = 1
    elif total_pages == 0:
        st.session_state["current_page"] = 0


    # Pagination controls
    st.write(f"Page {st.session_state['current_page']} of {total_pages}")
    
    # Horizontal pagination buttons
    # Create columns dynamically for page numbers
    page_cols = st.columns([0.1] + [0.05] * min(total_pages, 5) + [0.1]) # Prev, up to 5 page numbers, Next
    
    with page_cols[0]:
        if st.button("Prev", key="prev_page_button", disabled=(st.session_state["current_page"] <= 1)):
            st.session_state["current_page"] -= 1
            st.rerun()
    
    # Individual page number buttons
    page_numbers_to_display = 5 # Number of page buttons to show
    start_page = max(1, st.session_state["current_page"] - page_numbers_to_display // 2)
    end_page = min(total_pages, start_page + page_numbers_to_display - 1)
    
    if end_page - start_page + 1 < page_numbers_to_display:
        start_page = max(1, end_page - page_numbers_to_display + 1)

    for i, p_num in enumerate(range(start_page, end_page + 1)):
        with page_cols[i + 1]: # Offset by 1 for the 'Prev' button column
            if st.button(str(p_num), key=f"page_button_{p_num}", type="primary" if p_num == st.session_state["current_page"] else "secondary"):
                st.session_state["current_page"] = p_num
                st.rerun()

    with page_cols[-1]: # Last column for 'Next' button
        if st.button("Next", key="next_page_button", disabled=(st.session_state["current_page"] >= total_pages)):
            st.session_state["current_page"] += 1
            st.rerun()

    start_idx = (st.session_state["current_page"] - 1) * st.session_state["items_per_page"]
    end_idx = start_idx + st.session_state["items_per_page"]
    paginated_embeddings = filtered_embeddings[start_idx:end_idx]

    st.write(f"Displaying {len(paginated_embeddings)} embeddings on this page ({len(filtered_embeddings)} total filtered, {len(embeddings)} total stored).")
    
    # Batch delete button at the top
    with st.form("delete_embeddings_form_top"):
        if st.form_submit_button("Delete Selected Embeddings (Top)", type="primary"):
            if st.session_state["selected_embeddings"]:
                with st.spinner("Deleting selected embeddings..."):
                    if delete_embeddings(rag_index, st.session_state["selected_embeddings"], user_id):
                        st.session_state["delete_message"] = {"type": "success", "content": "Selected embeddings deleted successfully!"}
                        st.session_state["selected_embeddings"] = [] # Clear selection
                        st.session_state["delete_triggered"] = True # Set flag to force re-evaluation of filtered_embeddings
                        st.rerun() # Rerun to refresh the list
                    else:
                        st.session_state["delete_message"] = {"type": "error", "content": "Failed to delete embeddings."}
                        st.session_state["delete_triggered"] = True # Set flag
                        st.rerun()
            else:
                st.session_state["delete_message"] = {"type": "warning", "content": "No embeddings selected for deletion."}
                st.session_state["delete_triggered"] = True # Set flag
                st.rerun()

    # Display embeddings with checkboxes and individual delete buttons
    for i, match in enumerate(paginated_embeddings):
        # Removed card_style for selected items
        # st.markdown(f"<div style='border: 1px solid #ccc; border-radius: 5px; padding: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
        col_checkbox, col_content, col_delete_individual = st.columns([0.1, 0.7, 0.2])
        with col_checkbox:
            checkbox_key = f"checkbox_{match.id}"
            # Use a callback to update session state directly
            def on_checkbox_change(embedding_id):
                if st.session_state[f"checkbox_{embedding_id}"]:
                    if embedding_id not in st.session_state["selected_embeddings"]:
                        st.session_state["selected_embeddings"].append(embedding_id)
                else:
                    if embedding_id in st.session_state["selected_embeddings"]:
                        st.session_state["selected_embeddings"].remove(embedding_id)
            
            st.checkbox("", key=checkbox_key, value=(match.id in st.session_state["selected_embeddings"]), on_change=on_checkbox_change, args=(match.id,))
        with col_content:
            st.markdown(f"**ID:** `{match.id}`")
            st.markdown(f"**Text:** {match.metadata.get('text', 'N/A')}")
            st.markdown(f"**Original Text ID:** `{match.metadata.get('original_text_id', 'N/A')}`")
            st.markdown(f"**Insert Date:** `{match.metadata.get('insert_date', 'N/A')}`")
        with col_delete_individual:
            if st.button("Delete", key=f"delete_individual_{match.id}", type="primary"):
                with st.spinner(f"Deleting embedding {match.id}..."):
                    if delete_embeddings(rag_index, [match.id], user_id):
                        # Remove from selected_embeddings if it was selected
                        if match.id in st.session_state["selected_embeddings"]:
                            st.session_state["selected_embeddings"].remove(match.id)
                        st.session_state["delete_message"] = {"type": "success", "content": f"Embedding {match.id} deleted successfully!"}
                        st.session_state["delete_triggered"] = True # Set flag
                        st.rerun()
                    else:
                        st.session_state["delete_message"] = {"type": "error", "content": f"Failed to delete embedding {match.id}."}
                        st.session_state["delete_triggered"] = True # Set flag
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True) # Close the div for card style
    
    # No need to reassign selected_embeddings here, it's updated by callbacks

    # Batch delete button at the bottom
    with st.form("delete_embeddings_form_bottom"):
        if st.form_submit_button("Delete Selected Embeddings (Bottom)", type="primary"):
            if st.session_state["selected_embeddings"]:
                with st.spinner("Deleting selected embeddings..."):
                    if delete_embeddings(rag_index, st.session_state["selected_embeddings"], user_id):
                        st.session_state["delete_message"] = {"type": "success", "content": "Selected embeddings deleted successfully!"}
                        st.session_state["selected_embeddings"] = [] # Clear selection
                        st.session_state["delete_triggered"] = True # Set flag
                        st.rerun() # Rerun to refresh the list
                    else:
                        st.session_state["delete_message"] = {"type": "error", "content": "Failed to delete embeddings."}
                        st.session_state["delete_triggered"] = True # Set flag
                        st.rerun()
            else:
                st.session_state["delete_message"] = {"type": "warning", "content": "No embeddings selected for deletion."}
                st.session_state["delete_triggered"] = True # Set flag
                st.rerun()

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
