import bcrypt
import requests
import streamlit as st # Streamlit is needed for st.error in get_ollama_embedding
import os
from dotenv import load_dotenv
from pinecone_utils import initialize_pinecone_user_index, add_user_to_pinecone_index, get_user_from_pinecone_index, get_all_users_from_pinecone_index

load_dotenv() # Load environment variables from .env file

OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL", "http://localhost:11434/api/embeddings")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm:33m") # New environment variable for model selection

# Initialize Pinecone User Index
user_index = initialize_pinecone_user_index()

# --- User Management Functions (Pinecone-based) ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_next_user_id():
    if user_index is None:
        return "1" # Fallback if index not initialized
    all_users = get_all_users_from_pinecone_index(user_index)
    if not all_users:
        return "1"
    # Find the maximum user_id and increment it
    max_id = 0
    for user_data in all_users:
        try:
            max_id = max(max_id, int(user_data.get("user_id", 0)))
        except ValueError:
            continue # Skip invalid user_ids
    return str(max_id + 1)

def add_user(username, password):
    if user_index is None:
        st.error("Pinecone user index not initialized. Cannot add user.")
        return False
    
    # Check if username already exists
    existing_user = get_user_by_username(username)
    if existing_user:
        st.error("Username already exists.")
        return False

    hashed_pw = hash_password(password)
    new_user_id = get_next_user_id()
    return add_user_to_pinecone_index(user_index, username, hashed_pw, new_user_id)

def get_user_by_username(username):
    if user_index is None:
        st.error("Pinecone user index not initialized. Cannot retrieve user.")
        return None
    return get_user_from_pinecone_index(user_index, username)

# --- Ollama Embedding Function ---
def get_ollama_embedding(text):
    try:
        response = requests.post(
            OLLAMA_EMBEDDING_URL,
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama. Make sure the Ollama service is running and accessible at 'http://ollama:11434'.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting embedding from Ollama: {e}")
        return None
