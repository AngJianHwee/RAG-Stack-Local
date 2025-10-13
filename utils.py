import json
import bcrypt
import requests
import streamlit as st # Streamlit is needed for st.error in get_ollama_embedding

USERS_FILE = "users.json"
OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"

# --- User Management Functions ---
def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)["users"]
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump({"users": users}, f, indent=4)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

# --- Ollama Embedding Function ---
def get_ollama_embedding(text):
    try:
        response = requests.post(
            OLLAMA_EMBEDDING_URL,
            json={"model": "all-minilm:33m", "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama. Make sure the Ollama service is running and accessible at 'http://ollama:11434'.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting embedding from Ollama: {e}")
        return None
