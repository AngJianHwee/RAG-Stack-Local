import streamlit as st
import requests
from pinecone import Pinecone, Index, PodSpec
import time

# Configuration
OLLAMA_URL = "http://ollama:11434/api/embeddings"
PINECONE_API_KEY = "YOUR_API_KEY" # Pinecone local doesn't strictly need this, but the client requires it
PINECONE_HOST = "pinecone-local" # This will be the service name in docker-compose
PINECONE_ENVIRONMENT = "local" # Placeholder for local
INDEX_NAME = "rag-index"
MODEL_NAME = "all-minilm:33m"

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
        if INDEX_NAME not in pc.list_indexes():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384, # Dimension for all-minilm:33m
                metric='cosine',
                spec=PodSpec(environment=PINECONE_ENVIRONMENT)
            )
        return pc.Index(INDEX_NAME)
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        st.stop()

pinecone_index = init_pinecone()

# Function to get embeddings from Ollama
def get_embedding(text):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": text},
            timeout=60 # Increased timeout for Ollama
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama. Make sure the Ollama service is running.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting embedding from Ollama: {e}")
        return None

st.title("Streamlit RAG with Ollama and Pinecone Local")

user_text = st.text_area("Enter text here:", height=150)

if st.button("Store Text and Embeddings"):
    if user_text:
        with st.spinner("Getting embedding from Ollama..."):
            embedding = get_embedding(user_text)

        if embedding:
            try:
                # Generate a unique ID for the entry
                vector_id = f"doc-{int(time.time())}"
                pinecone_index.upsert(
                    vectors=[{"id": vector_id, "values": embedding, "metadata": {"text": user_text}}]
                )
                st.success(f"Text stored successfully with ID: {vector_id}")
            except Exception as e:
                st.error(f"Error storing embedding in Pinecone: {e}")
        else:
            st.warning("Could not get embedding. Please check Ollama service.")
    else:
        st.warning("Please enter some text to store.")

st.markdown("---")

st.header("Retrieve Similar Text")
query_text = st.text_input("Enter query text to retrieve similar documents:")

if st.button("Retrieve Similar Text"):
    if query_text:
        with st.spinner("Getting query embedding from Ollama..."):
            query_embedding = get_embedding(query_text)

        if query_embedding:
            try:
                with st.spinner("Searching Pinecone..."):
                    search_results = pinecone_index.query(
                        vector=query_embedding,
                        top_k=5,
                        include_metadata=True
                    )
                
                if search_results.matches:
                    st.subheader("Most Similar Documents:")
                    for i, match in enumerate(search_results.matches):
                        st.write(f"**Match {i+1} (Score: {match.score:.4f}):**")
                        st.write(match.metadata['text'])
                else:
                    st.info("No similar documents found.")
            except Exception as e:
                st.error(f"Error retrieving from Pinecone: {e}")
        else:
            st.warning("Could not get query embedding. Please check Ollama service.")
    else:
        st.warning("Please enter some text to query.")
