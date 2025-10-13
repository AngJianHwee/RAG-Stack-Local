import streamlit as st
import requests
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
import time

# Configuration for Ollama and Pinecone Local
OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
PINECONE_API_KEY = "pclocal" # Pinecone local doesn't strictly need a key, but the client requires it
PINECONE_HOST = "http://localhost:5081"
INDEX_NAME = "index1"
DIMENSION = 384 # Dimension for all-minilm:33m

# Initialize Pinecone
try:
    pc = PineconeGRPC(api_key=PINECONE_API_KEY, host=PINECONE_HOST, ssl_verify=False)
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
        )
    index = pc.Index(INDEX_NAME)
    st.success(f"Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    st.error(f"Error connecting to Pinecone: {e}")
    st.stop()

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

st.title("Streamlit RAG with Ollama and Pinecone Local")

user_text = st.text_area("Enter text to embed and store:", height=150)

if st.button("Store Embedding"):
    if user_text:
        with st.spinner("Getting embedding from Ollama..."):
            embedding = get_ollama_embedding(user_text)
        
        if embedding:
            unique_id = str(time.time()) # Simple unique ID
            try:
                index.upsert(vectors=[{"id": unique_id, "values": embedding, "metadata": {"text": user_text}}])
                st.success(f"Text embedded and stored in Pinecone with ID: {unique_id}")
            except Exception as e:
                st.error(f"Error storing embedding in Pinecone: {e}")
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
                    include_metadata=True
                )
                st.write("Similar entries found:")
                for match in results.matches:
                    st.write(f"- **Score:** {match.score:.2f}, **Text:** {match.metadata['text']}")
            except Exception as e:
                st.error(f"Error retrieving similar embeddings from Pinecone: {e}")
    else:
        st.warning("Please enter some query text.")
