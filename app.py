import streamlit as st
import requests
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
                unique_id = f"{str(time.time())}-{i}" # Unique ID for each chunk
                try:
                    index.upsert(vectors=[{"id": unique_id, "values": embedding, "metadata": {"text": chunk, "original_text_id": str(time.time())}}])
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
                    include_metadata=True
                )
                st.write("Similar entries found:")
                for match in results.matches:
                    st.write(f"- **Score:** {match.score:.2f}, **Text:** {match.metadata['text']}")
            except Exception as e:
                st.error(f"Error retrieving similar embeddings from Pinecone: {e}")
    else:
        st.warning("Please enter some query text.")
