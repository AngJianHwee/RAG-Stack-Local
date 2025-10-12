import streamlit as st
import requests
import os
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec

# Environment variables for service hosts
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PINECONE_HOST = os.getenv("PINECONE_HOST", "http://localhost:5081")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pclocal") # Default to 'pclocal' as per notes

st.title("Streamlit RAG with Ollama and Pinecone Local")

# --- Ollama Integration ---
st.header("Ollama Embeddings")
ollama_model = st.text_input("Ollama Model (e.g., all-minilm:33m)", "all-minilm:33m")
text_to_embed = st.text_area("Text to embed with Ollama", "The quick brown fox jumps over the lazy dog.")

if st.button("Generate Embeddings"):
    try:
        data = {
            "model": ollama_model,
            "input": [text_to_embed]
        }
        response = requests.post(f"{OLLAMA_HOST}/api/embed", json=data)
        response.raise_for_status() # Raise an exception for HTTP errors
        embeddings = response.json()
        st.json(embeddings)
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to Ollama at {OLLAMA_HOST}. Is the service running and accessible?")
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating embeddings: {e}")

# --- Pinecone Integration ---
st.header("Pinecone Local Vector Database")

@st.cache_resource
def get_pinecone_client():
    try:
        pc = PineconeGRPC(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
        return pc
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {e}")
        return None

pc = get_pinecone_client()
index_name = "my-rag-index"
dimension = 384 # all-minilm:33m has 384 dimensions

if pc:
    if st.button(f"Create Pinecone Index '{index_name}'"):
        if not pc.has_index(index_name):
            try:
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1",
                    )
                )
                st.success(f"Index '{index_name}' created successfully!")
            except Exception as e:
                st.error(f"Error creating index: {e}")
        else:
            st.info(f"Index '{index_name}' already exists.")

    if pc.has_index(index_name):
        index = pc.Index(index_name)
        st.subheader("Upsert and Query")
        upsert_text = st.text_area("Text to upsert into Pinecone", "This is a document about Streamlit.")
        query_text = st.text_area("Text to query Pinecone with", "What is Streamlit?")

        if st.button("Upsert and Query"):
            try:
                # Generate embedding for upsert text
                data_upsert = {
                    "model": ollama_model,
                    "input": [upsert_text]
                }
                response_upsert = requests.post(f"{OLLAMA_HOST}/api/embed", json=data_upsert)
                response_upsert.raise_for_status()
                embedding_upsert = response_upsert.json()["embeddings"][0]

                # Upsert into Pinecone
                index.upsert(vectors=[{"id": "doc1", "values": embedding_upsert}])
                st.success("Document upserted successfully!")

                # Generate embedding for query text
                data_query = {
                    "model": ollama_model,
                    "input": [query_text]
                }
                response_query = requests.post(f"{OLLAMA_HOST}/api/embed", json=data_query)
                response_query.raise_for_status()
                embedding_query = response_query.json()["embeddings"][0]

                # Query Pinecone
                query_results = index.query(vector=embedding_query, top_k=3, include_values=False)
                st.subheader("Query Results:")
                st.json(query_results)

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to Ollama at {OLLAMA_HOST}. Is the service running and accessible?")
            except requests.exceptions.RequestException as e:
                st.error(f"Error with Ollama or Pinecone during upsert/query: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning(f"Pinecone index '{index_name}' does not exist. Please create it first.")
