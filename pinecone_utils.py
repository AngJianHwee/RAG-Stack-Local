import streamlit as st
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec

# Configuration for Pinecone Local
PINECONE_API_KEY = "pclocal" # Pinecone local doesn't strictly need a key, but the client requires it
PINECONE_HOST = "http://localhost:5081"
INDEX_NAME = "index1"
DIMENSION = 384 # Dimension for all-minilm:33m

def initialize_pinecone():
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
        return index
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {e}")
        st.stop()
        return None
