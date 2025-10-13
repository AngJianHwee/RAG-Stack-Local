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

def get_user_embeddings(index, user_id):
    try:
        # Pinecone's query method is primarily for similarity search.
        # To retrieve all vectors for a user, we can query with a dummy vector
        # and a high top_k, filtered by user_id.
        # This might not be efficient for a very large number of embeddings,
        # but works for typical admin page scenarios.
        results = index.query(
            vector=[0.0] * DIMENSION, # Dummy vector
            top_k=10000, # A sufficiently large number to retrieve all (or most)
            include_metadata=True,
            filter={"user_id": user_id}
        )
        return results.matches
    except Exception as e:
        st.error(f"Error retrieving user embeddings from Pinecone: {e}")
        return []

def delete_embeddings(index, ids, user_id):
    try:
        # Ensure deletion is scoped to the user_id for security
        index.delete(ids=ids, filter={"user_id": user_id})
        st.success(f"Successfully deleted {len(ids)} embeddings for user {user_id}.")
        return True
    except Exception as e:
        st.error(f"Error deleting embeddings from Pinecone: {e}")
        return False
