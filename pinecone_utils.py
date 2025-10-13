import streamlit as st
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Configuration for Pinecone Local
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
RAG_INDEX_NAME = os.getenv("RAG_INDEX_NAME")
USER_INDEX_NAME = os.getenv("USER_INDEX_NAME")
DIMENSION = int(os.getenv("DIMENSION", 384)) # Dimension for all-minilm:33m, with default

def initialize_pinecone_rag_index():
    try:
        pc = PineconeGRPC(api_key=PINECONE_API_KEY, host=PINECONE_HOST, ssl_verify=False)
        if not pc.has_index(RAG_INDEX_NAME):
            pc.create_index(
                name=RAG_INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                )
            )
        index = pc.Index(RAG_INDEX_NAME)
        st.success(f"Connected to Pinecone RAG index: {RAG_INDEX_NAME}")
        return index
    except Exception as e:
        st.error(f"Error connecting to Pinecone RAG index: {e}")
        st.stop()
        return None

def initialize_pinecone_user_index():
    try:
        pc = PineconeGRPC(api_key=PINECONE_API_KEY, host=PINECONE_HOST, ssl_verify=False)
        if not pc.has_index(USER_INDEX_NAME):
            pc.create_index(
                name=USER_INDEX_NAME,
                dimension=DIMENSION, # Using same dimension for dummy vectors
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                )
            )
        index = pc.Index(USER_INDEX_NAME)
        st.success(f"Connected to Pinecone User index: {USER_INDEX_NAME}")
        return index
    except Exception as e:
        st.error(f"Error connecting to Pinecone User index: {e}")
        st.stop()
        return None

def add_user_to_pinecone_index(user_index, username, hashed_password, user_id):
    try:
        # Store user data with a dummy vector, actual data in metadata
        user_index.upsert(vectors=[
            {"id": user_id, "values": [0.0] * DIMENSION, "metadata": {"username": username, "password": hashed_password, "user_id": user_id}}
        ])
        return True
    except Exception as e:
        st.error(f"Error adding user to Pinecone: {e}")
        return False

def get_user_from_pinecone_index(user_index, username):
    try:
        # Query with a filter to find the user by username
        results = user_index.query(
            vector=[0.0] * DIMENSION, # Dummy vector
            top_k=1,
            include_metadata=True,
            filter={"username": username}
        )
        if results.matches:
            return results.matches[0].metadata
        return None
    except Exception as e:
        st.error(f"Error retrieving user from Pinecone: {e}")
        return None

def get_all_users_from_pinecone_index(user_index):
    try:
        # To get all users, query with a dummy vector and a high top_k
        results = user_index.query(
            vector=[0.0] * DIMENSION,
            top_k=1000, # Assuming max 1000 users for now
            include_metadata=True
        )
        return [match.metadata for match in results.matches]
    except Exception as e:
        st.error(f"Error retrieving all users from Pinecone: {e}")
        return []

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
        # The IDs passed here are already filtered by user_id from get_user_embeddings.
        # Pinecone's delete operation does not allow explicit IDs and a filter simultaneously.
        # Therefore, we only pass the IDs.
        index.delete(ids=ids)
        st.success(f"Successfully deleted {len(ids)} embeddings for user {user_id}.")
        return True
    except Exception as e:
        st.error(f"Error deleting embeddings from Pinecone: {e}")
        return False

def get_user_rag_stats(index, user_id):
    try:
        # Query to get all embeddings for a specific user
        results = index.query(
            vector=[0.0] * DIMENSION, # Dummy vector
            top_k=10000, # A sufficiently large number to retrieve all
            include_metadata=True,
            filter={"user_id": user_id}
        )
        
        total_chunks = len(results.matches)
        
        unique_original_text_ids = set()
        for match in results.matches:
            if "original_text_id" in match.metadata:
                unique_original_text_ids.add(match.metadata["original_text_id"])
        
        total_documents = len(unique_original_text_ids)
        
        return {"total_documents": total_documents, "total_chunks": total_chunks}
    except Exception as e:
        st.error(f"Error retrieving RAG statistics for user {user_id} from Pinecone: {e}")
        return {"total_documents": 0, "total_chunks": 0}
