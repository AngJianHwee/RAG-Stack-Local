import pytest
from unittest.mock import patch, MagicMock
import sys
import os
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(dotenv_path='tests/.env.test', override=True)

# Mock the Streamlit st object
st = MagicMock()
sys.modules['streamlit'] = st

# Add the parent directory to the sys.path to allow importing pinecone_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinecone_utils import (
    initialize_pinecone_rag_index, initialize_pinecone_user_index,
    add_user_to_pinecone_index, get_user_from_pinecone_index,
    get_all_users_from_pinecone_index, get_user_embeddings, delete_embeddings,
    DIMENSION, RAG_INDEX_NAME, USER_INDEX_NAME
)

@pytest.fixture
def mock_pinecone_grpc():
    with patch('pinecone_utils.PineconeGRPC') as mock_pc_grpc:
        yield mock_pc_grpc

@pytest.fixture
def mock_pinecone_index():
    return MagicMock()

# Test initialize_pinecone_rag_index
def test_initialize_pinecone_rag_index_exists(mock_pinecone_grpc, mock_pinecone_index):
    mock_pc_instance = mock_pinecone_grpc.return_value
    mock_pc_instance.has_index.return_value = True
    mock_pc_instance.Index.return_value = mock_pinecone_index

    index = initialize_pinecone_rag_index()
    assert index == mock_pinecone_index
    mock_pc_instance.has_index.assert_called_once_with(RAG_INDEX_NAME)
    mock_pc_instance.create_index.assert_not_called()
    st.success.assert_called_once_with(f"Connected to Pinecone RAG index: {RAG_INDEX_NAME}")

def test_initialize_pinecone_rag_index_creates(mock_pinecone_grpc, mock_pinecone_index):
    mock_pc_instance = mock_pinecone_grpc.return_value
    mock_pc_instance.has_index.return_value = False
    mock_pc_instance.Index.return_value = mock_pinecone_index

    index = initialize_pinecone_rag_index()
    assert index == mock_pinecone_index
    mock_pc_instance.has_index.assert_called_once_with(RAG_INDEX_NAME)
    mock_pc_instance.create_index.assert_called_once()
    st.success.assert_called_once_with(f"Connected to Pinecone RAG index: {RAG_INDEX_NAME}")

def test_initialize_pinecone_rag_index_error(mock_pinecone_grpc):
    mock_pc_instance = mock_pinecone_grpc.return_value
    mock_pc_instance.has_index.side_effect = Exception("Connection error")

    index = initialize_pinecone_rag_index()
    assert index is None
    st.error.assert_called_once_with("Error connecting to Pinecone RAG index: Connection error")

# Test initialize_pinecone_user_index
def test_initialize_pinecone_user_index_exists(mock_pinecone_grpc, mock_pinecone_index):
    mock_pc_instance = mock_pinecone_grpc.return_value
    mock_pc_instance.has_index.return_value = True
    mock_pc_instance.Index.return_value = mock_pinecone_index

    index = initialize_pinecone_user_index()
    assert index == mock_pinecone_index
    mock_pc_instance.has_index.assert_called_once_with(USER_INDEX_NAME)
    mock_pc_instance.create_index.assert_not_called()
    st.success.assert_called_once_with(f"Connected to Pinecone User index: {USER_INDEX_NAME}")

# Test add_user_to_pinecone_index
def test_add_user_to_pinecone_index_success(mock_pinecone_index):
    mock_pinecone_index.upsert.return_value = None # upsert doesn't return anything specific
    result = add_user_to_pinecone_index(mock_pinecone_index, "testuser", "hashed_pw", "1")
    assert result is True
    mock_pinecone_index.upsert.assert_called_once()
    st.error.assert_not_called()

def test_add_user_to_pinecone_index_error(mock_pinecone_index):
    mock_pinecone_index.upsert.side_effect = Exception("Upsert error")
    result = add_user_to_pinecone_index(mock_pinecone_index, "testuser", "hashed_pw", "1")
    assert result is False
    st.error.assert_called_once_with("Error adding user to Pinecone: Upsert error")

# Test get_user_from_pinecone_index
def test_get_user_from_pinecone_index_found(mock_pinecone_index):
    mock_query_response = MagicMock()
    mock_match = MagicMock()
    mock_match.metadata = {"username": "testuser", "password": "hashed_pw", "user_id": "1"}
    mock_query_response.matches = [mock_match]
    mock_pinecone_index.query.return_value = mock_query_response

    user_data = get_user_from_pinecone_index(mock_pinecone_index, "testuser")
    assert user_data == {"username": "testuser", "password": "hashed_pw", "user_id": "1"}
    mock_pinecone_index.query.assert_called_once()

def test_get_user_from_pinecone_index_not_found(mock_pinecone_index):
    mock_query_response = MagicMock()
    mock_query_response.matches = []
    mock_pinecone_index.query.return_value = mock_query_response

    user_data = get_user_from_pinecone_index(mock_pinecone_index, "nonexistent")
    assert user_data is None
    mock_pinecone_index.query.assert_called_once()

# Test get_all_users_from_pinecone_index
def test_get_all_users_from_pinecone_index_success(mock_pinecone_index):
    mock_query_response = MagicMock()
    mock_match1 = MagicMock()
    mock_match1.metadata = {"username": "user1", "user_id": "1"}
    mock_match2 = MagicMock()
    mock_match2.metadata = {"username": "user2", "user_id": "2"}
    mock_query_response.matches = [mock_match1, mock_match2]
    mock_pinecone_index.query.return_value = mock_query_response

    users = get_all_users_from_pinecone_index(mock_pinecone_index)
    assert users == [{"username": "user1", "user_id": "1"}, {"username": "user2", "user_id": "2"}]
    mock_pinecone_index.query.assert_called_once()

# Test get_user_embeddings
def test_get_user_embeddings_success(mock_pinecone_index):
    mock_query_response = MagicMock()
    mock_match1 = MagicMock()
    mock_match1.id = "id1"
    mock_match1.metadata = {"user_id": "1", "text": "text1"}
    mock_match2 = MagicMock()
    mock_match2.id = "id2"
    mock_match2.metadata = {"user_id": "1", "text": "text2"}
    mock_query_response.matches = [mock_match1, mock_match2]
    mock_pinecone_index.query.return_value = mock_query_response

    embeddings = get_user_embeddings(mock_pinecone_index, "1")
    assert len(embeddings) == 2
    assert embeddings[0].id == "id1"
    assert embeddings[1].id == "id2"
    mock_pinecone_index.query.assert_called_once()

# Test delete_embeddings
def test_delete_embeddings_success(mock_pinecone_index):
    mock_pinecone_index.delete.return_value = None
    result = delete_embeddings(mock_pinecone_index, ["id1", "id2"], "1")
    assert result is True
    mock_pinecone_index.delete.assert_called_once_with(ids=["id1", "id2"])
    st.success.assert_called_once_with("Successfully deleted 2 embeddings for user 1.")

def test_delete_embeddings_error(mock_pinecone_index):
    mock_pinecone_index.delete.side_effect = Exception("Delete error")
    result = delete_embeddings(mock_pinecone_index, ["id1"], "1")
    assert result is False
    st.error.assert_called_once_with("Error deleting embeddings from Pinecone: Delete error")
