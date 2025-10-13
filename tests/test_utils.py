import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import requests # Import requests here

# Mock the Streamlit st object to prevent errors during testing
# This is a common pattern when testing Streamlit apps without running the app
st = MagicMock()
sys.modules['streamlit'] = st

# Add the parent directory to the sys.path to allow importing utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import hash_password, check_password, get_ollama_embedding, get_next_user_id, add_user, get_user_by_username
from pinecone_utils import initialize_pinecone_user_index, get_all_users_from_pinecone_index, add_user_to_pinecone_index, get_user_from_pinecone_index

# Mock the user_index from utils.py
@pytest.fixture(autouse=True)
def mock_user_index():
    with patch('utils.user_index', MagicMock()) as mock_index:
        yield mock_index

# Test password hashing functions
def test_hash_password():
    password = "test_password"
    hashed_password = hash_password(password)
    assert isinstance(hashed_password, str)
    assert len(hashed_password) > 0
    assert hashed_password != password

def test_check_password_correct():
    password = "test_password"
    hashed_password = hash_password(password)
    assert check_password(password, hashed_password)

def test_check_password_incorrect():
    password = "test_password"
    hashed_password = hash_password(password)
    assert not check_password("wrong_password", hashed_password)

# Test get_ollama_embedding with mocking requests
@patch('requests.post')
def test_get_ollama_embedding_success(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_post.return_value = mock_response

    text = "test text"
    embedding = get_ollama_embedding(text)
    assert embedding == [0.1, 0.2, 0.3]
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/embeddings",
        json={"model": "all-minilm:33m", "prompt": text}
    )

@patch('requests.post')
def test_get_ollama_embedding_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError
    
    text = "test text"
    embedding = get_ollama_embedding(text)
    assert embedding is None
    st.error.assert_called_once_with("Could not connect to Ollama. Make sure the Ollama service is running and accessible at 'http://ollama:11434'.")

@patch('requests.post')
def test_get_ollama_embedding_request_exception(mock_post):
    mock_post.side_effect = requests.exceptions.RequestException("Test error")
    
    text = "test text"
    embedding = get_ollama_embedding(text)
    assert embedding is None
    st.error.assert_called_once_with("Error getting embedding from Ollama: Test error")

# Test Pinecone-based user management functions
@patch('utils.get_all_users_from_pinecone_index')
def test_get_next_user_id_empty(mock_get_all_users, mock_user_index):
    mock_get_all_users.return_value = []
    assert get_next_user_id() == "1"

@patch('utils.get_all_users_from_pinecone_index')
def test_get_next_user_id_existing_users(mock_get_all_users, mock_user_index):
    mock_get_all_users.return_value = [
        {"user_id": "1", "username": "user1"},
        {"user_id": "3", "username": "user3"},
        {"user_id": "2", "username": "user2"},
    ]
    assert get_next_user_id() == "4"

@patch('utils.get_user_by_username')
@patch('utils.add_user_to_pinecone_index')
@patch('utils.get_next_user_id', return_value="1")
def test_add_user_success(mock_get_next_user_id, mock_add_to_pinecone, mock_get_user_by_username, mock_user_index):
    mock_get_user_by_username.return_value = None # User does not exist
    mock_add_to_pinecone.return_value = True
    
    result = add_user("newuser", "password")
    assert result is True
    mock_add_to_pinecone.assert_called_once()
    st.error.assert_not_called()

@patch('utils.get_user_by_username')
@patch('utils.add_user_to_pinecone_index')
def test_add_user_exists(mock_add_to_pinecone, mock_get_user_by_username, mock_user_index):
    mock_get_user_by_username.return_value = {"user_id": "1", "username": "existinguser"}
    
    result = add_user("existinguser", "password")
    assert result is False
    mock_add_to_pinecone.assert_not_called()
    st.error.assert_called_once_with("Username already exists.")

@patch('utils.get_user_from_pinecone_index')
def test_get_user_by_username_found(mock_get_from_pinecone, mock_user_index):
    mock_get_from_pinecone.return_value = {"user_id": "1", "username": "testuser", "password": "hashed_pw"}
    
    user_data = get_user_by_username("testuser")
    assert user_data == {"user_id": "1", "username": "testuser", "password": "hashed_pw"}
    mock_get_from_pinecone.assert_called_once_with(mock_user_index, "testuser")

@patch('utils.get_user_from_pinecone_index')
def test_get_user_by_username_not_found(mock_get_from_pinecone, mock_user_index):
    mock_get_from_pinecone.return_value = None
    
    user_data = get_user_by_username("nonexistent")
    assert user_data is None
    mock_get_from_pinecone.assert_called_once_with(mock_user_index, "nonexistent")
