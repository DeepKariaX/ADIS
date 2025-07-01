"""
Pytest configuration for the Advanced Agentic Document Intelligence System

This file provides fixtures and configuration for both test_config.py and test_system.py
"""

import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def test_settings():
    """Provide test settings that don't require real API keys"""
    from config.settings import Settings
    
    return Settings(
        cerebras_api_key="test-cerebras-key",
        groq_api_key="test-groq-key",
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        embedding_model="all-MiniLM-L6-v2",
        mongodb_url="mongodb://localhost:27017",
        database_name="test_document_intelligence",
        vector_db_path="./test_data/vector_db",
        upload_dir="./test_data/uploads",
        processed_dir="./test_data/processed",
        max_iterations=5,
        chunk_size=500,
        chunk_overlap=100,
        log_level="INFO",
        log_file="./test_logs/app.log"
    )


@pytest.fixture(scope="session")
def setup_test_directories():
    """Setup test directories and ensure they exist"""
    test_dirs = [
        "./test_data/uploads",
        "./test_data/processed",
        "./test_data/vector_db",
        "./test_logs"
    ]
    
    created_dirs = []
    for dir_path in test_dirs:
        path_obj = Path(dir_path)
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
            created_dirs.append(path_obj)
    
    yield test_dirs
    
    # Optional cleanup - commented out to preserve test data for debugging
    # for dir_path in created_dirs:
    #     if dir_path.exists():
    #         shutil.rmtree(dir_path)


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing environment variable loading"""
    env_content = """
# Test Environment Configuration
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=test-groq-key
CEREBRAS_API_KEY=test-cerebras-key
EMBEDDING_MODEL=all-MiniLM-L6-v2
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=test_document_intelligence
VECTOR_DB_PATH=./test_data/vector_db
UPLOAD_DIR=./test_data/uploads
PROCESSED_DIR=./test_data/processed
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RAG_TOP_K=5
LOG_LEVEL=DEBUG
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(env_content)
        temp_file = f.name
    
    # Store original environment
    original_env = os.environ.copy()
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_document_data():
    """Provide sample document data for testing"""
    return {
        "document_id": "test_doc_20240101_120000_123",
        "metadata": {
            "filename": "sample_document.pdf",
            "file_path": "/test/path/sample_document.pdf",
            "file_size": 2048,
            "file_type": ".pdf",
            "page_count": 2,
            "author": "Test Author",
            "title": "Sample Test Document"
        },
        "elements": [
            {
                "element_id": "elem_text_001",
                "content_type": "text",
                "text_content": {
                    "content": "This is the main heading of the document",
                    "font_size": 18.0,
                    "is_bold": True
                },
                "bounding_box": {"x1": 50, "y1": 100, "x2": 400, "y2": 130, "page_number": 1}
            },
            {
                "element_id": "elem_text_002",
                "content_type": "text", 
                "text_content": {
                    "content": "This document contains sample content for testing the document intelligence system. It includes various types of content to validate multi-modal processing capabilities.",
                    "font_size": 12.0,
                    "is_bold": False
                },
                "bounding_box": {"x1": 50, "y1": 150, "x2": 500, "y2": 200, "page_number": 1}
            },
            {
                "element_id": "elem_table_001",
                "content_type": "table",
                "table_content": {
                    "headers": ["Feature", "Status", "Notes"],
                    "rows": [
                        ["Text Extraction", "Complete", "Working well"],
                        ["Table Processing", "Complete", "Supports complex tables"],
                        ["Image Analysis", "Complete", "Metadata extraction"]
                    ],
                    "caption": "System Features Status"
                },
                "bounding_box": {"x1": 50, "y1": 250, "x2": 500, "y2": 350, "page_number": 1}
            },
            {
                "element_id": "elem_image_001",
                "content_type": "image",
                "image_content": {
                    "image_path": "/test/images/diagram.png",
                    "caption": "System Architecture Diagram",
                    "description": "A flowchart showing the document processing pipeline"
                },
                "bounding_box": {"x1": 100, "y1": 400, "x2": 400, "y2": 600, "page_number": 2}
            }
        ]
    }


def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring a real API key"
    )
    config.addinivalue_line(
        "markers", "requires_mongodb: mark test as requiring MongoDB connection"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring full system"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle environment-specific skipping"""
    
    # Check for real API keys
    has_real_groq_key = (
        os.getenv("GROQ_API_KEY") and 
        not os.getenv("GROQ_API_KEY").startswith("test") and
        not os.getenv("GROQ_API_KEY").startswith("your_")
    )
    
    has_real_cerebras_key = (
        os.getenv("CEREBRAS_API_KEY") and 
        not os.getenv("CEREBRAS_API_KEY").startswith("test") and
        not os.getenv("CEREBRAS_API_KEY").startswith("your_")
    )
    
    has_real_api_key = has_real_groq_key or has_real_cerebras_key
    
    # Check for MongoDB availability
    has_mongodb = os.getenv("MONGODB_URL", "").startswith("mongodb://")
    
    # Apply markers based on environment
    skip_api_tests = pytest.mark.skip(reason="No real API key available")
    skip_mongodb_tests = pytest.mark.skip(reason="MongoDB not available")
    
    for item in items:
        # Skip tests requiring API keys if not available
        if "requires_api_key" in item.keywords and not has_real_api_key:
            item.add_marker(skip_api_tests)
        
        # Skip tests requiring MongoDB if not available  
        if "requires_mongodb" in item.keywords and not has_mongodb:
            item.add_marker(skip_mongodb_tests)


def pytest_runtest_setup(item):
    """Setup for individual test runs"""
    # Set testing environment variable for all tests
    os.environ['TESTING'] = '1'


def pytest_runtest_teardown(item):
    """Cleanup after individual test runs"""
    # Clean up testing environment variable
    if 'TESTING' in os.environ:
        del os.environ['TESTING']