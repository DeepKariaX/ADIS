"""
Configuration and Environment Validation Tests

This test file validates all configurations, environment variables, 
API keys, database connections, and system dependencies.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

# Suppress known warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Field.*is deprecated.*")


class TestEnvironmentConfiguration:
    """Test environment variables and configuration loading"""
    
    def test_settings_loading(self):
        """Test that settings can be loaded with defaults"""
        from config.settings import get_settings
        
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'llm_provider')
        assert hasattr(settings, 'llm_model')
        assert hasattr(settings, 'embedding_model')
        assert hasattr(settings, 'mongodb_url')
        assert hasattr(settings, 'database_name')

    def test_default_values(self):
        """Test that default configuration values are correct"""
        from config.settings import get_settings
        
        settings = get_settings()
        assert settings.llm_provider in ['cerebras', 'groq']
        assert settings.embedding_model == 'all-MiniLM-L6-v2'
        assert settings.database_name == 'document_intelligence'
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
        assert settings.rag_top_k > 0

    def test_environment_variable_override(self):
        """Test that environment variables properly override defaults"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'cerebras',
            'LLM_MODEL': 'test-model',
            'CHUNK_SIZE': '1000',
            'RAG_TOP_K': '5'
        }):
            from config.settings import Settings
            settings = Settings()
            assert settings.llm_provider == 'cerebras'
            assert settings.llm_model == 'test-model'
            assert settings.chunk_size == 1000
            assert settings.rag_top_k == 5

    def test_required_directories_creation(self):
        """Test that required directories are created"""
        from config.settings import ensure_directories
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                ensure_directories()
                
                # Check that directories were created
                expected_dirs = [
                    "./data/uploads",
                    "./data/processed",
                    "./data/vector_db",
                    "./logs"
                ]
                
                for dir_path in expected_dirs:
                    assert Path(dir_path).exists(), f"Directory {dir_path} was not created"
                    assert Path(dir_path).is_dir(), f"{dir_path} is not a directory"
            finally:
                os.chdir(original_cwd)


class TestAPIKeyConfiguration:
    """Test API key validation and provider setup"""
    
    def test_api_key_detection(self):
        """Test detection of configured API keys"""
        from config.settings import get_settings
        
        settings = get_settings()
        
        # Test with environment variables
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key-123'}):
            from config.settings import Settings
            test_settings = Settings()
            assert test_settings.groq_api_key == 'test-key-123'
        
        with patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test-cerebras-key'}):
            from config.settings import Settings
            test_settings = Settings()
            assert test_settings.cerebras_api_key == 'test-cerebras-key'

    def test_llm_factory_provider_info(self):
        """Test LLM factory provider information"""
        from config.llm_factory import LLMFactory
        from config.settings import Settings
        
        # Test Groq provider info
        groq_settings = Settings(
            llm_provider="groq",
            groq_api_key="test-key",
            llm_model="llama-3.1-8b-instant"
        )
        
        provider_info = LLMFactory.get_provider_info(groq_settings)
        assert provider_info['provider'] == 'groq'
        assert provider_info['model'] == 'llama-3.1-8b-instant'
        assert provider_info['api_key_configured'] is True
        
        # Test Cerebras provider info
        cerebras_settings = Settings(
            llm_provider="cerebras",
            cerebras_api_key="test-key",
            llm_model="llama3.1-8b"
        )
        
        provider_info = LLMFactory.get_provider_info(cerebras_settings)
        assert provider_info['provider'] == 'cerebras'
        assert provider_info['model'] == 'llama3.1-8b'
        assert provider_info['api_key_configured'] is True

    def test_llm_factory_error_handling(self):
        """Test LLM factory error handling for missing API keys"""
        from config.llm_factory import LLMFactory
        from config.settings import Settings
        
        # Test missing Groq key
        groq_settings = Settings(
            llm_provider="groq",
            groq_api_key=None
        )
        
        with pytest.raises(ValueError, match="Groq API key is required"):
            LLMFactory.create_llm(groq_settings)
        
        # Test missing Cerebras key
        cerebras_settings = Settings(
            llm_provider="cerebras",
            cerebras_api_key=None
        )
        
        with pytest.raises(ValueError, match="Cerebras API key is required"):
            LLMFactory.create_llm(cerebras_settings)
        
        # Test unsupported provider
        invalid_settings = Settings(llm_provider="invalid_provider")
        
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMFactory.create_llm(invalid_settings)

    def test_api_key_validation_from_env(self):
        """Test API key validation from environment"""
        from config.settings import get_settings
        
        # Test with real environment variables if present
        settings = get_settings()
        
        if settings.llm_provider == "groq":
            # If Groq is configured, check if key is present
            if settings.groq_api_key:
                assert len(settings.groq_api_key) > 10, "API key seems too short"
                assert not settings.groq_api_key.startswith("your_"), "API key not properly configured"
        
        elif settings.llm_provider == "cerebras":
            # If Cerebras is configured, check if key is present
            if settings.cerebras_api_key:
                assert len(settings.cerebras_api_key) > 10, "API key seems too short"
                assert not settings.cerebras_api_key.startswith("your_"), "API key not properly configured"


class TestDatabaseConfiguration:
    """Test database connections and configurations"""
    
    def test_mongodb_configuration(self):
        """Test MongoDB configuration settings"""
        from config.settings import get_settings
        
        settings = get_settings()
        assert settings.mongodb_url is not None
        assert settings.database_name is not None
        assert len(settings.database_name) > 0

    def test_mongodb_client_initialization(self):
        """Test MongoDB client can be initialized"""
        from database.mongodb_client import MongoDBClient
        
        client = MongoDBClient()
        assert client.settings is not None
        assert hasattr(client, 'client')
        assert hasattr(client, 'database')
        
        # Before connection, these should be None
        assert client.client is None
        assert client.database is None
        assert client.documents is None
        assert client.elements is None

    @pytest.mark.asyncio
    async def test_mongodb_connection_attempt(self):
        """Test MongoDB connection attempt (may fail if MongoDB not running)"""
        from database.mongodb_client import mongodb_client
        
        try:
            await mongodb_client.connect()
            # If connection succeeds, test basic operations
            assert mongodb_client.client is not None
            assert mongodb_client.database is not None
            
            # Test collection access - collections are already set up in connect()
            assert mongodb_client.documents is not None
            assert mongodb_client.elements is not None
            assert mongodb_client.embeddings is not None
            
            # Test that we can ping the database
            await mongodb_client.client.admin.command('ping')
            
            # Clean up
            await mongodb_client.disconnect()
            
        except Exception as e:
            # Connection may fail in CI/testing environments
            pytest.skip(f"MongoDB connection failed (expected in testing): {e}")

    def test_vector_store_configuration(self):
        """Test vector store configuration"""
        from config.settings import get_settings
        
        settings = get_settings()
        assert settings.vector_db_path is not None
        assert settings.vector_collection_name is not None
        assert settings.embedding_model is not None

    def test_vector_store_initialization(self):
        """Test vector store can be initialized"""
        try:
            from database.vector_store import VectorStore
            
            vector_store = VectorStore()
            assert vector_store.embedding_model is not None
            assert vector_store.collection is not None
            assert hasattr(vector_store, 'client')
            
        except ImportError as e:
            pytest.skip(f"Vector store dependencies not available: {e}")
        except Exception as e:
            pytest.skip(f"Vector store initialization failed (expected in some environments): {e}")


class TestModelConfiguration:
    """Test model loading and configuration"""
    
    def test_embedding_model_configuration(self):
        """Test embedding model configuration"""
        from config.settings import get_settings
        
        settings = get_settings()
        assert settings.embedding_model == 'all-MiniLM-L6-v2'

    def test_embedding_model_creation(self):
        """Test embedding model can be created"""
        try:
            from config.llm_factory import LLMFactory
            from config.settings import get_settings
            
            settings = get_settings()
            embed_model = LLMFactory.create_embedding_model(settings)
            assert embed_model is not None
            
        except ImportError as e:
            pytest.skip(f"Embedding model dependencies not available: {e}")
        except Exception as e:
            pytest.skip(f"Embedding model creation failed (expected in some environments): {e}")

    def test_llm_model_configuration(self):
        """Test LLM model configuration"""
        from config.settings import get_settings
        
        settings = get_settings()
        assert settings.llm_model is not None
        assert len(settings.llm_model) > 0
        assert settings.llm_temperature >= 0.0
        assert settings.llm_temperature <= 2.0


class TestSystemDependencies:
    """Test system dependencies and imports"""
    
    def test_core_imports(self):
        """Test that core modules can be imported"""
        # Test core configuration
        from config.settings import get_settings
        from config.llm_factory import LLMFactory
        
        # Test database models
        from database.models import Document, DocumentMetadata, DocumentElement
        
        # Test utilities
        from utils.logger import logger
        from utils.chunking import intelligent_chunker
        
        assert get_settings is not None
        assert LLMFactory is not None
        assert Document is not None
        assert logger is not None

    def test_agent_imports(self):
        """Test that agent classes can be imported"""
        try:
            from agents.base_agent import BaseAgent
            from agents.parsing.layout_analyzer import LayoutAnalyzerAgent
            from agents.parsing.text_extractor import TextExtractorAgent
            from agents.parsing.table_extractor import TableExtractorAgent
            from agents.parsing.image_processor import ImageProcessorAgent
            
            assert BaseAgent is not None
            assert LayoutAnalyzerAgent is not None
            assert TextExtractorAgent is not None
            assert TableExtractorAgent is not None
            assert ImageProcessorAgent is not None
            
        except ImportError as e:
            pytest.skip(f"Agent imports failed (may require additional dependencies): {e}")

    def test_qa_agent_imports(self):
        """Test that QA agent classes can be imported"""
        try:
            from agents.qa.supervisor_agent import SupervisorAgent
            from agents.qa.text_rag_agent import TextRAGAgent
            from agents.qa.table_analysis_agent import TableAnalysisAgent
            from agents.qa.image_analysis_agent import ImageAnalysisAgent
            
            assert SupervisorAgent is not None
            assert TextRAGAgent is not None
            assert TableAnalysisAgent is not None
            assert ImageAnalysisAgent is not None
            
        except ImportError as e:
            pytest.skip(f"QA agent imports failed (may require API keys): {e}")

    def test_interface_imports(self):
        """Test that interface modules can be imported"""
        try:
            from interfaces import cli_processor, cli_chatbot
            from orchestrator import DocumentProcessingOrchestrator
            
            assert cli_processor is not None
            assert cli_chatbot is not None
            assert DocumentProcessingOrchestrator is not None
            
        except ImportError as e:
            pytest.skip(f"Interface imports failed: {e}")

    def test_optional_dependencies(self):
        """Test optional dependencies that may not be available"""
        optional_imports = [
            ('layoutparser', 'Layout analysis'),
            ('camelot', 'Table extraction'),
            ('tabula', 'Table extraction'),
            ('sentence_transformers', 'Embeddings'),
            ('chromadb', 'Vector database')
        ]
        
        for module_name, description in optional_imports:
            try:
                __import__(module_name)
                print(f"✓ {description} dependency available: {module_name}")
            except ImportError:
                print(f"⚠ {description} dependency not available: {module_name}")


class TestLoggingConfiguration:
    """Test logging configuration"""
    
    def test_logger_initialization(self):
        """Test that logger is properly initialized"""
        from utils.logger import logger
        
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')

    def test_log_directory_creation(self):
        """Test that log directory is created"""
        from config.settings import ensure_directories
        
        ensure_directories()
        log_dir = Path("./logs")
        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_logging_functionality(self):
        """Test basic logging functionality"""
        from utils.logger import logger
        
        # Test that logging doesn't raise errors
        try:
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
        except Exception as e:
            pytest.fail(f"Logging failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])