import os
from pathlib import Path
from typing import Optional, Dict
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict

class Settings(BaseSettings):
    # API Keys
    cerebras_api_key: Optional[str] = Field(default=None, description="Cerebras API key")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    
    # Database Configuration
    mongodb_url: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URL")
    database_name: str = Field(default="document_intelligence", description="MongoDB database name")
    
    # Vector Database
    vector_db_path: str = Field(default="./data/vector_db", description="ChromaDB storage path")
    
    # Model Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model name")
    llm_provider: str = Field(default="groq", description="LLM provider: 'cerebras' or 'groq'")
    llm_model: str = Field(default="llama-3.1-8b-instant", description="LLM model name")
    llm_temperature: float = Field(default=0.1, description="LLM temperature for response generation")
    llm_max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for LLM responses")
    
    # Document Processing
    upload_dir: str = Field(default="./data/uploads", description="Document upload directory")
    processed_dir: str = Field(default="./data/processed", description="Processed documents directory")
    
    # Agent Configuration
    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    chunk_size: int = Field(default=2000, description="Text chunk size for embeddings")
    chunk_overlap: int = Field(default=400, description="Overlap between text chunks")
    min_chunk_size: int = Field(default=500, description="Minimum text chunk size")
    max_chunk_size: int = Field(default=4000, description="Maximum text chunk size")
    enable_model_downloads: bool = Field(default=True, description="Allow downloading large models (disable for testing/basic mode)")
    layout_model_timeout: int = Field(default=300, description="Timeout for layout model initialization in seconds")
    
    # Threshold settings for processing
    thresholds: Dict[str, float] = Field(default={
        "min_image_size": 50.0,
        "large_font_size": 16.0,
        "caption_search_margin": 50.0,
        "table_accuracy_lattice": 50.0,
        "table_accuracy_stream": 40.0,
        "table_confidence_default": 0.7,
        "layout_detection_threshold": 0.8,
        "rag_similarity_threshold": 0.3,
        "min_text_length": 200.0,
        "chunk_length_threshold_high": 1500.0,
        "chunk_length_threshold_medium": 800.0
    }, description="Various threshold values used in processing")
    
    # RAG Configuration
    rag_top_k: int = Field(default=8, description="Default number of chunks to retrieve for RAG")
    rag_context_limit: int = Field(default=6, description="Maximum chunks for context window")
    rag_enable_reranking: bool = Field(default=True, description="Enable chunk reranking for better relevance")
    
    # Database Configuration
    vector_collection_name: str = Field(default="document_embeddings", description="Vector database collection name")
    vector_space_type: str = Field(default="cosine", description="Vector space type (cosine, l2, ip)")
    
    # Processing Limits
    max_documents_overview: int = Field(default=10, description="Maximum documents for overview generation")
    max_table_rows_process: int = Field(default=50, description="Maximum table rows to process")
    max_image_analysis_batch: int = Field(default=5, description="Maximum images to analyze in one batch")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="./logs/app.log", description="Log file path")
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_prefix="",
        extra="forbid"
    )

def get_settings() -> Settings:
    return Settings()

# Create directories if they don't exist
def ensure_directories():
    dirs = [
        "./data/uploads",
        "./data/processed", 
        "./data/vector_db",
        "./logs"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)