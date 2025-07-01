from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from database.models import AgentResponse
from config.llm_factory import LLMFactory
from config.settings import get_settings
from utils.logger import logger

class BaseAgent(ABC):
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.settings = get_settings()
        
        # Initialize LLM and embedding models
        self.llm = LLMFactory.create_llm(self.settings)
        self.embed_model = LLMFactory.create_embedding_model(self.settings)
        
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> AgentResponse:
        pass
    
    def create_response(
        self,
        content: Any,
        response_type: str = "text",
        sources: List[str] = None,
        metadata: Dict[str, Any] = None,
        processing_time: Optional[float] = None
    ) -> AgentResponse:
        return AgentResponse(
            agent_name=self.name,
            response_type=response_type,
            content=content,
            sources=sources or [],
            metadata=metadata or {},
            processing_time=processing_time
        )
    
    def log_activity(self, message: str, level: str = "info"):
        log_message = f"[{self.name}] {message}"
        if level == "error":
            logger.error(log_message)
        elif level == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)