"""
LLM Factory for managing different LLM providers (Cerebras, Groq)
"""
from typing import Optional, Union
from llama_index.llms.groq import Groq
from llama_index.core.llms import LLM
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from config.settings import Settings, get_settings


class LLMFactory:
    """Factory class for creating LLM instances based on configuration."""
    
    @staticmethod
    def create_llm(settings: Optional[Settings] = None) -> LLM:
        """Create an LLM instance based on settings configuration."""
        if settings is None:
            settings = get_settings()
        
        provider = settings.llm_provider.lower()
        
        if provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("Groq API key is required when using Groq provider")
            
            return Groq(
                model=settings.llm_model,
                api_key=settings.groq_api_key,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
        
        elif provider == "cerebras":
            if not settings.cerebras_api_key:
                raise ValueError("Cerebras API key is required when using Cerebras provider")
            
            # For LlamaIndex, we'll use a generic LLM wrapper
            # Since Cerebras doesn't have a direct LlamaIndex integration, 
            # we'll create a custom wrapper if needed
            from llama_index.llms.langchain import LangChainLLM
            langchain_llm = ChatCerebras(
                model=settings.llm_model,
                api_key=settings.cerebras_api_key,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
            return LangChainLLM(llm=langchain_llm)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def create_embedding_model(settings: Optional[Settings] = None):
        """Create an embedding model instance."""
        if settings is None:
            settings = get_settings()
            
        # For now, we'll use HuggingFace embeddings regardless of LLM provider
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        return HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            trust_remote_code=True
        )
    
    @staticmethod
    def get_provider_info(settings: Optional[Settings] = None) -> dict:
        """Get information about the current LLM configuration."""
        if settings is None:
            settings = get_settings()
        
        provider = settings.llm_provider.lower()
        
        info = {
            "provider": provider,
            "embedding_model": settings.embedding_model
        }
        
        if provider == "groq":
            info.update({
                "model": settings.llm_model,
                "api_key_configured": bool(settings.groq_api_key)
            })
        elif provider == "cerebras":
            info.update({
                "model": settings.llm_model,
                "api_key_configured": bool(settings.cerebras_api_key)
            })
        
        return info


    @staticmethod
    def create_langchain_llm(settings: Optional[Settings] = None) -> Union[ChatCerebras, ChatGroq]:
        """Create a LangChain-compatible LLM instance based on settings configuration."""
        if settings is None:
            settings = get_settings()
        
        provider = settings.llm_provider.lower()
        
        if provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("Groq API key is required when using Groq provider")
            
            return ChatGroq(
                model=settings.llm_model,
                groq_api_key=settings.groq_api_key,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
        
        elif provider == "cerebras":
            if not settings.cerebras_api_key:
                raise ValueError("Cerebras API key is required when using Cerebras provider")
            
            return ChatCerebras(
                model=settings.llm_model,
                api_key=settings.cerebras_api_key,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm(settings: Optional[Settings] = None) -> LLM:
    """Convenience function to get an LLM instance."""
    return LLMFactory.create_llm(settings)


def get_langchain_llm(settings: Optional[Settings] = None) -> Union[ChatCerebras, ChatGroq]:
    """Convenience function to get a LangChain-compatible LLM instance."""
    return LLMFactory.create_langchain_llm(settings)