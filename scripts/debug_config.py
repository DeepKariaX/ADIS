#!/usr/bin/env python3
"""
Debug script to check configuration and model names
"""
import os
from config.settings import get_settings
from config.llm_factory import LLMFactory

def debug_configuration():
    """Debug the current configuration."""
    print("🔍 Configuration Debug Report\n")
    
    # Check environment variables
    print("📋 Environment Variables:")
    env_vars = [
        'CEREBRAS_API_KEY',
        'GROQ_API_KEY', 
        'LLM_PROVIDER',
        'LLM_MODEL'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Hide API keys for security
            if 'API_KEY' in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = f"'{value}'"
            print(f"  {var}: {display_value}")
        else:
            print(f"  {var}: NOT SET")
    
    print("\n📊 Pydantic Settings:")
    try:
        settings = get_settings()
        print(f"  llm_provider: '{settings.llm_provider}'")
        print(f"  llm_model: '{settings.llm_model}'")
        print(f"  cerebras_api_key: {'SET' if settings.cerebras_api_key else 'NOT SET'}")
        print(f"  groq_api_key: {'SET' if settings.groq_api_key else 'NOT SET'}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\n🏭 LLM Factory Test:")
    try:
        settings = get_settings()
        provider_info = LLMFactory.get_provider_info(settings)
        print(f"  Provider: {provider_info['provider']}")
        print(f"  Model: {provider_info['model']}")
        print(f"  API Key Configured: {provider_info['api_key_configured']}")
        
        # Try to create LLM
        llm = LLMFactory.create_llm(settings)
        print(f"  LLM Created: {type(llm).__name__}")
        print(f"  Model Name from LLM: {getattr(llm, 'model', 'Unknown')}")
        
    except Exception as e:
        print(f"  ERROR: {e}")

def suggest_fix():
    """Suggest how to fix the configuration."""
    print("\n💡 Suggested Fix:")
    
    settings = get_settings()
    
    if settings.llm_provider == "groq" and not settings.groq_api_key:
        print("  1. Set GROQ_API_KEY in your .env file")
        print("  2. OR change LLM_PROVIDER to 'cerebras' if you have Cerebras key")
    
    if settings.llm_provider == "cerebras" and not settings.cerebras_api_key:
        print("  1. Set CEREBRAS_API_KEY in your .env file")
        print("  2. OR change LLM_PROVIDER to 'groq' if you have Groq key")
    
    print("\n  Quick fix options:")
    print("  A) Create .env file from .env.example")
    print("  B) Add missing variables to .env file")

if __name__ == "__main__":
    debug_configuration()
    suggest_fix()