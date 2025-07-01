# 🤖 Agentic Document Intelligence System (ADIS)

A sophisticated multi-agent system for deep document understanding and intelligent information retrieval. This system processes complex documents by analyzing their layout and extracting diverse content types (text, tables, images) using specialized agents, then provides an intelligent QA interface with **LlamaIndex ReAct agent orchestration** for querying the processed information.

## ✨ Key Features

- **🚀 Multi-Provider LLM Support**: Seamless switching between Cerebras and Groq
- **🧠 LlamaIndex ReAct Integration**: Advanced agentic workflows with intelligent reasoning
- **⚡ High-Performance Processing**: Parallel content extraction with specialized agents
- **🎯 Intelligent Query Routing**: Dynamic agent selection based on query intent
- **📊 Multi-Modal Content**: Advanced text, table, and image analysis
- **🔍 Semantic Search**: Vector-based retrieval with ChromaDB
- **💾 Robust Storage**: MongoDB integration with optimized schemas

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Processing Pipeline             │
├─────────────────────────────────────────────────────────────┤
│  Layout Analyzer → Text Extractor → Table Extractor → Image │
│                                Processor                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Knowledge Base                          │
├─────────────────────────────────────────────────────────────┤
│  MongoDB (Structured Data) + ChromaDB (Vector Embeddings)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              LlamaIndex ReAct Supervisor                    │
├─────────────────────────────────────────────────────────────┤
│  Intelligent Tool Selection → Text RAG | Table | Image      │
│                                Agent    Agent   Agent       │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone and setup
git clone https://github.com/DeepKariaX/StepsAI-Assignment
cd steps_ai_test
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings:
LLM_PROVIDER=groq  # or cerebras
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your_groq_api_key
MONGODB_URL=mongodb://localhost:27017
```

### 3. Initialize System
```bash
# Setup directories and test connections
python main.py init
```

### 4. Process Documents
```bash
# Process single document
python main.py process document research_paper.pdf

# Process directory
python main.py process directory ./documents --pattern "*.pdf"

# Check status
python main.py process status --all
```

### 5. Interactive QA
```bash
# Start chat session
python main.py qa chat

# Ask single question
python main.py qa ask "What are the main findings?"

# Test system
python main.py qa test
```

## 📋 Requirements

### API Keys (Get at least one)
- **Groq**: Free tier available at [console.groq.com](https://console.groq.com)
- **Cerebras**: High-performance option at [cloud.cerebras.ai](https://cloud.cerebras.ai)

### Database
- **MongoDB**: Local installation, Docker, or [MongoDB Atlas](https://www.mongodb.com/atlas) (free tier)

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install ghostscript

# macOS
brew install ghostscript

# Windows: Download from ghostscript.com
```

## 🎯 Usage Examples

### Document Processing
```bash
# Basic processing
python main.py process document paper.pdf

# Batch processing with filters
python main.py process directory ./docs --pattern "*.pdf"

# Monitor progress
python main.py process status document_id_here
```

### Question Answering
```bash
# Interactive mode
python main.py qa chat
> What does Table 2 show about performance metrics?
> Summarize the conclusions from Figure 3

# API-style usage
python main.py qa ask "Compare the results in different tables"

# System validation
python main.py qa test
```

### Advanced Configuration
```bash
# Debug configuration
python scripts/debug_config.py

# Provider switching
export LLM_PROVIDER=cerebras
export LLM_MODEL=llama3.1-8b

# Performance tuning
export RAG_TOP_K=10
export LLM_TEMPERATURE=0.1
```

## 🏛️ Architecture Highlights

### Multi-Agent Design
- **Document Processing Agents**: Layout analysis, text extraction, table processing, image handling
- **QA Agents**: Text RAG, table analysis, image analysis with intelligent routing
- **Supervisor Orchestration**: LlamaIndex ReAct framework for advanced reasoning

### LLM Provider Support
- **Unified Interface**: Single configuration for multiple providers
- **Performance Optimization**: Provider-specific optimizations
- **Fallback Strategies**: Graceful degradation when providers unavailable

### Data Architecture
- **MongoDB**: Structured storage with optimized schemas and indexing
- **ChromaDB**: Vector embeddings for semantic search and retrieval
- **Clean APIs**: Consistent interfaces for data access and manipulation

## 📚 Documentation

Comprehensive documentation available in the `docs/` folder:

- **[Installation Guide](docs/INSTALL.md)**: Detailed setup instructions
- **[Architecture Overview](docs/ARCHITECTURE.md)**: System design and components
- **[Project Summary](docs/PROJECT_SUMMARY.md)**: Features and capabilities
- **[Assessment Requirements](docs/ASSESSMENT.md)**: Original project requirements

## 🧪 Testing & Validation

### Comprehensive Test Suite

The system includes two focused test files for comprehensive validation:

```bash
# Quick configuration validation (no external dependencies)
python run_tests.py config

# System functionality tests (requires API keys)
python run_tests.py system

# Run all tests
python run_tests.py all

# Quick tests only (fastest)
python run_tests.py quick
```

### Test Categories

**Configuration Tests (`test_config.py`)**:
- Environment variable validation
- API key configuration
- Database connection setup
- Model initialization
- Dependency checks

**System Tests (`test_system.py`)**:
- Agent functionality
- Document processing pipeline
- Question answering system
- End-to-end workflows
- Performance validation

### Individual Test Commands
```bash
# Configuration and environment validation
python -m pytest tests/test_config.py -v

# System functionality testing
python -m pytest tests/test_system.py -v

# Run with specific markers
python -m pytest -m "not requires_api_key" -v  # Skip API-dependent tests
python -m pytest -m "requires_mongodb" -v      # Only MongoDB tests
```

### Quick Validation
```bash
# Basic functionality test
python main.py qa test

# Configuration validation
python scripts/debug_config.py

# System initialization check
python main.py init
```

### Demo & Examples
```bash
# Full system demonstration
python demo.py

# Process sample document
echo "# Test\nThis is a sample document." > test.txt
python main.py process document test.txt
python main.py qa ask "What is this document about?"
```

## 🔧 Configuration Options

### LLM Providers
```bash
# Groq (Fast, Free Tier)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant

# Cerebras (High Performance)
LLM_PROVIDER=cerebras
LLM_MODEL=llama3.1-8b
```

### Performance Tuning
```bash
# Retrieval settings
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.3

# Processing limits
MAX_CHUNK_SIZE=2000
CHUNK_OVERLAP=400

# LLM parameters
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000
```

## 🚨 Troubleshooting

### Common Issues

**API Key Problems:**
```bash
python scripts/debug_config.py  # Check configuration
```

**Database Connection:**
```bash
# Test MongoDB connection
python -c "
import asyncio
from database.mongodb_client import mongodb_client
asyncio.run(mongodb_client.connect())
print('✅ MongoDB connected')
"
```

**Dependency Issues:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Getting Help

1. **Check logs**: `tail -f logs/app.log`
2. **Verify configuration**: `python scripts/debug_config.py`
3. **Test minimal setup**: Use basic installation first
4. **Review documentation**: Check `docs/` folder for detailed guides

## 🏆 Key Achievements

- ✅ **Complete Multi-Agent System**: Sophisticated document processing and QA pipeline
- ✅ **Modern LLM Integration**: Cutting-edge provider support with unified interface
- ✅ **Production Ready**: Robust error handling, logging, and monitoring
- ✅ **Extensible Architecture**: Plugin-based design for future enhancements
- ✅ **Comprehensive Testing**: Full test suite with validation and examples

## 🔮 Future Enhancements

- **Vision Models**: GPT-4 Vision integration for advanced image analysis
- **Web Interface**: Streamlit/FastAPI UI for broader accessibility
- **Microservices**: Containerized architecture for scalability
- **Custom Models**: Domain-specific fine-tuning capabilities

## 📄 License

This project is part of a technical assessment demonstrating advanced agentic document intelligence capabilities.

---

**Built with**: Python, LlamaIndex, LangChain, MongoDB, ChromaDB, Groq, Cerebras
