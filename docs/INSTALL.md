# 📦 Installation Guide

## Quick Start

For basic functionality with Groq or Cerebras LLMs:

```bash
# 1. Clone the repository
git clone <repository-url>
cd steps_ai_test

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys (see Configuration section)

# 5. Initialize system
python main.py init
```

## Prerequisites

### System Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **Memory**: 4GB+ RAM
- **Storage**: 2GB+ free space
- **OS**: Windows, macOS, or Linux

### Required System Dependencies

**For PDF Processing:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ghostscript python3-tk

# macOS
brew install ghostscript tk

# Windows
# Download Ghostscript from: https://www.ghostscript.com/download/
```

**For OCR (Optional):**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Database Setup

### MongoDB Configuration

**Option 1: MongoDB Atlas (Recommended - Free Tier Available)**
1. Sign up at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free cluster
3. Get connection string
4. Set `MONGODB_URL` in your `.env` file

**Option 2: Local MongoDB**
```bash
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS
brew install mongodb-community

# Windows
# Download from: https://www.mongodb.com/try/download/community
```

**Option 3: Docker**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

## Configuration

### Environment Variables

Create `.env` file from `.env.example`:

```env
# LLM Provider Configuration
LLM_PROVIDER=groq                    # "groq" or "cerebras"
LLM_MODEL=llama-3.1-8b-instant      # Model name for selected provider

# API Keys (get at least one)
GROQ_API_KEY=your_groq_api_key_here
CEREBRAS_API_KEY=your_cerebras_api_key_here

# Database
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=document_intelligence

# Optional: Advanced Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_TEMPERATURE=0.1
RAG_TOP_K=5
```

### Get API Keys

**Groq (Recommended - Fast & Free Tier):**
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for free account
3. Navigate to API Keys
4. Create new API key
5. Copy to `.env` file

**Cerebras (High Performance):**
1. Visit [Cerebras Cloud](https://cloud.cerebras.ai/)
2. Sign up for account
3. Get API credentials
4. Copy to `.env` file

## Installation Methods

### Method 1: Standard Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python scripts/debug_config.py
```

### Method 2: Development Installation

```bash
# Install with development tools
pip install -r requirements.txt
pip install pytest black flake8

# Install in development mode
pip install -e .
```

### Method 3: Minimal Installation

For basic functionality only:

```bash
# Core dependencies only
pip install langchain langchain-groq langchain-cerebras
pip install llama-index llama-index-llms-groq
pip install pymongo motor chromadb sentence-transformers
pip install rich click python-dotenv pydantic-settings
pip install PyPDF2 python-docx pandas pillow

# Set minimal mode
export MINIMAL_MODE=true
```

## Advanced Installation (Optional)

For enhanced layout analysis capabilities:

### Prerequisites for Advanced Features

```bash
# Install PyTorch (adjust for your system)
pip install torch torchvision torchaudio

# Install advanced layout analysis (optional)
pip install layoutparser[layoutmodels,tesseract]
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Note**: Advanced features are optional. The system works without them using basic layout analysis.

## Verification

### Step 1: Configuration Check
```bash
# Check configuration
python scripts/debug_config.py
```

### Step 2: System Initialization
```bash
# Initialize directories and test connections
python main.py init
```

### Step 3: Test Installation

**Quick Configuration Test:**
```bash
# Test configuration and dependencies (fastest)
python run_tests.py config

# Alternative: Direct pytest
python -m pytest tests/test_config.py -v
```

**Full System Test:**
```bash
# Test complete system functionality (requires API keys)
python run_tests.py system

# Or run all tests
python run_tests.py all
```

**Built-in System Test:**
```bash
# Run built-in QA system test
python main.py qa test

# Or test with demo
python demo.py
```

### Step 4: Process Sample Document
```bash
# Create sample document
echo "# Test Document\nThis is a test." > test.txt

# Process it
python main.py process document test.txt

# Query it
python main.py qa ask "What is this document about?"
```

## Troubleshooting

### Common Issues

**1. API Key Issues:**
```bash
# Verify keys are set
python scripts/debug_config.py

# Test provider connection
python -c "
from config.llm_factory import LLMFactory
from config.settings import get_settings
try:
    llm = LLMFactory.create_llm(get_settings())
    print('✅ LLM factory working')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

**2. MongoDB Connection:**
```bash
# Test MongoDB connection
python -c "
import asyncio
from database.mongodb_client import mongodb_client
async def test():
    try:
        await mongodb_client.connect()
        print('✅ MongoDB connected')
        await mongodb_client.disconnect()
    except Exception as e:
        print(f'❌ MongoDB error: {e}')
asyncio.run(test())
"
```

**3. Dependency Issues:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**4. Permission Errors:**
```bash
# Fix directory permissions
mkdir -p data/uploads data/processed data/vector_db logs
chmod 755 data logs
```

### Platform-Specific Notes

**Windows:**
- Use Command Prompt or PowerShell as Administrator
- Some packages may require Visual Studio Build Tools
- Consider using WSL2 for better compatibility

**macOS:**
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies

**Linux:**
- Install build essentials: `sudo apt-get install build-essential python3-dev`
- Ensure sufficient disk space for models and data

## Docker Installation (Alternative)

```bash
# Create Dockerfile (basic setup)
cat > Dockerfile << EOF
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

CMD ["python", "main.py", "info"]
EOF

# Build and run
docker build -t doc-intelligence .
docker run -it --rm -v \$(pwd)/data:/app/data doc-intelligence
```

## Performance Optimization

### Memory Management
```bash
# For large documents, increase memory limits
export PYTHONMEMORY=4G

# Use streaming for very large files
export STREAMING_MODE=true
```

### Provider Selection
```bash
# Groq: Fastest inference, good for development
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant

# Cerebras: High performance, good for production
LLM_PROVIDER=cerebras
LLM_MODEL=llama3.1-8b
```

## Next Steps

After successful installation:

1. **Process Documents**: `python main.py process document <file>`
2. **Interactive Chat**: `python main.py qa chat`
3. **System Status**: `python main.py process status --all`
4. **Configuration**: Edit `.env` for custom settings

## 🧪 Testing & Validation

### Test Categories

The system includes a comprehensive test suite divided into two main categories:

**1. Configuration Tests (`test_config.py`)**
- Environment variable validation
- API key configuration checks
- Database connection testing
- Model initialization verification
- System dependency validation

**2. System Tests (`test_system.py`)**
- Agent initialization and functionality
- Document processing pipeline
- Question answering system
- End-to-end workflow validation
- Performance and resource management

### Running Tests

**Quick Start - Test Configuration Only:**
```bash
# Fastest test - no external dependencies required
python run_tests.py config
```

**Full System Testing:**
```bash
# Test complete functionality (requires API keys)
python run_tests.py system

# Run all tests
python run_tests.py all

# Quick tests without database dependencies
python run_tests.py quick
```

**Advanced Testing:**
```bash
# Configuration tests only
python -m pytest tests/test_config.py -v

# System functionality tests
python -m pytest tests/test_system.py -v

# Skip tests requiring API keys
python -m pytest -m "not requires_api_key" -v

# Only MongoDB-related tests
python -m pytest -m "requires_mongodb" -v

# Verbose output with no capture (show print statements)
python run_tests.py all --verbose --no-capture
```

### Test Environment Setup

**For Development Testing:**
```bash
# Install development dependencies
pip install pytest pytest-asyncio pytest-mock

# Set testing environment
export TESTING=1

# Run tests with coverage (optional)
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=html
```

**Environment Variables for Testing:**
```env
# Test configuration (add to .env)
TESTING=1                    # Enables testing mode
LOG_LEVEL=DEBUG             # Verbose logging for debugging
MINIMAL_MODE=true           # Skip heavy model downloads
```

### Test Markers

The test suite uses markers for selective test execution:

- `requires_api_key`: Tests requiring real API keys
- `requires_mongodb`: Tests requiring MongoDB connection
- `integration`: Full integration tests
- `slow`: Long-running tests

### Continuous Integration

For CI/CD environments:
```yaml
# GitHub Actions example
- name: Run Tests
  run: |
    python run_tests.py config  # Always run config tests
    if [ -n "$GROQ_API_KEY" ]; then
      python run_tests.py system  # Only if API keys available
    fi
```

## Getting Help

If you encounter issues:

1. **Check Configuration**: `python run_tests.py config`
2. **Verify Setup**: `python scripts/debug_config.py`
3. **Check Logs**: `tail -f logs/app.log`
4. **Test Step by Step**:
   ```bash
   python run_tests.py config   # Start with configuration
   python main.py init          # Initialize system
   python run_tests.py system   # Test full functionality
   ```
5. **Environment Issues**: Check `.env` file settings
6. **Dependency Problems**: Try minimal installation first

### Common Test Issues

**API Key Tests Skipped:**
- Expected if no real API keys are configured
- Use test API keys for basic validation
- Set real keys for full system testing

**MongoDB Tests Failed:**
- Check MongoDB is running: `docker run -d -p 27017:27017 mongo`
- Verify connection string in `.env`
- Tests will skip gracefully if MongoDB unavailable

**Import Errors:**
- Install missing dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (requires 3.8+)
- Verify virtual environment is activated

The system is designed to work with minimal setup - you should be able to run basic document processing and QA with just the core dependencies. Tests are designed to skip gracefully when dependencies are unavailable.