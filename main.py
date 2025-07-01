#!/usr/bin/env python3
"""
Advanced Agentic Document Intelligence System

Main entry point for the Document Intelligence System that provides
multi-modal document processing and intelligent question-answering capabilities.

Usage:
    python main.py --help
"""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.group()
@click.version_option(version="1.0.0")
def main():
    """
    🤖 Advanced Agentic Document Intelligence System
    
    A sophisticated multi-agent system for document processing and intelligent QA.
    """
    pass

@main.group()
def process():
    """Document processing commands."""
    pass

@main.group() 
def qa():
    """Question-answering commands."""
    pass

@main.command()
def info():
    """Show system information and architecture overview."""
    
    info_text = """
# 🤖 Advanced Agentic Document Intelligence System

## System Overview

This system uses a multi-agent architecture to:

1. **Parse Complex Documents**: Analyze layout and extract multi-modal content
2. **Build Knowledge Base**: Store structured information with vector embeddings
3. **Answer Questions**: Use specialized agents for intelligent QA

## Architecture Components

**Phase 1 - Document Processing:**
- Layout Analyzer Agent: Identifies document structure
- Text Extraction Agent: Extracts and processes textual content  
- Table Extraction Agent: Processes tabular data
- Image Processing Agent: Handles figures and images

**Phase 2 - Question Answering:**
- Supervisor Agent: Orchestrates query routing and response synthesis
- Text RAG Agent: Retrieval-augmented generation for text queries
- Table Analysis Agent: Specialized table querying and analysis
- Image Analysis Agent: Image and figure understanding

**Data Storage:**
- MongoDB: Document metadata and structured content
- ChromaDB: Vector embeddings for semantic search

## Quick Start

1. **Initialize System:**
   ```bash
   python main.py init
   ```

2. **Process Documents:**
   ```bash
   python main.py process document <file_path>
   python main.py process directory <dir_path>
   ```

3. **Ask Questions:**
   ```bash
   python main.py qa chat          # Interactive mode
   python main.py qa ask "question" # Single question
   ```

## Requirements

- Python 3.8+
- MongoDB (local or remote)
- Cerebras or Groq API key
- Required Python packages (see requirements.txt)
    """
    
    console.print(Panel(
        info_text,
        title="System Information",
        border_style="blue"
    ))

@main.command()
def init():
    """Initialize the system (setup directories, test connections)."""
    import subprocess
    import sys
    
    # Run the init command from processor CLI
    cmd = [sys.executable, '-m', 'interfaces.cli_processor', 'init']
    subprocess.run(cmd)

# Import and register processor commands
@process.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def document(file_path: str, verbose: bool):
    """Process a single document."""
    import subprocess
    import sys
    
    cmd = [sys.executable, '-m', 'interfaces.cli_processor', 'process-document', file_path]
    if verbose:
        cmd.append('--verbose')
    
    subprocess.run(cmd)

@process.command()
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--pattern', '-p', default='*.pdf', help='File pattern to match')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def directory(directory_path: str, pattern: str, verbose: bool):
    """Process all documents in a directory."""
    import subprocess
    import sys
    
    cmd = [sys.executable, '-m', 'interfaces.cli_processor', 'process-directory', directory_path, '--pattern', pattern]
    if verbose:
        cmd.append('--verbose')
    
    subprocess.run(cmd)

@process.command()
@click.argument('document_id', required=False)
@click.option('--all', '-a', is_flag=True, help='Show all documents')
def status(document_id: str, all: bool):
    """Check document processing status."""
    import subprocess
    import sys
    
    cmd = [sys.executable, '-m', 'interfaces.cli_processor', 'status']
    if all:
        cmd.append('--all')
    elif document_id:
        cmd.append(document_id)
    
    subprocess.run(cmd)

# Import and register QA commands
@qa.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def chat(verbose: bool):
    """Start interactive QA chat session."""
    import subprocess
    import sys
    
    cmd = [sys.executable, '-m', 'interfaces.cli_chatbot', 'chat']
    if verbose:
        cmd.append('--verbose')
    
    subprocess.run(cmd)

@qa.command()
@click.argument('question')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def ask(question: str, verbose: bool):
    """Ask a single question."""
    import subprocess
    import sys
    
    cmd = [sys.executable, '-m', 'interfaces.cli_chatbot', 'ask', question]
    if verbose:
        cmd.append('--verbose')
    
    subprocess.run(cmd)

@qa.command()
def test():
    """Test the QA system."""
    import subprocess
    import sys
    
    cmd = [sys.executable, '-m', 'interfaces.cli_chatbot', 'test']
    subprocess.run(cmd)

if __name__ == '__main__':
    main()