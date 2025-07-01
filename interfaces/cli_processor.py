#!/usr/bin/env python3
import asyncio
import click
from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from orchestrator import document_orchestrator
from database.mongodb_client import mongodb_client
from config.settings import ensure_directories
from utils.logger import logger

console = Console()

@click.group()
def cli():
    """Document Intelligence System - Document Processing CLI"""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
async def process_document(file_path: str, verbose: bool):
    """Process a single document through the intelligence pipeline."""
    
    if verbose:
        console.print(f"[blue]Processing document:[/blue] {file_path}")
    
    # Ensure database connection
    await mongodb_client.connect()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing document...", total=None)
            
            result = await document_orchestrator.process_document(file_path)
            
            progress.update(task, description="✅ Processing complete!")
        
        if result['status'] == 'success':
            # Display success information
            console.print(Panel.fit(
                f"[green]✅ Document processed successfully![/green]\n\n"
                f"Document ID: {result['document_id']}\n"
                f"Processing Time: {result['processing_time']:.2f}s\n"
                f"Elements Extracted: {result['elements_extracted']}\n"
                f"Embeddings Generated: {result['embeddings_generated']}\n\n"
                f"Breakdown:\n"
                f"  • Text Elements: {result['breakdown']['text_elements']}\n"
                f"  • Table Elements: {result['breakdown']['table_elements']}\n"
                f"  • Image Elements: {result['breakdown']['image_elements']}",
                title="Processing Result",
                border_style="green"
            ))
        else:
            console.print(Panel.fit(
                f"[red]❌ Document processing failed![/red]\n\n"
                f"Error: {result.get('error', 'Unknown error')}\n"
                f"Processing Time: {result.get('processing_time', 0):.2f}s",
                title="Processing Error",
                border_style="red"
            ))
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"CLI processing error: {e}")
    finally:
        await mongodb_client.disconnect()

@cli.command()
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--pattern', '-p', default='*.pdf', help='File pattern to match (default: *.pdf)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
async def process_directory(directory_path: str, pattern: str, verbose: bool):
    """Process all documents in a directory."""
    
    directory = Path(directory_path)
    files = list(directory.glob(pattern))
    
    if not files:
        console.print(f"[yellow]No files found matching pattern '{pattern}' in {directory_path}[/yellow]")
        return
    
    console.print(f"[blue]Found {len(files)} files to process[/blue]")
    
    # Ensure database connection
    await mongodb_client.connect()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing {len(files)} documents...", total=None)
            
            file_paths = [str(f) for f in files]
            result = await document_orchestrator.process_multiple_documents(file_paths)
            
            progress.update(task, description="✅ Batch processing complete!")
        
        # Display results table
        table = Table(title="Batch Processing Results")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Elements", justify="right")
        table.add_column("Time (s)", justify="right")
        
        for file_result in result['results']:
            filename = Path(file_result['file_path']).name
            status = "✅ Success" if file_result['status'] == 'success' else "❌ Failed"
            elements = str(file_result.get('elements_extracted', 0))
            time_taken = f"{file_result.get('processing_time', 0):.1f}"
            
            table.add_row(filename, status, elements, time_taken)
        
        console.print(table)
        
        # Display summary
        console.print(Panel.fit(
            f"[green]Batch Processing Summary[/green]\n\n"
            f"Total Documents: {result['total_documents']}\n"
            f"Successful: {result['successful']}\n"
            f"Failed: {result['failed']}\n"
            f"Total Time: {result['total_processing_time']:.2f}s",
            title="Summary",
            border_style="blue"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"CLI batch processing error: {e}")
    finally:
        await mongodb_client.disconnect()

@cli.command()
@click.argument('document_id', required=False)
@click.option('--all', '-a', is_flag=True, help='Show status of all documents')
async def status(document_id: str, all: bool):
    """Check the processing status of documents."""
    
    await mongodb_client.connect()
    
    try:
        if all:
            # Show all documents
            documents = await mongodb_client.list_documents()
            
            if not documents:
                console.print("[yellow]No documents found in the database.[/yellow]")
                return
            
            table = Table(title="All Documents Status")
            table.add_column("Document ID", style="cyan")
            table.add_column("Filename", style="magenta")
            table.add_column("Status", style="green")
            table.add_column("Elements", justify="right")
            table.add_column("Created", style="dim")
            
            for doc in documents:
                elements = await mongodb_client.get_elements_by_document(doc.document_id)
                
                status_style = "green" if doc.processing_status == "completed" else "red" if doc.processing_status == "failed" else "yellow"
                
                table.add_row(
                    doc.document_id[:12] + "...",
                    doc.metadata.filename,
                    f"[{status_style}]{doc.processing_status}[/{status_style}]",
                    str(len(elements)),
                    doc.created_at.strftime("%Y-%m-%d %H:%M")
                )
            
            console.print(table)
            
        elif document_id:
            # Show specific document
            status_info = await document_orchestrator.get_processing_status(document_id)
            
            if status_info['status'] == 'not_found':
                console.print(f"[red]Document with ID '{document_id}' not found.[/red]")
                return
            
            console.print(Panel.fit(
                f"[blue]Document Status[/blue]\n\n"
                f"Document ID: {status_info['document_id']}\n"
                f"Filename: {status_info['filename']}\n"
                f"Status: {status_info['status']}\n"
                f"Elements: {status_info['elements_count']}\n"
                f"Embeddings: {status_info['embeddings_count']}\n"
                f"Created: {status_info['created_at']}\n"
                f"Updated: {status_info['updated_at']}" +
                (f"\nErrors: {', '.join(status_info['errors'])}" if status_info.get('errors') else ""),
                title="Document Information",
                border_style="blue"
            ))
        else:
            console.print("[yellow]Please specify a document ID or use --all flag.[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"CLI status error: {e}")
    finally:
        await mongodb_client.disconnect()

@cli.command()
async def init():
    """Initialize the system (create directories, test connections)."""
    
    try:
        console.print("[blue]Initializing Document Intelligence System...[/blue]")
        
        # Create directories
        ensure_directories()
        console.print("✅ Created required directories")
        
        # Test database connection
        await mongodb_client.connect()
        console.print("✅ Database connection successful")
        await mongodb_client.disconnect()
        
        # Test vector store
        from database.vector_store import vector_store
        stats = vector_store.get_collection_stats()
        console.print(f"✅ Vector store initialized ({stats['total_embeddings']} embeddings)")
        
        console.print(Panel.fit(
            "[green]✅ System initialization complete![/green]\n\n"
            "You can now start processing documents with:\n"
            "  python -m interfaces.cli_processor process-document <file_path>\n"
            "  python -m interfaces.cli_processor process-directory <directory_path>",
            title="Initialization Complete",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        logger.error(f"CLI initialization error: {e}")

def main():
    """Main entry point for CLI."""
    # Convert click commands to async
    def run_async(coro):
        return asyncio.run(coro)
    
    # Store original callbacks before creating wrappers
    original_callbacks = {
        'process_document': process_document.callback,
        'process_directory': process_directory.callback,
        'status': status.callback,
        'init': init.callback
    }
    
    # Create sync wrappers for async commands
    def sync_process_document(*args, **kwargs):
        return run_async(original_callbacks['process_document'](*args, **kwargs))
    
    def sync_process_directory(*args, **kwargs):
        return run_async(original_callbacks['process_directory'](*args, **kwargs))
    
    def sync_status(*args, **kwargs):
        return run_async(original_callbacks['status'](*args, **kwargs))
    
    def sync_init(*args, **kwargs):
        return run_async(original_callbacks['init'](*args, **kwargs))
    
    # Replace with sync wrappers
    process_document.callback = sync_process_document
    process_directory.callback = sync_process_directory
    status.callback = sync_status
    init.callback = sync_init
    
    try:
        cli()
    finally:
        # Restore original callbacks
        process_document.callback = original_callbacks['process_document']
        process_directory.callback = original_callbacks['process_directory']
        status.callback = original_callbacks['status']
        init.callback = original_callbacks['init']

if __name__ == '__main__':
    main()