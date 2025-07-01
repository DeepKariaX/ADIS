#!/usr/bin/env python3
import asyncio
import click
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table
from agents.qa.supervisor_agent import SupervisorAgent
from database.models import UserQuery, QueryIntent
from database.mongodb_client import mongodb_client
from config.settings import get_settings
from utils.logger import logger

console = Console()

class DocumentChatbot:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.settings = get_settings()
        self.session_active = False
    
    async def start_session(self):
        """Start a new chat session."""
        await mongodb_client.connect()
        self.session_active = True
        
        # Welcome message
        console.print(Panel.fit(
            "[bold blue]🤖 Document Intelligence QA Chatbot[/bold blue]\n\n"
            "Ask me questions about your processed documents!\n\n"
            "[dim]Commands:[/dim]\n"
            "  • Type your question and press Enter\n"
            "  • '/help' - Show help\n"
            "  • '/stats' - Show database statistics\n"
            "  • '/history' - Show recent queries\n"
            "  • '/quit' - Exit chatbot\n\n"
            "[yellow]Note: Make sure you've processed some documents first![/yellow]",
            title="Welcome",
            border_style="blue"
        ))
    
    async def handle_query(self, query_text: str) -> None:
        """Process a user query."""
        try:
            # Create user query object with clean structure
            user_query = UserQuery(
                query_text=query_text,
                query_type="chat",
                expected_response_format="text",
                session_id=f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            console.print(f"\n[dim]Processing: {query_text}[/dim]")
            
            # Process with supervisor agent
            with console.status("[bold green]Thinking...") as status:
                response = await self.supervisor.process(user_query)
            
            # Extract the answer
            if response.response_type == "supervisor_response":
                content = response.content
                answer = content.get('answer', 'No answer generated')
                intent = content.get('intent', 'unknown')
                sources = content.get('sources', [])
                
                # Display the answer
                console.print(Panel(
                    Markdown(answer),
                    title=f"Answer (Intent: {intent})",
                    border_style="green"
                ))
                
                # Display sources if available
                if sources:
                    console.print(f"\n[dim]Sources: {', '.join(sources)}[/dim]")
                
                
            else:
                # Error response
                console.print(Panel(
                    f"[red]Error: {response.content}[/red]",
                    title="Error",
                    border_style="red"
                ))
                
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")
            logger.error(f"Chatbot query error: {e}")
    
    async def show_help(self):
        """Display help information."""
        help_text = """
# Document Intelligence QA Chatbot Help

## How to ask questions:

**Text Questions:**
- "What is the main conclusion of the research?"
- "Summarize the key findings"
- "What does the document say about climate change?"

**Table Questions:**
- "What was the revenue in Q3?"
- "Show me all entries where price > 100"
- "What's the average score in the results table?"

**Image Questions:**
- "Describe Figure 1"
- "What does the chart in page 3 show?"
- "Explain the diagram"

**Multi-modal Questions:**
- "Compare the data in Table 2 with the text analysis"
- "How do the figures support the conclusions?"

## Commands:
- `/help` - Show this help
- `/stats` - Database statistics
- `/history` - Recent queries
- `/quit` - Exit
        """
        
        console.print(Panel(
            Markdown(help_text),
            title="Help",
            border_style="blue"
        ))
    
    async def show_stats(self):
        """Display database statistics."""
        try:
            # Get document count
            documents = await mongodb_client.list_documents()
            
            # Get collection stats
            from database.vector_store import vector_store
            vector_stats = vector_store.get_collection_stats()
            
            table = Table(title="Database Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", justify="right", style="magenta")
            
            table.add_row("Documents Processed", str(len(documents)))
            table.add_row("Total Embeddings", str(vector_stats['total_embeddings']))
            
            # Count by content type
            for content_type, count in vector_stats.get('content_type_distribution', {}).items():
                table.add_row(f"  {content_type.title()} Elements", str(count))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error getting stats: {e}[/red]")
    
    async def show_history(self, limit: int = 5):
        """Show recent query history."""
        try:
            # Get recent queries from database
            # Note: This is a simplified version - in practice you'd implement pagination
            console.print(f"[dim]Recent {limit} queries:[/dim]\n")
            
            # For now, just show a placeholder since we'd need to implement query history tracking
            console.print("[yellow]Query history tracking not implemented in this demo.[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Error getting history: {e}[/red]")
    
    async def run_interactive(self):
        """Run the interactive chat loop."""
        await self.start_session()
        
        try:
            while self.session_active:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command == 'quit' or command == 'exit':
                        console.print("[yellow]Goodbye! 👋[/yellow]")
                        break
                    elif command == 'help':
                        await self.show_help()
                    elif command == 'stats':
                        await self.show_stats()
                    elif command == 'history':
                        await self.show_history()
                    else:
                        console.print(f"[red]Unknown command: /{command}[/red]")
                        console.print("Type '/help' for available commands.")
                else:
                    # Process as query
                    await self.handle_query(user_input)
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Chat session interrupted. Goodbye! 👋[/yellow]")
        except Exception as e:
            console.print(f"[red]Chat session error: {e}[/red]")
            logger.error(f"Chat session error: {e}")
        finally:
            await mongodb_client.disconnect()
            self.session_active = False

@click.group()
def cli():
    """Document Intelligence System - QA Chatbot CLI"""
    pass

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
async def chat(verbose: bool):
    """Start an interactive chat session with the document QA system."""
    chatbot = DocumentChatbot()
    await chatbot.run_interactive()

@cli.command()
@click.argument('question')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
async def ask(question: str, verbose: bool):
    """Ask a single question and get an answer."""
    
    await mongodb_client.connect()
    
    try:
        console.print(f"[blue]Question:[/blue] {question}\n")
        
        supervisor = SupervisorAgent()
        user_query = UserQuery(
            query_text=question,
            query_type="api",
            expected_response_format="text",
            session_id=f"single_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        with console.status("[bold green]Processing...") as status:
            response = await supervisor.process(user_query)
        
        if response.response_type == "supervisor_response":
            content = response.content
            answer = content.get('answer', 'No answer generated')
            sources = content.get('sources', [])
            
            console.print(Panel(
                Markdown(answer),
                title="Answer",
                border_style="green"
            ))
            
            if sources:
                console.print(f"\n[dim]Sources: {', '.join(sources)}[/dim]")
                
        else:
            console.print(Panel(
                f"[red]Error: {response.content}[/red]",
                title="Error",
                border_style="red"
            ))
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"CLI ask error: {e}")
    finally:
        await mongodb_client.disconnect()

@cli.command()
async def test():
    """Test the QA system with sample questions."""
    
    sample_questions = [
        "What documents are available in the system?",
        "Summarize the main topics discussed",
        "What data is available in the tables?",
        "Are there any figures or images?"
    ]
    
    await mongodb_client.connect()
    
    try:
        supervisor = SupervisorAgent()
        
        console.print(Panel.fit(
            "[blue]Testing QA System with Sample Questions[/blue]\n\n"
            "This will test the system with basic questions to verify functionality.",
            title="QA System Test",
            border_style="blue"
        ))
        
        for i, question in enumerate(sample_questions, 1):
            console.print(f"\n[cyan]Test {i}/4:[/cyan] {question}")
            
            try:
                user_query = UserQuery(
                    query_text=question,
                    query_type="chat",
                    expected_response_format="text",
                    session_id=f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                with console.status("Processing..."):
                    response = await supervisor.process(user_query)
                
                if response.response_type == "supervisor_response":
                    content = response.content
                    answer = content.get('answer', 'No answer')
                    # Truncate long answers for test display
                    if len(answer) > 200:
                        answer = answer[:200] + "..."
                    
                    console.print(f"[green]✅ Answer:[/green] {answer}")
                    
                else:
                    console.print(f"[red]❌ Error:[/red] {response.content}")
                    
            except Exception as e:
                console.print(f"[red]❌ Test failed:[/red] {e}")
        
        console.print(Panel.fit(
            "[green]Testing complete![/green]\n\n"
            "If all tests passed, the QA system is working correctly.\n"
            "You can now use 'chat' command for interactive sessions.",
            title="Test Results",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Test initialization failed: {e}[/red]")
        logger.error(f"CLI test error: {e}")
    finally:
        await mongodb_client.disconnect()

def main():
    """Main entry point for CLI."""
    def run_async(coro):
        return asyncio.run(coro)
    
    # Store original callbacks first
    original_callbacks = {
        'chat': chat.callback,
        'ask': ask.callback,
        'test': test.callback
    }
    
    # Create sync wrappers for async commands
    def sync_chat(*args, **kwargs):
        return run_async(original_callbacks['chat'](*args, **kwargs))
    
    def sync_ask(*args, **kwargs):
        return run_async(original_callbacks['ask'](*args, **kwargs))
    
    def sync_test(*args, **kwargs):
        return run_async(original_callbacks['test'](*args, **kwargs))
    
    # Replace with sync wrappers
    chat.callback = sync_chat
    ask.callback = sync_ask
    test.callback = sync_test
    
    try:
        cli()
    finally:
        # Restore original callbacks
        chat.callback = original_callbacks['chat']
        ask.callback = original_callbacks['ask']
        test.callback = original_callbacks['test']

if __name__ == '__main__':
    main()