import typer
import os
import logging
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

# Import searchbot agent
# Replace relative imports like:
# from ...agents.searchbot.agent import SearchbotAgent
# With absolute imports:
from agents.searchbot.agent import SearchbotAgent

# Setup logging
logger = logging.getLogger("instrukt.search")

# Create Typer app
app = typer.Typer()

# Create console for rich output
console = Console()

@app.command("ask")
def ask(
    query: str = typer.Argument(..., help="Question to ask"),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection to search in"),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum number of results to retrieve"),
    show_sources: bool = typer.Option(False, "--sources", "-s", help="Show source documents used for the answer")
):
    """Ask a question and get an answer using RAG with Mistral."""
    # Initialize agent
    agent = SearchbotAgent()
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Searching and generating answer..."),
        transient=True
    ) as progress:
        progress.add_task("search", total=None)
        
        # Process query
        try:
            result = agent.search(query, collection=collection, limit=limit)
            
            if not result["success"]:
                console.print(f"[red]Error:[/red] {result['message']}")
                raise typer.Exit(1)
                
            response = result["response"]
            
            # Display response
            console.print(Panel(Markdown(response), title="Answer", border_style="blue"))
            
            # Show sources if requested
            if show_sources and "results" in result:
                console.print("\n[bold]Sources:[/bold]")
                
                for i, source in enumerate(result["results"]):
                    similarity = source.get("similarity", 0) * 100
                    content = source.get("content", "")
                    # Truncate content if too long
                    if len(content) > 300:
                        content = content[:297] + "..."
                    
                    metadata = source.get("metadata", {})
                    source_info = ""
                    if "file_path" in metadata:
                        source_info = f"File: {metadata['file_path']}"
                    elif "source" in metadata:
                        source_info = f"Source: {metadata['source']}"
                    
                    console.print(f"[bold]{i+1}.[/bold] [cyan]{source_info}[/cyan] [green](Relevance: {similarity:.1f}%)[/green]")
                    console.print(f"{content}\n")
                    
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

@app.command("file")
def search_file(
    query: str = typer.Argument(..., help="Question to ask about the file"),
    file_path: str = typer.Argument(..., help="Path to the file to search in"),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum number of results to retrieve"),
    show_sources: bool = typer.Option(False, "--sources", "-s", help="Show source documents used for the answer")
):
    """Ask a question about a specific file using RAG with Mistral."""
    # Check if file exists
    if not os.path.exists(file_path):
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)
    
    # Initialize agent
    agent = SearchbotAgent()
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]Searching in {os.path.basename(file_path)}..."),
        transient=True
    ) as progress:
        progress.add_task("search", total=None)
        
        # Process query
        try:
            result = agent.search_by_file(query, file_path, limit=limit)
            
            if not result["success"]:
                console.print(f"[red]Error:[/red] {result['message']}")
                raise typer.Exit(1)
                
            response = result["response"]
            
            # Display response
            console.print(Panel(Markdown(response), title=f"Answer about {os.path.basename(file_path)}", border_style="blue"))
            
            # Show sources if requested
            if show_sources and "results" in result:
                console.print("\n[bold]Sources:[/bold]")
                
                for i, source in enumerate(result["results"]):
                    similarity = source.get("similarity", 0) * 100
                    content = source.get("content", "")
                    # Truncate content if too long
                    if len(content) > 300:
                        content = content[:297] + "..."
                    
                    metadata = source.get("metadata", {})
                    location = ""
                    if "line_range" in metadata:
                        location = f"Lines {metadata['line_range'][0]}-{metadata['line_range'][1]}"
                    
                    console.print(f"[bold]{i+1}.[/bold] [cyan]{location}[/cyan] [green](Relevance: {similarity:.1f}%)[/green]")
                    console.print(f"{content}\n")
                    
        except Exception as e:
            logger.error(f"Error in file search: {str(e)}")
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)

@app.command("list-collections")
def list_collections():
    """List all available search collections."""
    # Initialize agent
    agent = SearchbotAgent()
    
    # Get collections from the DB handler
    collections = agent.db_handler.list_collections()
    
    # Create table
    table = Table(title="Available Search Collections")
    table.add_column("Collection", style="cyan")
    table.add_column("Documents", style="green")
    
    # Add rows
    for collection in collections:
        stats = agent.db_handler.collection_stats(collection)
        doc_count = stats["document_count"] if stats else 0
        table.add_row(collection, str(doc_count))
    
    # Display table
    console.print(table)