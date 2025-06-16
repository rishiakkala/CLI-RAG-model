import typer
import os
import logging
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Import embedder agent
# Replace relative imports like:
# from ...agents.embedder.agent import EmbedderAgent
# With absolute imports:
from agents.embedder.agent import EmbedderAgent

# Setup logging
logger = logging.getLogger("instrukt.embed")

# Create Typer app
app = typer.Typer()

# Create console for rich output
console = Console()

@app.command("file")
def embed_file(
    file_path: Path = typer.Argument(..., help="Path to the file to embed", exists=True),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection name for storing embeddings"),
    chunk_size: int = typer.Option(512, "--chunk-size", "-s", help="Size of text chunks for embedding")
):
    """Generate embeddings for a file using MiniLM."""
    # Initialize agent
    agent = EmbedderAgent()
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating embeddings..."),
        transient=True
    ) as progress:
        progress.add_task("embed", total=None)
        
        # Process file
        try:
            result = agent.embed_file(str(file_path), collection=collection, chunk_size=chunk_size)
            
            if not result["success"]:
                console.print(f"[red]Error:[/red] {result['message']}")
                raise typer.Exit(1)
                
            # Display results
            console.print(f"[green]Successfully embedded:[/green] {file_path.name}")
            console.print(f"Collection: [blue]{collection}[/blue]")
            console.print(f"Chunks created: [blue]{result.get('chunk_count', 0)}[/blue]")
            console.print(f"Embedding dimensions: [blue]{result.get('embedding_dim', 0)}[/blue]")
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            logger.exception("Error in file embedding")
            raise typer.Exit(1)

@app.command("directory")
def embed_directory(
    directory: Path = typer.Argument(..., help="Directory containing files to embed", exists=True),
    collection: str = typer.Option(None, "--collection", "-c", help="Collection name prefix for storing embeddings"),
    extensions: List[str] = typer.Option(["txt", "md", "py", "js", "html", "css", "json"], "--ext", "-e", help="File extensions to embed"),
    chunk_size: int = typer.Option(512, "--chunk-size", "-s", help="Size of text chunks for embedding")
):
    """Generate embeddings for multiple files in a directory."""
    # Initialize agent
    agent = EmbedderAgent()
    
    # Get files to process
    files = []
    for ext in extensions:
        files.extend(list(directory.glob(f"**/*.{ext}")))
    
    if not files:
        console.print(f"[yellow]Warning:[/yellow] No matching files found in {directory}")
        raise typer.Exit(0)
    
    console.print(f"[bold]Found {len(files)} files to embed[/bold]")
    
    # Process each file
    successful = 0
    failed = 0
    
    with Progress() as progress:
        task = progress.add_task("Embedding files...", total=len(files))
        
        for file in files:
            rel_path = file.relative_to(directory)
            progress.update(task, description=f"Embedding {rel_path}")
            
            # Determine collection name
            file_collection = collection if collection else f"dir_{directory.name}"
            
            try:
                result = agent.embed_file(str(file), collection=file_collection, chunk_size=chunk_size)
                
                if result["success"]:
                    successful += 1
                else:
                    console.print(f"[red]Failed to embed {rel_path}:[/red] {result['message']}")
                    failed += 1
                    
            except Exception as e:
                logger.exception(f"Error embedding {file}")
                console.print(f"[red]Error embedding {rel_path}:[/red] {str(e)}")
                failed += 1
                
            progress.advance(task)
    
    # Show summary
    console.print(f"\n[bold]Embedding Summary:[/bold] {successful} successful, {failed} failed")
    console.print(f"Collection: [blue]{collection or f'dir_{directory.name}'}[/blue]")

@app.command("list")
def list_collections():
    """List all available embedding collections."""
    # Initialize agent
    agent = EmbedderAgent()
    
    try:
        # Get collections
        collections = agent.list_collections()
        
        if not collections:
            console.print("[yellow]No embedding collections found[/yellow]")
            return
        
        # Display collections
        console.print("[bold]Available Embedding Collections:[/bold]")
        
        for collection in collections:
            stats = agent.get_collection_stats(collection)
            
            if stats:
                console.print(f"[blue]{collection}[/blue]: {stats.get('document_count', 0)} documents")
            else:
                console.print(f"[blue]{collection}[/blue]")
                
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        logger.exception("Error listing collections")
        raise typer.Exit(1)

@app.command("delete")
def delete_collection(
    collection: str = typer.Argument(..., help="Name of the collection to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation")
):
    """Delete an embedding collection."""
    # Initialize agent
    agent = EmbedderAgent()
    
    try:
        # Check if collection exists
        collections = agent.list_collections()
        
        if collection not in collections:
            console.print(f"[yellow]Collection not found:[/yellow] {collection}")
            raise typer.Exit(1)
        
        # Confirm deletion
        if not force:
            confirm = typer.confirm(f"Are you sure you want to delete collection '{collection}'?")
            if not confirm:
                console.print("Deletion cancelled")
                return
        
        # Delete collection
        result = agent.delete_collection(collection)
        
        if result:
            console.print(f"[green]Successfully deleted collection:[/green] {collection}")
        else:
            console.print(f"[red]Failed to delete collection:[/red] {collection}")
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        logger.exception("Error deleting collection")
        raise typer.Exit(1)