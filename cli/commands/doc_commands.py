import os
import logging
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

# Use absolute imports instead of relative imports
from agents.doc_loader.agent import DocumentLoaderAgent as DocLoaderAgent  # Add alias here

# Setup logging
logger = logging.getLogger("instrukt.doc")

# Create Typer app
app = typer.Typer()

# Create console for rich output
console = Console()

@app.command("summarize")
def summarize(
    file_path: Path = typer.Argument(..., help="Path to the document file to summarize", exists=True),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save the summary"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output")
):
    """Summarize a document using Mistral 7B."""
    # Validate file extension
    valid_extensions = [".pdf", ".docx", ".doc", ".txt", ".md"]
    if file_path.suffix.lower() not in valid_extensions:
        console.print(f"[red]Error:[/red] Unsupported file format. Supported formats: {', '.join(valid_extensions)}")
        raise typer.Exit(1)
    
    # Initialize agent
    agent = DocLoaderAgent()  # This can stay the same
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Summarizing document..."),
        transient=True
    ) as progress:
        progress.add_task("summarize", total=None)
        
        # Process document
        try:
            result = agent.process(str(file_path))  # Use process instead of summarize_document
            
            if not result["success"]:
                console.print(f"[red]Error:[/red] {result['message']}")
                raise typer.Exit(1)
                
            summary = result["summary"]
            
            # Save to file if requested
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(summary)
                console.print(f"[green]Summary saved to:[/green] {output_file}")
            
            # Display summary
            console.print(Panel(summary, title=f"Summary of {file_path.name}", border_style="blue"))
            
            # Show metadata if verbose
            if verbose and "metadata" in result:
                console.print("[bold]Document Metadata:[/bold]")
                for key, value in result["metadata"].items():
                    console.print(f"  [blue]{key}:[/blue] {value}")
                    
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            logger.exception("Error in document summarization")
            raise typer.Exit(1)

@app.command("batch")
def batch_summarize(
    directory: Path = typer.Argument(..., help="Directory containing documents to summarize", exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory to save summaries"),
    file_types: list[str] = typer.Option(["pdf", "docx", "doc", "txt", "md"], "--types", "-t", help="File types to process")
):
    """Summarize multiple documents in a directory."""
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize agent
    agent = DocumentLoaderAgent()  # Change this line to use DocumentLoaderAgent instead of DocLoaderAgent
    
    # Get files to process
    files = []
    for ext in file_types:
        files.extend(list(directory.glob(f"*.{ext}")))
    
    if not files:
        console.print(f"[yellow]Warning:[/yellow] No matching documents found in {directory}")
        raise typer.Exit(0)
    
    console.print(f"[bold]Found {len(files)} documents to process[/bold]")
    
    # Process each file
    successful = 0
    failed = 0
    
    for file in files:
        console.print(f"Processing: [blue]{file.name}[/blue]")
        
        try:
            result = agent.summarize_document(str(file))
            
            if not result["success"]:
                console.print(f"  [red]Failed:[/red] {result['message']}")
                failed += 1
                continue
                
            summary = result["summary"]
            
            # Save to file if requested
            if output_dir:
                output_file = output_dir / f"{file.stem}_summary.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(summary)
                console.print(f"  [green]Summary saved to:[/green] {output_file}")
            else:
                # Display brief summary
                brief = summary[:100] + "..." if len(summary) > 100 else summary
                console.print(f"  [green]Summary:[/green] {brief}")
            
            successful += 1
            
        except Exception as e:
            console.print(f"  [red]Error:[/red] {str(e)}")
            logger.exception(f"Error processing {file}")
            failed += 1
    
    # Show summary
    console.print(f"\n[bold]Batch Summary:[/bold] {successful} successful, {failed} failed")