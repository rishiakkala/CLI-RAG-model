import typer
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text  # Add this import
from rich.panel import Panel  # Also add this for Panel used later

# Use direct imports
from cli.commands import doc_commands, embed_commands, search_commands

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("instrukt")

# Create Typer app
app = typer.Typer(
    name="instrukt",
    help="Terminal-based modular AI agent system",
    add_completion=False,
)

# Create console for rich output
console = Console()

# Add subcommands
app.add_typer(doc_commands.app, name="doc", help="Document processing commands")
app.add_typer(embed_commands.app, name="embed", help="Embedding generation commands")
app.add_typer(search_commands.app, name="search", help="Search and question answering commands")

@app.callback()
def callback():
    """Terminal-based modular AI agent system."""
    # Display welcome message
    welcome_text = Text()
    welcome_text.append("\n✨ ", style="bold yellow")
    welcome_text.append("INSTRUKT AI AGENTS", style="bold blue")
    welcome_text.append(" ✨\n", style="bold yellow")
    welcome_text.append("Terminal-based modular AI agent system\n")
    welcome_text.append("Type ", style="dim")
    welcome_text.append("instrukt --help", style="bold green")
    welcome_text.append(" to see available commands", style="dim")
    
    console.print(Panel(welcome_text, expand=False))

@app.command("version")
def version():
    """Show the version of the application."""
    console.print("[bold blue]Instrukt AI Agents[/bold blue] [yellow]v0.1.0[/yellow]")

@app.command("status")
def status():
    """Show the status of the AI models and services."""
    from models.mistral_runner import MistralRunner
    from models.embedding_runner import EmbeddingRunner
    
    console.print("[bold]Checking AI models status...[/bold]")
    
    # Check Mistral
    try:
        mistral = MistralRunner()
        mistral_status = "[green]Available[/green]"
    except Exception as e:
        mistral_status = f"[red]Unavailable[/red] ({str(e)})"
    
    # Check Embedding model
    try:
        embedding = EmbeddingRunner()
        embedding_status = "[green]Available[/green]"
    except Exception as e:
        embedding_status = f"[red]Unavailable[/red] ({str(e)})"
    
    # Display status
    console.print(Panel.fit(
        "\n".join([
            "[bold blue]Models Status[/bold blue]",
            f"Mistral (Document & Search): {mistral_status}",
            f"MiniLM (Embeddings): {embedding_status}"
        ]),
        title="Status",
        border_style="blue"
    ))

def main():
    """Entry point for the CLI application."""
    app()

if __name__ == "__main__":
    main()