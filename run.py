#!/usr/bin/env python3
"""
Instrukt AI Agents - Single-run launcher

This script provides a simplified way to run the Instrukt AI Agents system
without having to use the CLI directly. It supports all the main operations
of the system through command-line arguments.

Usage:
    python run.py [command] [subcommand] [options]

Examples:
    python run.py doc summarize path/to/document.pdf
    python run.py embed document path/to/document.txt
    python run.py search ask "What is the main topic of my documents?"
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Check for critical dependencies
try:
    import requests
    import yaml
    import typer
    import rich
except ImportError as e:
    print(f"Error: Missing required dependency - {e.name}")
    print("Please install all dependencies with: pip install -r requirements.txt")
    sys.exit(1)

# Run the CLI application
if __name__ == "__main__":
    from cli.main import app
    app()