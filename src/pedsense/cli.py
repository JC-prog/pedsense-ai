import typer
from rich.console import Console
from pathlib import Path

app = typer.Typer(help="PedSense: Pedestrian Intent Prediction Suite")
console = Console()

def setup_folders():
    """Ensures the project folders exist."""
    for folder in ["data/raw", "data/processed", "models"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

@app.command()
def setup():
    """Verify project structure and prepare environment."""
    setup_folders()
    console.print("[bold green] Project structure verified![/bold green]")
    console.print("Place your JAAD raw data in [bold cyan]data/raw/[/bold cyan]")

@app.command()
def train(model: str = typer.Option("yolo", help="Choice: yolo, resnet-lstm, hybrid")):
    """Train a specific model for intent prediction."""
    console.print(f"[bold yellow]Training {model} model...[/bold yellow]")
    # Logic will go here

@app.command()
def demo():
    """Launch the Gradio web interface."""
    console.print("[bold magenta]Launching Gradio demo...[/bold magenta]")
    # Logic will go here

if __name__ == "__main__":
    app()