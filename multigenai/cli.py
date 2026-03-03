"""
MultiGenAI OS — Command-Line Interface

Entry point: `mgos` (registered in pyproject.toml [project.scripts])
Also runnable as: python -m multigenai.cli

Subcommands:
  mgos image      --prompt "..." [--style ...] [--width ...] [--height ...]
  mgos video      --prompt "..." [--frames N] [--fps N]
  mgos audio      --prompt "..." [--type voice|music] [--duration N]
  mgos document   --prompt "..." [--format docx|pptx]
  mgos capability
  mgos identity   list | add | show | delete
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    name="mgos",
    help="MultiGenAI OS — Modular Multimodal Generation Operating System",
    add_completion=False,
    rich_markup_mode="rich",
)
identity_app = typer.Typer(help="Manage character identity profiles.")
app.add_typer(identity_app, name="identity")

console = Console()

# ---------------------------------------------------------------------------
# Shared context: lifecycle startup
# ---------------------------------------------------------------------------

def _startup():
    """Boot LifecycleManager and return (settings, ExecutionContext)."""
    from multigenai.core.lifecycle import LifecycleManager
    from multigenai.core.execution_context import ExecutionContext
    lm = LifecycleManager()
    settings = lm.startup()
    ctx = ExecutionContext.build(settings)
    return settings, ctx


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------

@app.command()
def image(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Image description."),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Style ID (e.g. cinematic-dark)."),
    width: int = typer.Option(1024, "--width", "-W", help="Image width in pixels."),
    height: int = typer.Option(1024, "--height", "-H", help="Image height in pixels."),
    steps: int = typer.Option(40, "--steps", help="Inference steps."),
    guidance: float = typer.Option(8.0, "--guidance", help="Guidance scale."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility."),
):
    """Generate a cinematic image from a text prompt."""
    settings, ctx = _startup()
    from multigenai.llm.schema_validator import ImageGenerationRequest
    from multigenai.core.generation_manager import GenerationManager

    request = ImageGenerationRequest(
        prompt=prompt, style=style,
        width=width, height=height,
        seed=seed,
    )
    console.print(Panel(f"[bold cyan]Generating image…[/bold cyan]\nPrompt: {prompt}", title="Image Engine"))
    result = GenerationManager(ctx).generate_image(request)
    if result.success:
        console.print(f"[bold green]✔ Done![/bold green] Saved to: {result.path}")
        console.print(f"   Seed: {result.seed} | Size: {result.width}×{result.height}")
    else:
        console.print(f"[bold red]✘ Failed:[/bold red] {result.error}", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------

@app.command()
def video(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Video scene description."),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Style ID."),
    frames: int = typer.Option(10, "--frames", "-f", help="Number of frames."),
    fps: int = typer.Option(24, "--fps", help="Frames per second."),
    width: int = typer.Option(512, "--width", "-W"),
    height: int = typer.Option(512, "--height", "-H"),
    seed: Optional[int] = typer.Option(None, "--seed"),
):
    """Generate a video from a text prompt."""
    settings, ctx = _startup()
    from multigenai.llm.schema_validator import VideoGenerationRequest
    from multigenai.core.generation_manager import GenerationManager

    request = VideoGenerationRequest(
        prompt=prompt, style_id=style,
        num_frames=frames, fps=fps,
        width=width, height=height, seed=seed,
    )
    console.print(Panel(f"[bold cyan]Generating video ({frames} frames)…[/bold cyan]\nPrompt: {prompt}", title="Video Engine"))
    result = GenerationManager(ctx).generate_video(request)
    if result.success:
        console.print(f"[bold green]✔ Done![/bold green] Saved to: {result.path}")
    else:
        console.print(f"[bold red]✘ Failed:[/bold red] {result.error}", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

@app.command()
def audio(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Audio description or script."),
    audio_type: str = typer.Option("voice", "--type", "-t", help="Type: voice | music | sfx | ambient."),
    duration: float = typer.Option(5.0, "--duration", "-d", help="Duration in seconds."),
    output_format: str = typer.Option("wav", "--format", help="Output format: wav | mp3."),
):
    """Generate audio (voice, music, or sfx) from a text prompt."""
    settings, ctx = _startup()
    from multigenai.llm.schema_validator import AudioGenerationRequest
    from multigenai.core.generation_manager import GenerationManager

    request = AudioGenerationRequest(
        prompt=prompt, audio_type=audio_type,
        duration_seconds=duration, output_format=output_format,
    )
    console.print(Panel(f"[bold cyan]Generating audio ({audio_type}, {duration}s)…[/bold cyan]", title="Audio Engine"))
    result = GenerationManager(ctx).generate_audio(request)
    if result.success:
        console.print(f"[bold green]✔ Done![/bold green] Saved to: {result.path}")
    else:
        console.print(f"[bold red]✘ Failed:[/bold red] {result.error}", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------

@app.command()
def document(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Document topic or description."),
    output_format: str = typer.Option("docx", "--format", help="Format: docx | pptx."),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Style ID."),
):
    """Generate a Word document or PowerPoint presentation."""
    settings, ctx = _startup()
    from multigenai.llm.schema_validator import DocumentGenerationRequest
    from multigenai.core.generation_manager import GenerationManager

    request = DocumentGenerationRequest(
        prompt=prompt, output_format=output_format, style_id=style,
        doc_type="presentation" if output_format == "pptx" else "report",
    )
    console.print(Panel(f"[bold cyan]Generating {output_format}…[/bold cyan]\nTopic: {prompt}", title="Document Engine"))

    manager = GenerationManager(ctx)
    if output_format == "pptx":
        result = manager.generate_presentation(request)
    else:
        result = manager.generate_document(request)

    if success:
        console.print(f"[bold green]✔ Done![/bold green] Saved to: {path}")
    else:
        console.print(f"[bold red]✘ Failed:[/bold red] {error}", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# capability
# ---------------------------------------------------------------------------

@app.command()
def capability():
    """Print a full system capability report."""
    from multigenai.core.capability_report import CapabilityReport
    CapabilityReport().report()


# ---------------------------------------------------------------------------
# identity subcommands
# ---------------------------------------------------------------------------

@identity_app.command("list")
def identity_list():
    """List all registered character identities."""
    from multigenai.core.lifecycle import LifecycleManager
    from multigenai.memory.identity_store import IdentityStore
    lm = LifecycleManager()
    settings = lm.startup()
    store = IdentityStore(store_dir=settings.memory.store_dir)
    ids = store.list_all()
    if not ids:
        console.print("[yellow]No identities registered yet.[/yellow]")
        return
    table = Table(title="Registered Identities")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Has Face Embedding")
    for cid in ids:
        p = store.get(cid)
        table.add_row(cid, p.name if p else "?", "✔" if p and p.face_embedding else "✘")
    console.print(table)


@identity_app.command("add")
def identity_add(
    character_id: str = typer.Argument(..., help="Unique character ID."),
    name: str = typer.Option(..., "--name", "-n", help="Character display name."),
    description: str = typer.Option("", "--description", "-d"),
):
    """Register a new character identity profile."""
    from multigenai.core.lifecycle import LifecycleManager
    from multigenai.memory.identity_store import IdentityStore, CharacterProfile
    lm = LifecycleManager()
    settings = lm.startup()
    store = IdentityStore(store_dir=settings.memory.store_dir)
    profile = CharacterProfile(character_id=character_id, name=name, description=description)
    store.add(profile)
    console.print(f"[bold green]✔[/bold green] Identity '{character_id}' saved.")


@identity_app.command("show")
def identity_show(character_id: str = typer.Argument(...)):
    """Show details for a registered character identity."""
    from multigenai.core.lifecycle import LifecycleManager
    from multigenai.memory.identity_store import IdentityStore
    lm = LifecycleManager()
    settings = lm.startup()
    store = IdentityStore(store_dir=settings.memory.store_dir)
    p = store.get(character_id)
    if not p:
        console.print(f"[red]Identity '{character_id}' not found.[/red]", err=True)
        raise typer.Exit(1)
    console.print(Panel(
        f"[bold]ID:[/bold] {p.character_id}\n"
        f"[bold]Name:[/bold] {p.name}\n"
        f"[bold]Description:[/bold] {p.description}\n"
        f"[bold]Lighting bias:[/bold] {p.lighting_bias}\n"
        f"[bold]Face embedding:[/bold] {'present' if p.face_embedding else 'not set'}\n"
        f"[bold]LoRA:[/bold] {p.lora_reference or 'not set'}",
        title=f"Identity: {p.name}",
    ))


@identity_app.command("delete")
def identity_delete(
    character_id: str = typer.Argument(...),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
):
    """Delete a character identity profile."""
    from multigenai.core.lifecycle import LifecycleManager
    from multigenai.memory.identity_store import IdentityStore
    if not confirm:
        typer.confirm(f"Delete identity '{character_id}'?", abort=True)
    lm = LifecycleManager()
    settings = lm.startup()
    store = IdentityStore(store_dir=settings.memory.store_dir)
    if store.delete(character_id):
        console.print(f"[bold green]✔[/bold green] Deleted '{character_id}'.")
    else:
        console.print(f"[red]Identity '{character_id}' not found.[/red]", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# __main__ fallback for `python -m multigenai.cli`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
