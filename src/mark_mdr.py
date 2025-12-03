#!/usr/bin/env python
"""CLI tool to manage MDR (Multi-Drug Resistant) patient status."""
from __future__ import annotations

import typer
from rich import print as rprint
from rich.table import Table

from face_db import load_facebank
from mdr_tracker import (
    get_mdr_patients,
    is_mdr_patient,
    mark_as_mdr,
    unmark_mdr,
)

app = typer.Typer(add_completion=False, help="Manage MDR patient status.")


@app.command()
def mark(name: str = typer.Argument(..., help="Patient name to mark as MDR")):
    """Mark a registered patient as MDR (Multi-Drug Resistant)."""
    # Check if patient is registered
    registry = load_facebank()
    if name not in registry:
        rprint(f"[bold red]Error:[/] Patient '{name}' is not registered.")
        rprint("[yellow]Tip:[/] Register the patient first using register_face.py")
        raise typer.Exit(1)
    
    if mark_as_mdr(name):
        rprint(f"[bold green]✓[/] Patient '{name}' marked as MDR.")
    else:
        rprint(f"[yellow]ℹ[/] Patient '{name}' is already marked as MDR.")


@app.command()
def unmark(name: str = typer.Argument(..., help="Patient name to remove MDR status")):
    """Remove MDR status from a patient."""
    if unmark_mdr(name):
        rprint(f"[bold green]✓[/] MDR status removed from '{name}'.")
    else:
        rprint(f"[yellow]ℹ[/] Patient '{name}' is not marked as MDR.")


@app.command()
def list():
    """List all patients currently marked as MDR."""
    mdr_patients = get_mdr_patients()
    
    if not mdr_patients:
        rprint("[yellow]No patients are currently marked as MDR.[/]")
        return
    
    table = Table(title="MDR Patients", show_header=True, header_style="bold magenta")
    table.add_column("Patient Name", style="cyan", no_wrap=True)
    table.add_column("Status", style="red")
    
    for patient in mdr_patients:
        table.add_row(patient, "MDR")
    
    rprint(table)
    rprint(f"\n[bold]Total MDR patients:[/] {len(mdr_patients)}")


@app.command()
def check(name: str = typer.Argument(..., help="Patient name to check MDR status")):
    """Check if a specific patient is marked as MDR."""
    if is_mdr_patient(name):
        rprint(f"[bold red]✓[/] Patient '{name}' is marked as MDR.")
    else:
        rprint(f"[bold green]✓[/] Patient '{name}' is NOT marked as MDR.")


if __name__ == "__main__":
    app()
