"""
Rich utilities for beautiful CLI output in NIOSH ergonomic analysis tool.
Provides professional formatting, tables, progress bars, and styled panels.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.theme import Theme
from rich import box
from rich.prompt import Confirm
from typing import Dict, List, Any

# Custom theme for NIOSH application
NIOSH_THEME = Theme(
    {
        "safe": "bold green",
        "warning": "bold yellow",
        "danger": "bold red",
        "info": "bold blue",
        "success": "bold white on green",
        "header": "bold cyan",
        "parameter": "cyan",
        "result": "magenta",
        "border": "blue",
        "task_desc": "bold blue",
    }
)

# Initialize console with custom theme and UTF-8 encoding for Windows compatibility
import sys
import os
import io

# Configure console for Windows UTF-8 support
if sys.platform == "win32":
    # Set environment variable for Python encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Configure Windows console for UTF-8
    try:
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass
    
    try:
        # Change console code page to UTF-8
        os.system('chcp 65001 > nul 2>&1')
    except:
        pass

# Create a custom stdout with UTF-8 encoding for Windows
if sys.platform == "win32":
    try:
        # Wrap stdout to handle UTF-8 encoding
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

console = Console(
    theme=NIOSH_THEME,
    force_terminal=True,
    file=None,  # Use stdout with proper encoding
    width=None,
    legacy_windows=False,  # Use modern Windows console handling
    no_color=False,
    emoji=False,  # Disable emoji rendering to avoid Unicode issues
    markup=True
)


def create_niosh_results_table(calc_result, params) -> Table:
    """Create a professional table for NIOSH calculation results"""

    table = Table(
        title="[header]NIOSH Lifting Equation Results[/header]",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        border_style="border",
    )

    # Add columns with styling
    table.add_column("Parameter", style="parameter", width=25)
    table.add_column("Origin", justify="right", style="result")
    table.add_column("Destination", justify="right", style="result")

    # Add calculation results - fixed format specifier issue
    rwl_dest_str = (
        f"{calc_result.rwl_destination_lbs:.1f}"
        if calc_result.rwl_destination_lbs is not None
        else "N/A"
    )
    li_dest_str = (
        f"{calc_result.li_destination:.2f}"
        if calc_result.li_destination is not None
        else "N/A"
    )

    table.add_row("RWL (lbs)", f"{calc_result.rwl_origin_lbs:.1f}", rwl_dest_str)
    table.add_row("Lifting Index", f"{calc_result.li_origin:.2f}", li_dest_str)

    # Risk assessment with color coding
    risk_style = get_risk_style(calc_result.risk_category)

    table.add_row(
        "Risk Category",
        f"[{risk_style}]{calc_result.risk_category}[/]",
        f"[{risk_style}]{calc_result.risk_category}[/]",
    )

    return table


def create_multipliers_table(
    multipliers: Dict[str, float], location: str = "Origin"
) -> Table:
    """Create a table showing NIOSH multipliers"""

    table = Table(
        title=f"[header]NIOSH Multipliers - {location}[/header]",
        show_header=True,
        header_style="bold cyan",
        box=box.MINIMAL_DOUBLE_HEAD,
    )

    table.add_column("Multiplier", style="parameter")
    table.add_column("Value", justify="right", style="result")
    table.add_column("Description", style="dim")

    multiplier_descriptions = {
        "HM": "Horizontal Multiplier",
        "VM": "Vertical Multiplier",
        "DM": "Distance Multiplier",
        "AM": "Asymmetry Multiplier",
        "FM": "Frequency Multiplier",
        "CM": "Coupling Multiplier",
    }

    for key, value in multipliers.items():
        description = multiplier_descriptions.get(key, "Unknown")
        # Color code low multipliers (more stressful)
        value_style = "warning" if value < 0.5 else "result" if value < 0.8 else "safe"
        table.add_row(key, f"[{value_style}]{value:.3f}[/]", description)

    return table


def create_parameters_summary(params) -> Table:
    """Create a summary table of input parameters"""

    table = Table(
        title="[header]Input Parameters Summary[/header]",
        show_header=False,
        box=box.SIMPLE,
    )

    table.add_column("Parameter", style="parameter", width=30)
    table.add_column("Value", justify="right", style="result")

    # Add parameters with appropriate formatting - safely handle None values
    table.add_row("Weight", f"{params.weight:.1f} lbs")
    table.add_row("Horizontal Distance (Origin)", f"{params.horizontal_origin:.1f} in")
    table.add_row(
        "Horizontal Distance (Destination)", f"{params.horizontal_destination:.1f} in"
    )
    table.add_row("Vertical Height (Origin)", f"{params.vertical_origin:.1f} in")
    table.add_row(
        "Vertical Height (Destination)", f"{params.vertical_destination:.1f} in"
    )
    table.add_row("Asymmetry Angle", f"{params.asymmetry_angle:.1f}°")
    table.add_row("Frequency", f"{params.frequency:.1f} lifts/min")
    table.add_row("Duration", str(params.duration) if params.duration else "N/A")
    table.add_row("Coupling", str(params.coupling) if params.coupling else "N/A")
    table.add_row("Significant Control", "Yes" if params.significant_control else "No")

    # Additional parameters for simplified calculation
    table.add_row("Gender", str(params.gender) if params.gender else "N/A")
    table.add_row("Age", str(params.age) if params.age is not None else "N/A")
    table.add_row("Judgment", str(params.judgment) if params.judgment else "N/A")
    table.add_row("One Limb Lifting", "Yes" if params.one_limb_lifting else "No")
    table.add_row(
        "Two Operators Lifting", "Yes" if params.two_operators_lifting else "No"
    )

    return table


def create_analysis_progress() -> Progress:
    """Create a progress bar for the NIOSH analysis pipeline"""

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[task_desc]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    )

    return progress


def create_multi_task_progress() -> Progress:
    """Progress for multi-task analysis"""

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        expand=True,
    )

    return progress


def get_risk_style(risk_category: str) -> str:
    """Return appropriate Rich style based on risk category"""

    risk_styles = {
        "acceptable": "safe",
        "slightly_stressful": "warning",
        "moderately_stressful": "warning",
        "hazardous": "danger",
    }

    return risk_styles.get(risk_category, "info")


def create_risk_panel(risk_category: str, risk_comment: str) -> Panel:
    """Create a styled panel for risk assessment"""

    style = get_risk_style(risk_category)
    icons = {
        "acceptable": "[success]OK[/]",
        "slightly_stressful": "[warning]CAUTION[/]",
        "moderately_stressful": "[warning]CAUTION[/]",
        "hazardous": "[danger]HAZARD[/]",
    }

    icon = icons.get(risk_category, "[info]INFO[/]")

    content = f"[{style}]{icon} Risk Level: {risk_category.upper().replace('_', ' ')}[/]\n\n{risk_comment}"

    panel = Panel(
        content, title="Risk Assessment", border_style=style, padding=(1, 2)
    )

    return panel


def create_status_panel(message: str, status_type: str = "info") -> Panel:
    """Create status panels for different scenarios"""

    styles = {
        "success": ("success",),
        "warning": ("warning"),
        "error": ("danger"),
        "info": ("info"),
    }

    style = styles.get(status_type, "info")

    panel = Panel(f"[{style}]{message}[/]", border_style=style, padding=(0, 1))

    return panel


def create_text_section_panel(
    text: str, title: str, border_style: str = "cyan"
) -> Panel:
    """Create a panel for text sections like Job Analysis, Hazard Assessment, etc."""

    # Truncate very long text for better display
    if len(text) > 2000:
        text = text[:2000] + "\n\n[...] (text truncated for display)"

    panel = Panel(text, title=title, border_style=border_style, padding=(1, 2))

    return panel


def create_multi_task_summary(multi_task_result) -> Table:
    """Create a summary for multi-task analysis with Composite Lifting Index"""

    table = Table(
        title="[header]Multi-Task Job Analysis Summary[/header]",
        show_header=True,
        header_style="bold magenta",
        box=box.DOUBLE,
        border_style="border",
    )

    table.add_column("Task ID", style="parameter")
    table.add_column("Task Description", style="info")
    table.add_column("LI", justify="right", style="result")
    table.add_column("STLI", justify="right", style="result")
    table.add_column("Risk Level", style="result")

    if hasattr(multi_task_result, "task_results"):
        for i, task_result in enumerate(multi_task_result.task_results, 1):
            risk_style = get_risk_style(task_result.risk_category)
            description = (
                task_result.task_description[:40] + "..."
                if len(task_result.task_description) > 40
                else task_result.task_description
            )
            table.add_row(
                f"Task {i}",
                description,
                f"{task_result.li_origin:.2f}",
                (
                    f"{multi_task_result.stli_values[i-1]:.2f}"
                    if hasattr(multi_task_result, "stli_values")
                    and i - 1 < len(multi_task_result.stli_values)
                    else "N/A"
                ),
                f"[{risk_style}]{task_result.risk_category}[/]",
            )

        # Composite Lifting Index row
        if hasattr(multi_task_result, "composite_lifting_index"):
            table.add_row(
                "",
                "[bold cyan]COMPOSITE LIFTING INDEX (CLI)[/]",
                "",
                f"[bold magenta]{multi_task_result.composite_lifting_index:.2f}[/]",
                "",
            )

    return table


def create_welcome_panel() -> Panel:
    """Create the welcome panel for the application"""

    content = """[header]🏗 NIOSH Job Description Generator - Enhanced Edition[/]

[info]Sistema Ibrido:[/] Ollama(Generation) + Python(Extraction)
[success]✨ Beautiful Terminal Output with Rich Library[/]
[info]🔄 Enhanced with Unit Tests and Retry Logic[/]"""

    panel = Panel(content, title=">> Welcome <<", border_style="blue", padding=(1, 2))

    return panel


def create_examples_table() -> Table:
    """Create a table with example inputs"""

    examples = [
        "operatore che carica bobine da 50 libbre su un macchinario",
        "magazziniere che sposta scatole da scaffali bassi a carrello",
        "worker loading boxes from floor to conveyor belt",
        "lifting bags into a hopper with twisting motion",
    ]

    table = Table(title="[info]Example Inputs[/]", show_header=False, box=box.SIMPLE)
    table.add_column("Examples", style="cyan")

    for example in examples:
        table.add_row(example)

    return table


def create_batch_summary_table(inputs: List[str], results: List[Any]) -> Table:
    """Create a summary table for batch processing results"""

    table = Table(title="[header]Batch Processing Summary[/]", box=box.DOUBLE)
    table.add_column("Input", style="info", width=50)
    table.add_column("Status", justify="center", style="result")

    for input_text, result in zip(inputs, results):
        status = "[success]Success[/]" if result else "[danger]Failed[/]"
        truncated_input = (
            input_text[:47] + "..." if len(input_text) > 50 else input_text
        )
        table.add_row(truncated_input, status)

    return table


def show_ollama_spinner(message: str = "Processing with AI..."):
    """Context manager for Ollama API calls with spinner"""
    return console.status(f"[bold blue]{message}", spinner="dots")


def validate_input_with_rich(user_input: str) -> bool:
    """Enhanced input validation with Rich feedback"""

    if not user_input.strip():
        console.print("[warning]Input cannot be empty[/]")
        return False

    if len(user_input) < 10:
        console.print("[warning]Please provide a more detailed job description[/]")
        return False

    console.print("[success]Input validated[/]")
    return True


def get_save_confirmation() -> bool:
    """Get user confirmation for saving results with Rich styling"""
    return Confirm.ask("[bold green]Save results?[/]")


def get_user_input() -> str:
    """Get user input with Rich styling"""
    return console.input(
        "\n[bold yellow]Please describe the job (or 'quit' to exit):[/] "
    ).strip()


def show_goodbye_message() -> None:
    """Show styled goodbye message"""
    console.print("\n[success]👋 Arrivederci![/]\n")


def show_interrupt_message() -> None:
    """Show styled interrupt message"""
    console.print("\n[warning]User interrupted. Arrivederci![/]\n")


def show_error_message(error: Exception) -> None:
    """Show styled error message"""
    console.print(f"\n[danger]Error: {error}[/]\n")


def show_save_confirmation(json_filename: str, md_filename: str) -> None:
    """Show save confirmation with file paths"""
    console.print(f"[success]OK: Saved JSON to:[/] {json_filename}")
    console.print(f"[success]OK: Saved Report MD to:[/] {md_filename}")
