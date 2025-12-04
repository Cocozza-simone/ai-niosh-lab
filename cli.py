import json
import os
import sys
import time
import asyncio
import warnings
import threading
import re
from datetime import datetime
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from validators import intelligent_llm_prevalidator, validate_params

from job_description_generator import NIOSHJobDescriptionGenerator
from niosh_calculator import NIOSHCalculator, compute_recommended_weight_simplified
from niosh_parameters import NIOSHParameters
from job_analysis import NIOSHJobAnalysisGenerator
from hazard_assessment import NIOSHHazardAssessmentGenerator, HazardAssessmentInput
from redesign_suggestions import (
    NIOSHRedesignSuggestionsGenerator,
    RedesignSuggestionsInput,
)
from consistency_checks import ensure_job_consistency
from tag_extractor import extract_tagged_params, remove_tags
from multi_task_extractor import extract_multi_task_tags
from model_router import call_llm_with_system_prompt

from rich_utils import (
    console,
    create_niosh_results_table,
    create_multipliers_table,
    create_parameters_summary,
    create_analysis_progress,
    create_risk_panel,
    create_text_section_panel,
    create_welcome_panel,
    create_examples_table,
    show_ollama_spinner,
    validate_input_with_rich,
    get_save_confirmation,
    get_user_input,
    show_goodbye_message,
    show_interrupt_message,
    show_error_message,
    show_save_confirmation,
)

LLM_SEMAPHORE = threading.Semaphore(2)

# Suppress warnings
warnings.filterwarnings("ignore")

# Fix Windows encoding issues
if sys.platform == "win32" and hasattr(sys, "excepthook"):
    original_excepthook = sys.excepthook

    def simple_excepthook(exc_type, exc_value, exc_traceback):
        try:
            original_excepthook(exc_type, exc_value, exc_traceback)
        except (UnicodeEncodeError, UnicodeDecodeError):
            print(f"Exception: {exc_type.__name__}: {exc_value}")
        except:
            pass

    sys.excepthook = simple_excepthook


def classify_sentence(text: str, model="llama3.2") -> str:
    system = """
You are an expert in ergonomic task analysis and the Revised NIOSH Lifting Equation.

Classify the following lifting scenario into ONLY ONE category:

- multi
- repetitive
- single

Output ONLY one word.
    """

    with LLM_SEMAPHORE:
        response = (
            call_llm_with_system_prompt(
                model=model,
                system_prompt=system,
                user_prompt=f"Scenario: {text}",
                temperature=0.0,
            )
            .strip()
            .lower()
        )

    if "multi" in response:
        return "multi"
    if "repetitive" in response:
        return "repetitive"
    return "single"


def map_extractor_to_calculator(tasks_data_raw: List[dict]) -> List[dict]:
    """
    Mappa i dati estratti da extract_multi_task_tags (chiavi brevi H0, V0...)
    al formato richiesto da NIOSHCalculator (chiavi lunghe horizontal_origin...).
    Gestisce anche i valori di default.
    """
    tasks_data = []
    for t in tasks_data_raw:
        # Valori di default se mancanti
        w = t.get("weight")
        if w is None:
            w = 0.0

        h0 = t.get("H0")
        if h0 is None:
            h0 = 25.0

        v0 = t.get("V0")
        if v0 is None:
            v0 = 30.0

        # Se destinazione manca, assumiamo valori di default ragionevoli
        h1 = t.get("H1")
        if h1 is None:
            h1 = h0

        v1 = t.get("V1")
        if v1 is None:
            v1 = v0 + 10.0  # Lift di 10 pollici se non specificato

        tasks_data.append(
            {
                "task_description": f"Task {t.get('task', '?')}",
                "weight": w,
                "horizontal_origin": h0,
                "horizontal_destination": h1,
                "vertical_origin": v0,
                "vertical_destination": v1,
                "asymmetry_angle": t.get("A") or 0.0,
                "frequency": t.get("F") or 0.2,
                "duration": t.get("duration") or "2-8h",
                "coupling": t.get("coupling") or "good",
                "significant_control": True,  # Assumiamo controllo significativo per multi-task
            }
        )
    return tasks_data


def batch_process_file(
    input_file: str,
    output_dir: str = "batch_reports",
    model: str = "gemma3:12b",
    max_workers: int = 8,  # ← puoi alzarlo a 6–8 se hai CPU/GPU buona
):
    """
    Versione ottimizzata di batch_process_file:
    - Classificazione tramite LLM (single / repetitive / multi)
    - Parallel processing tramite ThreadPoolExecutor
    - Prompt caching attivo (richiede patch in model_router)
    """

    Path(output_dir).mkdir(exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        scenarios = [line.strip() for line in f.readlines() if line.strip()]

    jd_gen = NIOSHJobDescriptionGenerator(model=model)
    ja_gen = NIOSHJobAnalysisGenerator(model=model)
    hz_gen = NIOSHHazardAssessmentGenerator(model=model)
    rs_gen = NIOSHRedesignSuggestionsGenerator(model=model)
    calc = NIOSHCalculator()

    def process_single_scenario(idx: int, text: str):
        print(f"[Worker] Processing scenario {idx}: {text}")

        # 1) Classificazione LLM
        task_type = classify_sentence(text)
        is_multi = task_type == "multi"

        # 2) Job description (LLM)
        with LLM_SEMAPHORE:
            result = jd_gen.analyze_job(text,task_type)
            jd_text = result["description"]
            params = result["parameters"]

        # ➜ FIX: rimuoviamo i TAG dalla Job Description appena generata
        jd_text_clean = remove_tags(jd_text)

        # 3) SINGLE-TASK ============================================
        if not is_multi:
            params_raw = extract_tagged_params(jd_text)

            with LLM_SEMAPHORE:
                params_full = intelligent_llm_prevalidator(params_raw, jd_text)

            params = NIOSHParameters(**params_full)

            calc_result = calc.compute(params)

            # Job Analysis (pulito)
            with LLM_SEMAPHORE:
                ja_text = remove_tags(ja_gen.generate_job_analysis(params,task_type))

            # Hazard assessment
            hz_input = HazardAssessmentInput(
                task_description=jd_text_clean,  # <-- JD SENZA TAG
                weight_lbs=params.weight,
                rwl_origin_lbs=calc_result.rwl_origin_lbs,
                rwl_dest_lbs=calc_result.rwl_destination_lbs,
                li_origin=calc_result.li_origin,
                li_dest=calc_result.li_destination,
                significant_control=params.significant_control,
            )
            with LLM_SEMAPHORE:
                ha_text = remove_tags(hz_gen.generate_hazard_assessment(hz_input,task_type))

            # Redesign suggestions
            rs_input = RedesignSuggestionsInput(
                task_description=jd_text_clean,  # <-- JD SENZA TAG
                h_origin=params.horizontal_origin,
                h_dest=params.horizontal_destination,
                v_origin=params.vertical_origin,
                v_dest=params.vertical_destination,
                a_origin=params.asymmetry_angle,
                a_dest=params.asymmetry_angle,
                frequency_lifts_per_min=params.frequency,
                duration_class=params.duration,
                coupling=params.coupling,
                significant_control=params.significant_control,
                weight_lbs=params.weight,
                rwl_origin_lbs=calc_result.rwl_origin_lbs,
                rwl_dest_lbs=calc_result.rwl_destination_lbs,
                li_origin=calc_result.li_origin,
                li_dest=calc_result.li_destination,
                multipliers_origin=calc_result.multipliers_origin,
                multipliers_destination=calc_result.multipliers_destination,
                risk_category=calc_result.risk_category,
                risk_comment=calc_result.risk_comment,
            )
            with LLM_SEMAPHORE:
                rs_text = rs_gen.generate_redesign_suggestions(rs_input,task_type)

            report_type = "single_task"

        # 4) MULTI-TASK =============================================
        else:
            tasks_data_raw = extract_multi_task_tags(jd_text)

            tasks_data = map_extractor_to_calculator(tasks_data_raw)

            calc_result = calc.compute_multi_task(
                tasks_data, job_description=jd_text_clean
            )

            with LLM_SEMAPHORE:
                ja_text = remove_tags(
                    ja_gen.generate_multi_task_job_analysis(calc_result.as_dict(),task_type)
                )
            with LLM_SEMAPHORE:
                ha_text = remove_tags(
                    hz_gen.generate_multi_task_hazard_assessment(calc_result)
                )
            with LLM_SEMAPHORE:
                rs_text = remove_tags(rs_gen.generate_multi_task_redesign(calc_result))

            report_type = "multi_task"

        # 5) Salvataggio ============================================
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in text[:40])

        # Create subdirectory for task type
        type_dir = Path(output_dir) / task_type
        type_dir.mkdir(parents=True, exist_ok=True)

        json_path = type_dir / f"{safe_name}_{idx}.json"
        md_path = type_dir / f"{safe_name}_{idx}.md"

        json_data = {
            "input": text,
            "timestamp": timestamp,
            "description": jd_text_clean,  # <-- SALVIAMO LA VERSIONE SENZA TAG
            "analysis": {
                "job_analysis": ja_text,
                "hazard_assessment": ha_text,
                "redesign_suggestions": rs_text,
            },
            "calculations": (
                calc_result.as_dict()
                if report_type == "single_task"
                else {
                    "CLI": calc_result.cli,
                    "risk_category": calc_result.risk_category,
                    "risk_comment": calc_result.risk_comment,
                    "tasks": [t.__dict__ for t in calc_result.tasks],
                }
            ),
        }

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2, ensure_ascii=False)

        report_md = f"""# NIOSH Analysis Report

    **Scenario:** {text}  
    **Generated:** {timestamp}  
    **Model:** {model}  

    ## Job Description
    {jd_text_clean}

    ## Job Analysis
    {ja_text}

    ## Hazard Assessment
    {ha_text}

    ## Redesign Suggestions
    {rs_text}
    """
        # Clean tags from Markdown report
        report_md = re.sub(r"\[[A-Z0-9]+:[^\]]+\]", "", report_md)
        report_md = re.sub(r"[ \t]+", " ", report_md)

        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write(report_md)

        return idx, json_path, md_path

    # ================================================
    #   PARALLEL EXECUTION 🚀
    # ================================================
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, text in enumerate(scenarios, start=1):
            futures.append(executor.submit(process_single_scenario, idx, text))

        for future in as_completed(futures):
            idx, json_path, md_path = future.result()
            print(f"✓ Scenario {idx} completed → {json_path}")

    print("\n=== Parallel Batch Processing Completed ===")


# Note: retry_logic import skipped to avoid import issues
def detect_multi_task(user_text: str) -> bool:
    """
    Ritorna True se il job sembra MULTI-TASK secondo logica NIOSH.
    Ritorna False se sembra SINGLE-TASK.
    """

    txt = user_text.lower()

    # Pattern molto forti (quasi sempre multi-task)
    strong_multi = [
        r"\btier\b",
        r"\btiers\b",
        r"\blevels\b",
        r"\blayers\b",
        r"\bmultiple tasks\b",
        r"\bdifferent tasks\b",
        r"\bvarious tasks\b",
        r"\bmulti[- ]?task\b",
        r"\bseveral tasks\b",
        r"\bdistinct tasks\b",
        r"\bdifferent heights\b",
        r"\bdifferent shelves\b",
        r"\bmultiple shelves\b",
        r"\btask 1\b",
        r"\btask 2\b",
        r"\btask 3\b",
        r"\bfirst task\b",
        r"\bsecond task\b",
    ]

    for pat in strong_multi:
        if re.search(pat, txt):
            return True

    # Pattern medio-forti (es. depalletizing)
    medium_multi = [
        r"\b(top|middle|bottom) (shelf|tier|row)\b",
        r"\bfrom .* to .* to\b",  # es. "from cart to shelf 1 to shelf 2"
        r"\bthree shelves\b",
        r"\bfive tiers\b",
        r"\bstacked\b",
        r"\bstack of\b",
        r"\bunload(?:ing)? multiple\b",
    ]

    for pat in medium_multi:
        if re.search(pat, txt):
            return True

    # Pattern basato su numeri multipli di altezze/posizioni
    # se riconosciamo 3+ valori verticali → probabile multitask
    vertical_values = re.findall(r"\b(\d+)\s*(?:in|cm)\b", txt)
    if len(vertical_values) >= 3:
        return True

    # Se menziona 3 o più destinazioni/origini differenti
    if len(re.findall(r"\b(origin|destination)\b", txt)) >= 3:
        return True

    # Se menziona 3 o più oggetti diversi
    if len(re.findall(r"\bbox\b|\bcontainer\b|\broll\b|\bcan\b", txt)) >= 3:
        return True

    # Nessun pattern multi-task rilevato
    return False


def get_model_selection():
    """Ask user to choose between single model and multi-model mode"""
    console.print("\n[bold cyan]SELECT MODEL MODE[/bold cyan]")
    console.print("[info]Choose how you want to generate the analysis reports:[/]\n")

    console.print("1. [green]Single Model[/green] - Use only gemma3:12b (Ollama)")
    console.print("   Faster processing with single model output\n")

    console.print(
        "2. [yellow]Multi-Model[/yellow] - Generate reports with all available models:"
    )
    console.print("   - gemma3:12b (Ollama)")
    console.print("   - llama3.2:latest (Ollama)")
    console.print("   - llama3.1:8b (Ollama)")
    console.print("   - gemini-2.5 (Google, if available)")
    console.print("   Each model will generate a separate report file sequentially\n")

    while True:
        choice = console.input("[info]Select mode (1-2): [/]").strip()
        if choice in ["1", "2"]:
            return choice == "2"  # True for multi-model, False for single model
        console.print("[danger]Invalid choice. Please select 1-2.[/]")


def get_processing_mode():
    """Ask user to choose between automatic file processing or interactive mode"""
    console.print("\n[bold cyan]SELECT PROCESSING MODE[/bold cyan]")
    console.print("[info]Choose how you want to process ergonomic analysis:[/]\n")

    console.print(
        "1. [green]Interactive Mode[/green] - Analyze tasks one by one with real-time input"
    )
    console.print("   Type descriptions interactively and see immediate results\n")

    console.print(
        "2. [blue]Batch Mode[/blue] - Process multiple tasks from a file automatically"
    )
    console.print("   Load scenarios from a text file (one task per line)")
    console.print("   Option to use single or multi-model generation\n")

    while True:
        choice = console.input("[info]Select processing mode (1-2): [/]").strip()
        if choice == "1":
            return "interactive"
        elif choice == "2":
            return "batch"
        else:
            console.print("[danger]Invalid choice. Please select 1-2.[/]")


def get_batch_file_info():
    """Get file path and processing options for batch mode"""
    console.print("\n[bold cyan]BATCH PROCESSING CONFIGURATION[/bold cyan]")

    # Use default file path automatically (relative path)
    default_file_path = "scenari_niosh_fps.txt"
    file_path = default_file_path
    lines = []

    # Check if default file exists and has content
    if not os.path.exists(file_path):
        console.print(f"[yellow]Default file '{file_path}' not found.[/]")

        # Fallback: ask user for file path
        while True:
            file_path = console.input("[info]Enter path to scenarios file: [/]").strip()
            if not file_path:
                console.print("[yellow]Please enter a file path.[/]")
                continue

            if not os.path.exists(file_path):
                console.print(f"[danger]File not found: {file_path}[/]")
                retry = console.input("[info]Try again? (y/n): [/]").strip().lower()
                if retry in ["y", "yes"]:
                    continue
                else:
                    return None
            else:
                break
    else:
        console.print(f"[info]Using default scenarios file: {file_path}[/]")

    # Check if file has content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.startswith("#")
            ]
            if not lines:
                console.print("[yellow]Warning: No valid scenarios found in file.[/]")
                retry = (
                    console.input("[info]Try another file? (y/n): [/]").strip().lower()
                )
                if retry in ["y", "yes"]:
                    # Fallback to manual file selection
                    return get_batch_file_info_manual()
                else:
                    return None

            console.print(f"[success]Found {len(lines)} scenarios in file[/]")
    except Exception as e:
        console.print(f"[danger]Error reading file: {e}[/]")
        retry = console.input("[info]Try another file? (y/n): [/]").strip().lower()
        if retry in ["y", "yes"]:
            # Fallback to manual file selection
            return get_batch_file_info_manual()
        else:
            return None

    # Get number of examples (optional)
    num_examples = None
    limit_input = console.input(
        "[info]Process all scenarios or limit to a number? (Enter number or 'all'): [/]"
    ).strip()
    if limit_input.lower() != "all" and limit_input.isdigit():
        num_examples = int(limit_input)
        console.print(f"[info]Will process first {num_examples} scenarios[/]")
    else:
        console.print(f"[info]Will process all {len(lines)} scenarios[/]")

    # Get model selection
    use_multi_model = get_model_selection()

    return {
        "file_path": file_path,
        "num_examples": num_examples,
        "use_multi_model": use_multi_model,
    }


def get_batch_file_info_manual():
    """Manual file selection fallback"""
    console.print("\n[bold yellow]MANUAL FILE SELECTION[/bold yellow]")

    # Get file path
    while True:
        file_path = console.input("[info]Enter path to scenarios file: [/]").strip()
        if not file_path:
            console.print("[yellow]Please enter a file path.[/]")
            continue

        if not os.path.exists(file_path):
            console.print(f"[danger]File not found: {file_path}[/]")
            retry = console.input("[info]Try again? (y/n): [/]").strip().lower()
            if retry in ["y", "yes"]:
                continue
            else:
                return None

        # Check if file has content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [
                    line.strip()
                    for line in f.readlines()
                    if line.strip() and not line.startswith("#")
                ]
                if not lines:
                    console.print(
                        "[yellow]Warning: No valid scenarios found in file.[/]"
                    )
                    retry = (
                        console.input("[info]Try another file? (y/n): [/]")
                        .strip()
                        .lower()
                    )
                    if retry in ["y", "yes"]:
                        continue
                    else:
                        return None

                console.print(f"[success]Found {len(lines)} scenarios in file[/]")
        except Exception as e:
            console.print(f"[danger]Error reading file: {e}[/]")
            retry = console.input("[info]Try again? (y/n): [/]").strip().lower()
            if retry in ["y", "yes"]:
                continue
            else:
                return None

        break

    # Get number of examples (optional)
    num_examples = None
    limit_input = console.input(
        "[info]Process all scenarios or limit to a number? (Enter number or 'all'): [/]"
    ).strip()
    if limit_input.lower() != "all" and limit_input.isdigit():
        num_examples = int(limit_input)
        console.print(f"[info]Will process first {num_examples} scenarios[/]")
    else:
        console.print(f"[info]Will process all {len(lines)} scenarios[/]")

    # Get model selection
    use_multi_model = get_model_selection()

    return {
        "file_path": file_path,
        "num_examples": num_examples,
        "use_multi_model": use_multi_model,
    }


def check_gemini_availability():
    """Check if Gemini API is available"""
    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")

        return api_key is not None and genai is not None
    except ImportError:
        return False


def generate_multi_model_reports(
    user_input, generator, calc, ja_gen, hz_gen, rs_gen, task_type="single_task"
):
    """Generate reports using multiple models sequentially"""

    # Define available models
    models_config = [
        {
            "name": "gemma3:12b",
            "prefix": "gemma3-12b",
            "type": "ollama",
        },
        {
            "name": "llama3.2:latest",
            "prefix": "llama3.2",
            "type": "ollama",
        },
        {
            "name": "llama3.1:8b",
            "prefix": "llama3.1-8b",
            "type": "ollama",
        },
    ]

    # Check if Gemini is available
    gemini_available = check_gemini_availability()
    if gemini_available:
        models_config.append(
            {
                "name": "gemini-2.5-pro",  # Used for model initialization
                "prefix": "gemini-2.5",
                "type": "google",
            }
        )

    console.print(
        f"\n[info]Generating reports with {len(models_config)} models sequentially...[/]"
    )

    success_count = 0

    for i, model_config in enumerate(models_config, 1):
        try:
            console.print(
                f"\n[cyan]Processing with model {i}/{len(models_config)}: {model_config['name']}[/cyan]"
            )

            if model_config["type"] == "google":
                # Generate with Gemini
                files = generate_report_with_gemini(
                    user_input,
                    model_config,  # Pass entire config
                    generator,
                    calc,
                    ja_gen,
                    hz_gen,
                    rs_gen,
                    task_type=task_type,
                )
            else:
                # Generate with Ollama model
                files = generate_report_with_ollama(
                    user_input,
                    model_config,
                    generator,
                    calc,
                    ja_gen,
                    hz_gen,
                    rs_gen,
                    task_type=task_type,
                )

            if files:
                success_count += 1
                console.print(
                    f"[green]OK: Successfully generated report with {model_config['name']}[/]"
                )
                console.print(
                    f"  Files: {', '.join([os.path.basename(f) for f in files])}"
                )
            else:
                console.print(
                    f"[red]ERROR: Failed to generate report with {model_config['name']}[/]"
                )

        except Exception as e:
            console.print(
                f"[red]ERROR: Exception with {model_config['name']}: {str(e)}[/]"
            )
            continue

    console.print(f"\n[bold green]Multi-model generation complete![/]")
    console.print(
        f"[info]Successfully generated: {success_count}/{len(models_config)} reports[/]"
    )
    console.print("[info]All reports saved to 'report/' directory[/]")


def generate_report_with_ollama(
    user_input,
    model_config,
    base_generator,
    calc,
    ja_gen,
    hz_gen,
    rs_gen,
    task_type="single_task",
):
    """Generate a report using a specific Ollama model"""

    try:
        # Step 0: Set the correct model for all generators!
        model_name = model_config["name"]

        # We need to update the model on all instances to ensure we use the requested one
        # and not the default or the last used one.
        if hasattr(base_generator, "model"):
            base_generator.model = model_name
        if hasattr(ja_gen, "model"):
            ja_gen.model = model_name
        if hasattr(hz_gen, "model"):
            hz_gen.model = model_name
        if hasattr(rs_gen, "model"):
            rs_gen.model = model_name

        console.print(f"[info]Switched generators to model: {model_name}[/]")

        # Step 1: Extract parameters
        result = base_generator.analyze_job(user_input)
        if not result or not result.get("parameters"):
            return []

        description = result["description"]
        description_clean = remove_tags(description)
        params_dict = result["parameters"]

        # Step 2: Perform calculations using the same pattern as ss/cli.py
        p = NIOSHParameters(**params_dict)
        calc_result = calc.compute(p)

        # Step 3: Generate report sections using the same pattern as ss/cli.py

        job_analysis = remove_tags(ja_gen.generate_job_analysis(p))

        hz_input = HazardAssessmentInput(
            task_description=description_clean,
            weight_lbs=p.weight,
            rwl_origin_lbs=calc_result.rwl_origin_lbs,
            rwl_dest_lbs=calc_result.rwl_destination_lbs,
            li_origin=calc_result.li_origin,
            li_dest=calc_result.li_destination,
            significant_control=p.significant_control,
        )
        hazard_assessment = remove_tags(hz_gen.generate_hazard_assessment(hz_input))

        rs_input = RedesignSuggestionsInput(
            task_description=description_clean,
            h_origin=p.horizontal_origin,
            h_dest=p.horizontal_destination,
            v_origin=p.vertical_origin,
            v_dest=p.vertical_destination,
            a_origin=p.asymmetry_angle,
            a_dest=p.asymmetry_angle,
            frequency_lifts_per_min=p.frequency,
            duration_class=p.duration,
            coupling=p.coupling,
            significant_control=p.significant_control,
            weight_lbs=p.weight,
            rwl_origin_lbs=calc_result.rwl_origin_lbs,
            rwl_dest_lbs=calc_result.rwl_destination_lbs,
            li_origin=calc_result.li_origin,
            li_dest=calc_result.li_destination,
            multipliers_origin=calc_result.multipliers_origin,
            multipliers_destination=calc_result.multipliers_destination,
            risk_category=calc_result.risk_category,
            risk_comment=calc_result.risk_comment,
        )
        redesign_suggestions = rs_gen.generate_redesign_suggestions(rs_input)
        # Step 4: Create report and save files
        timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        professional_report = f"""# NIOSH Lifting Analysis Report

**Generated:** {timestamp}
**Task:** {user_input}
**Model:** {model_config['name']}

## Job Description
{description}

## Job Analysis
{job_analysis}

## Hazard Assessment
{hazard_assessment}

## Redesign Suggestions
{redesign_suggestions}

"""

        # Create report directory
        report_dir = Path("report") / task_type
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        safe_name = "".join([c if c.isalnum() else "_" for c in user_input[:30]]).strip(
            "_"
        )
        datestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        base_filename = f"{model_config['prefix']}_report_{safe_name}_{datestamp}"

        files_saved = []

        # Save JSON
        json_data = {
            "metadata": {
                "timestamp": timestamp,
                "analysis_type": "single_task",
                "generated_with": model_config["name"],
                "model": model_config["name"],
            },
            "input": user_input,
            "description": description,
            "parameters": params_dict,
            "calculations": {
                "rwl_origin_lbs": calc_result.rwl_origin_lbs,
                "rwl_destination_lbs": calc_result.rwl_destination_lbs,
                "li_origin": calc_result.li_origin,
                "li_destination": calc_result.li_destination,
                "multipliers_origin": calc_result.multipliers_origin,
                "multipliers_destination": calc_result.multipliers_destination,
                "risk_category": calc_result.risk_category,
                "risk_comment": calc_result.risk_comment,
            },
            "job_analysis_text": job_analysis,
            "hazard_assessment_text": hazard_assessment,
            "redesign_suggestions_text": redesign_suggestions,
        }

        json_filename = report_dir / f"{base_filename}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            __import__("json").dump(json_data, f, indent=2, ensure_ascii=False)
        files_saved.append(str(json_filename))

        # Save Markdown
        md_filename = report_dir / f"{base_filename}.md"
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(professional_report)
        files_saved.append(str(md_filename))

        return files_saved

    except Exception as e:
        console.print(
            f"[red]Error generating report with {model_config['name']}: {e}[/]"
        )
        return []


def generate_report_with_gemini(
    user_input,
    model_config,
    generator,
    calc,
    ja_gen,
    hz_gen,
    rs_gen,
    task_type="single_task",
):
    """Generate a report using Google Gemini with strict retry logic (no fallback to Ollama)"""

    try:
        # Extract model info
        prefix = model_config["prefix"]
        # Use the name provided in config, not hardcoded 'gemini-2.5-pro'
        target_model = model_config["name"]

        console.print(f"[info]Generating {prefix} report using unified pipeline...[/]")

        # --- STRICT RETRY LOGIC FOR GEMINI ---
        max_retries = 10  # More retries since we are waiting
        retry_delay = 10  # Start with 10 seconds

        # Helper to set model and wait if needed
        def set_model_strict(generator_instance, generator_name):
            if hasattr(generator_instance, "model"):
                generator_instance.model = target_model
            console.print(f"[info]Setting {generator_name} model to: {target_model}[/]")
            return True

        # Set models for all generators
        set_model_strict(generator, "Job Description")
        set_model_strict(ja_gen, "Job Analysis")
        set_model_strict(hz_gen, "Hazard Assessment")
        set_model_strict(rs_gen, "Redesign Suggestions")

        # Step 1: Analyze job using standard pipeline with retry
        result = None
        for attempt in range(max_retries):
            try:
                console.print(
                    f"[info]Attempt {attempt+1}/{max_retries}: Analyzing job with {generator.model}...[/]"
                )
                result = generator.analyze_job(user_input)

                if result and result.get("parameters"):
                    console.print(
                        f"[green]✅ Successfully analyzed job with {generator.model}[/]"
                    )
                    break
                else:
                    raise Exception("Parameters not extracted")

            except Exception as e:
                err_msg = str(e)
                console.print(f"[yellow]Error with {generator.model}: {err_msg}[/]")
                if (
                    "429" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                    or "quota" in err_msg.lower()
                ):
                    wait_time = retry_delay * (attempt + 1)
                    console.print(
                        f"[yellow]Quota exceeded. Waiting {wait_time}s before retry...[/]"
                    )
                    time.sleep(wait_time)
                else:
                    # For other errors, still wait a bit but maybe less
                    time.sleep(5)
                continue

        # Check if we successfully got parameters
        if not result or result.get("parameters") is None:
            console.print(
                "[red]ERROR: Failed to extract parameters after all retries.[/]"
            )
            return []

        description = result["description"]
        description_clean = remove_tags(description)
        params_dict = result["parameters"]

        # Step 2: Perform calculations
        p = NIOSHParameters(**params_dict)
        calc_result = calc.compute(p)

        # Step 3: Generate sections using specialized generators with Gemini

        # Helper for generating sections with retry
        def generate_section_with_retry(gen_func, input_data, section_name):
            for attempt in range(max_retries):
                try:
                    return gen_func(input_data)
                except Exception as e:
                    err_msg = str(e)
                    console.print(
                        f"[yellow]Error generating {section_name}: {err_msg}[/]"
                    )
                    if (
                        "429" in err_msg
                        or "RESOURCE_EXHAUSTED" in err_msg
                        or "quota" in err_msg.lower()
                    ):
                        wait_time = retry_delay * (attempt + 1)
                        console.print(
                            f"[yellow]Quota exceeded. Waiting {wait_time}s...[/]"
                        )
                        time.sleep(wait_time)
                    else:
                        time.sleep(5)
            raise Exception(f"Failed to generate {section_name} after retries")

        # Generate sections
        try:
            job_analysis = remove_tags(
                generate_section_with_retry(
                    ja_gen.generate_job_analysis, p, "Job Analysis"
                )
            )
            console.print(f"[green]✅ Job analysis generated with {ja_gen.model}[/]")

            hz_input = HazardAssessmentInput(
                task_description=description_clean,
                weight_lbs=p.weight,
                rwl_origin_lbs=calc_result.rwl_origin_lbs,
                rwl_dest_lbs=calc_result.rwl_destination_lbs,
                li_origin=calc_result.li_origin,
                li_dest=calc_result.li_destination,
                significant_control=p.significant_control,
            )
            hazard_assessment = remove_tags(
                generate_section_with_retry(
                    hz_gen.generate_hazard_assessment, hz_input, "Hazard Assessment"
                )
            )
            console.print(
                f"[green]✅ Hazard assessment generated with {hz_gen.model}[/]"
            )

            rs_input = RedesignSuggestionsInput(
                task_description=description_clean,
                h_origin=p.horizontal_origin,
                h_dest=p.horizontal_destination,
                v_origin=p.vertical_origin,
                v_dest=p.vertical_destination,
                a_origin=p.asymmetry_angle,
                a_dest=p.asymmetry_angle,
                frequency_lifts_per_min=p.frequency,
                duration_class=p.duration,
                coupling=p.coupling,
                significant_control=p.significant_control,
                weight_lbs=p.weight,
                rwl_origin_lbs=calc_result.rwl_origin_lbs,
                rwl_dest_lbs=calc_result.rwl_destination_lbs,
                li_origin=calc_result.li_origin,
                li_dest=calc_result.li_destination,
                multipliers_origin=calc_result.multipliers_origin,
                multipliers_destination=calc_result.multipliers_destination,
                risk_category=calc_result.risk_category,
                risk_comment=calc_result.risk_comment,
            )
            redesign_suggestions = generate_section_with_retry(
                rs_gen.generate_redesign_suggestions, rs_input, "Redesign Suggestions"
            )
            console.print(
                f"[green]✅ Redesign suggestions generated with {rs_gen.model}[/]"
            )

        except Exception as e:
            console.print(f"[red]Failed to generate report sections: {e}[/]")
            return []

        # Step 4: Create report and save files
        timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        professional_report = f"""# NIOSH Lifting Analysis Report

**Generated:** {timestamp}
**Task:** {user_input}
**Model:** {ja_gen.model}
**Note:** Generated with strict Gemini retry logic

## Job Description
{description}

## Job Analysis
{job_analysis}

## Hazard Assessment
{hazard_assessment}

## Redesign Suggestions
{redesign_suggestions}


"""

        # Create report directory
        report_dir = Path("report") / task_type
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        safe_name = "".join([c if c.isalnum() else "_" for c in user_input[:30]]).strip(
            "_"
        )
        datestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        base_filename = f"{prefix}_report_{safe_name}_{datestamp}"

        files_saved = []

        # Save JSON
        json_data = {
            "metadata": {
                "timestamp": timestamp,
                "analysis_type": "single_task",
                "generated_with": "NIOSH Analysis System (Gemini Strict)",
                "model": ja_gen.model,
                "retry_enabled": True,
            },
            "input": user_input,
            "description": description,
            "parameters": params_dict,
            "calculations": {
                "rwl_origin_lbs": calc_result.rwl_origin_lbs,
                "rwl_destination_lbs": calc_result.rwl_destination_lbs,
                "li_origin": calc_result.li_origin,
                "li_destination": calc_result.li_destination,
                "multipliers_origin": calc_result.multipliers_origin,
                "multipliers_destination": calc_result.multipliers_destination,
                "risk_category": calc_result.risk_category,
                "risk_comment": calc_result.risk_comment,
            },
            "job_analysis_text": job_analysis,
            "hazard_assessment_text": hazard_assessment,
            "redesign_suggestions_text": redesign_suggestions,
        }

        json_filename = report_dir / f"{base_filename}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            __import__("json").dump(json_data, f, indent=2, ensure_ascii=False)
        files_saved.append(str(json_filename))

        # Save Markdown
        md_filename = report_dir / f"{base_filename}.md"
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(professional_report)
        files_saved.append(str(md_filename))

        console.print(f"[green]+ {prefix} report generated successfully![/]")
        return files_saved

    except Exception as e:
        console.print(f"[red]Gemini generation error: {e}[/]")
        return []


def generate_single_model_report(
    user_input,
    generator,
    calc,
    ja_gen,
    hz_gen,
    rs_gen,
    auto_save=False,
    task_type="single_task",
):
    """Generate report using single model in SINGLE-TASK mode only."""

    try:
        # Step 1: Unified pipeline → ma forziamo SINGLE-TASK
        result = generator.analyze_job(user_input)
        if not result:
            console.print("[danger]ERROR: Job analysis failed (no result returned).[/]")
            return

        mode = result.get("mode", "single-task")
        if mode != "single-task":
            console.print("[danger]ERROR: Input detected as MULTI-TASK by analyzer.[/]")
            console.print(
                "[info]Use the multi-task / multi-model pipeline for this description.[/]"
            )
            return

        params_raw = result.get("parameters")
        if not params_raw:
            console.print("[danger]ERROR: Parameters not extracted.[/]")
            return

        # ---- Recupero descrizione con tag ----
        description = (
            result.get("description")
            or result.get("description_clean")
            or result.get("description_with_tags")
            or ""
        )

        # ---- FIX: Rimuoviamo i tag dalla Job Description ----
        description_clean = remove_tags(description)

        # ---- PREVALIDATORE LLM ----
        params_filled = intelligent_llm_prevalidator(params_raw, description)

        # ---- DEFAULTS per valori None ----
        default_values = {
            "weight": 25.0,
            "horizontal_origin": 20.0,
            "horizontal_destination": 20.0,
            "vertical_origin": 30.0,
            "vertical_destination": 30.0,
            "asymmetry_angle": 0.0,
            "frequency": 0.1,
            "duration": "<1h",
            "coupling": "fair",
            "significant_control": False,
            "gender": "M",
            "age": 25,
            "one_limb_lifting": False,
            "two_operators_lifting": False,
        }

        for key in default_values:
            if key in params_filled and params_filled[key] is None:
                params_filled[key] = default_values[key]
            elif key not in params_filled:
                params_filled[key] = default_values[key]

        # ---- VALIDAZIONE ----
        ok, err = validate_params(params_filled)
        if not ok:
            console.print(f"[danger]Validation error: {err}[/]")
            console.print("[yellow]Using fallback default parameters...[/]")
            params_filled = default_values.copy()

        # ---- COSTRUZIONE PARAMETRI FINALI ----
        try:
            p = NIOSHParameters(**params_filled)
        except Exception as e:
            console.print(f"[danger]Error creating NIOSHParameters: {e}[/]")
            console.print("[yellow]Using basic fallback parameters...[/]")
            p = NIOSHParameters(
                weight=25.0,
                horizontal_origin=20.0,
                horizontal_destination=20.0,
                vertical_origin=30.0,
                vertical_destination=30.0,
                asymmetry_angle=0.0,
                frequency=0.1,
                duration="<1h",
                coupling="fair",
                significant_control=False,
            )

        # Step 2: RNLE
        calc_result = calc.compute(p)

        # Step 3: Generate report sections (SINGLE-TASK)

        # ---- Job Analysis ----
        job_analysis_tagged = ja_gen.generate_job_analysis(p)
        job_analysis = remove_tags(job_analysis_tagged)

        # ---- Hazard Assessment ----
        hz_input = HazardAssessmentInput(
            task_description=description_clean,  # JD senza TAG
            weight_lbs=p.weight,
            rwl_origin_lbs=calc_result.rwl_origin_lbs,
            rwl_dest_lbs=calc_result.rwl_destination_lbs,
            li_origin=calc_result.li_origin,
            li_dest=calc_result.li_destination,
            significant_control=p.significant_control,
        )
        hazard_assessment_tagged = hz_gen.generate_hazard_assessment(hz_input)
        hazard_assessment = remove_tags(hazard_assessment_tagged)

        # ---- Redesign Suggestions ----
        rs_input = RedesignSuggestionsInput(
            task_description=description_clean,  # JD senza TAG
            h_origin=p.horizontal_origin,
            h_dest=p.horizontal_destination,
            v_origin=p.vertical_origin,
            v_dest=p.vertical_destination,
            a_origin=p.asymmetry_angle,
            a_dest=p.asymmetry_angle,
            frequency_lifts_per_min=p.frequency,
            duration_class=p.duration,
            coupling=p.coupling,
            significant_control=p.significant_control,
            weight_lbs=p.weight,
            rwl_origin_lbs=calc_result.rwl_origin_lbs,
            rwl_dest_lbs=calc_result.rwl_destination_lbs,
            li_origin=calc_result.li_origin,
            li_dest=calc_result.li_destination,
            multipliers_origin=calc_result.multipliers_origin,
            multipliers_destination=calc_result.multipliers_destination,
            risk_category=calc_result.risk_category,
            risk_comment=calc_result.risk_comment,
        )
        redesign_suggestions_tagged = rs_gen.generate_redesign_suggestions(rs_input)
        redesign_suggestions = remove_tags(redesign_suggestions_tagged)

        # Step 4: Create report
        timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        professional_report = f"""# NIOSH Lifting Analysis Report

**Generated:** {timestamp}
**Task:** {user_input}

## Job Description
{description_clean}

## Job Analysis
{job_analysis}

## Hazard Assessment
{hazard_assessment}

## Redesign Suggestions
{redesign_suggestions}
"""

        # Step 5: Save files
        if auto_save or get_save_confirmation():
            import json

            report_dir = Path("report")
            report_dir.mkdir(exist_ok=True)

            safe_name = "".join(
                [c if c.isalnum() else "_" for c in user_input[:30]]
            ).strip("_")
            base_filename = f"niosh_single_{safe_name}_{int(__import__('time').time())}"

            json_data = {
                "input_phrase": user_input,
                "description": description_clean,  # JD senza TAG
                "parameters": params_filled,
                "niosh_calculation": calc_result.as_dict(),
                "job_analysis_text": job_analysis,
                "hazard_assessment_text": hazard_assessment,
                "redesign_suggestions_text": redesign_suggestions,
                "timestamp": timestamp,
                "analysis_type": "single_task_single_model",
            }

            json_filename = report_dir / f"{base_filename}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            md_filename = report_dir / f"{base_filename}.md"
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(professional_report)

            console.print(
                f"[success]Report saved: {json_filename} and {md_filename}[/]"
            )

        # Display summary
        console.print("\n[bold green]Analysis Complete![/]")
        console.print(f"[info]LI Origin: {calc_result.li_origin:.2f}[/]")
        if calc_result.li_destination is not None:
            console.print(f"[info]LI Destination: {calc_result.li_destination:.2f}[/]")
        console.print(f"[info]Risk Category: {calc_result.risk_category}[/]")
        console.print(f"[info]RWL Origin: {calc_result.rwl_origin_lbs:.1f} lbs[/]")
        if calc_result.rwl_destination_lbs is not None:
            console.print(
                f"[info]RWL Destination: {calc_result.rwl_destination_lbs:.1f} lbs[/]"
            )

    except Exception as e:
        console.print(f"[danger]Error in analysis: {e}[/]")
        import traceback

        traceback.print_exc()


def interactive_chat():
    """Enhanced interactive chat with Rich formatting following ss/cli.py pattern"""

    # Display welcome message with Rich
    console.print(create_welcome_panel())
    console.print(create_examples_table())

    # Ask for processing mode first
    processing_mode = get_processing_mode()

    if processing_mode == "batch":
        # Batch mode configuration
        batch_config = get_batch_file_info()
        if batch_config is None:
            console.print(
                "[yellow]Batch mode cancelled. Returning to main menu...[/]\n"
            )
            return interactive_chat()

        console.print(f"\n[bold green]STARTING BATCH PROCESSING[/]")
        console.print(f"[info]File: {batch_config['file_path']}[/]")
        if batch_config["num_examples"]:
            console.print(f"[info]Limit: {batch_config['num_examples']} scenarios[/]")
        mode_text = "Multi-Model" if batch_config["use_multi_model"] else "Single Model"
        console.print(f"[info]Mode: {mode_text}[/]\n")

        try:
            batch_process_file(
                input_file=batch_config["file_path"],
                output_dir="batch_reports",
                model="gemma3:12b",
                max_workers=8,
            )
        except Exception as e:
            console.print(f"[danger]Error during batch processing: {e}[/]")

        continue_choice = (
            console.input(
                "\n[info]Process another file or return to interactive mode? (file/interactive): [/]"
            )
            .strip()
            .lower()
        )
        if continue_choice in ["interactive", "file"]:
            return interactive_chat()
        else:
            show_goodbye_message()
            return

    # Interactive mode continues with model selection
    use_multi_model = get_model_selection()
    mode_text = "Multi-Model Mode" if use_multi_model else "Single Model Mode"
    console.print(f"\n[info]Selected: {mode_text}[/]\n")

    console.print("[info]Checking Ollama service health...[/]")
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            console.print("[success] Ollama service is healthy![/]")
        else:
            console.print(
                f"[yellow]Warning: Ollama responded with status {response.status_code}[/]"
            )
            console.print("[yellow]Continuing anyway...[/]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not connect to Ollama: {e}[/]")
        console.print("[yellow]Continuing anyway...[/]")

    if use_multi_model:
        console.print("[info]Google Gemini status: Will be checked during execution[/]")

    console.print("[info]Initializing generators...[/]")
    try:
        generator = NIOSHJobDescriptionGenerator()
        calc = NIOSHCalculator()
        jd_gen = NIOSHJobDescriptionGenerator()
        ja_gen = NIOSHJobAnalysisGenerator()
        hz_gen = NIOSHHazardAssessmentGenerator()
        rs_gen = NIOSHRedesignSuggestionsGenerator()
        console.print("[success]All generators ready![/]")
    except Exception as e:
        console.print(f"[danger]Error initializing generators: {e}[/]")
        return

    console.print("[info]Skipping Ollama client initialization (basic mode)[/]")
    console.print("\n[bold green]SYSTEM READY FOR INPUT[/]")

    while True:
        try:
            user_input = get_user_input()

            if user_input.lower() in ["quit", "exit", "q"]:
                show_goodbye_message()
                break

            if not validate_input_with_rich(user_input):
                continue

            task_type = classify_sentence(user_input)
            console.print(f"[info]Detected task type: [bold]{task_type}[/]")

            if task_type == "repetitive":
                console.print(
                    "[info]Repetitive task detected → treating as SINGLE-TASK (RNLE rule)"
                )
                # Keep task_type as "repetitive" for folder organization
                # task_type = "single"

            is_multi = task_type == "multi"

            if use_multi_model:
                # Multi-model: genera sempre più report, ma usa la pipeline giusta!
                if is_multi:
                    console.print(
                        "[cyan]Using MULTI-TASK pipeline (multi-model mode)[/]"
                    )
                    generate_multi_model_reports(
                        user_input,
                        generator,
                        calc,
                        jd_gen,
                        ja_gen,
                        hz_gen,
                        rs_gen,
                        task_type=task_type,
                    )
                else:
                    console.print(
                        "[cyan]Using SINGLE-TASK pipeline (multi-model mode)[/]"
                    )
                    generate_multi_model_reports(
                        user_input,
                        generator,
                        calc,
                        jd_gen,
                        ja_gen,
                        hz_gen,
                        rs_gen,
                        task_type=task_type,
                    )
            else:
                # Single-model mode
                if is_multi:
                    console.print(
                        "[cyan]Detected MULTI-TASK → switching to multi-task pipeline[/]"
                    )
                    # Qui usi la pipeline multi-task (la stessa del batch)
                    job_desc_result = generator.analyze_job(user_input)
                    jd_text = job_desc_result["description"]
                    tasks_data_raw = extract_multi_task_tags(jd_text)
                    tasks_data = map_extractor_to_calculator(tasks_data_raw)
                    calc_result = calc.compute_multi_task(
                        tasks_data, job_description=user_input
                    )
                    ja_text = remove_tags(
                        ja_gen.generate_multi_task_job_analysis(calc_result.as_dict())
                    )
                    ha_text = remove_tags(
                        hz_gen.generate_multi_task_hazard_assessment(calc_result)
                    )
                    rs_text = remove_tags(
                        rs_gen.generate_multi_task_redesign(calc_result)
                    )
                    # --- Save Report Logic ---
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_dir = Path("reports") / task_type
                    report_dir.mkdir(parents=True, exist_ok=True)

                    base_filename = f"niosh_multitask_{timestamp}"

                    # 1. Construct Full Report (Markdown)
                    full_report_md = f"# NIOSH Multi-Task Analysis Report\n\n"
                    full_report_md += (
                        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    )
                    full_report_md += f"**Job Description:** {jd_text}\n\n"
                    full_report_md += f"## Job Analysis\n\n{ja_text}\n\n"
                    full_report_md += f"## Hazard Assessment\n\n{ha_text}\n\n"
                    full_report_md += f"## Redesign Suggestions\n\n{rs_text}\n\n"

                    # 2. Clean Tags for Final Output (preserving newlines)
                    # Remove [TAG:value] patterns
                    clean_report_md = re.sub(
                        r"\[[a-zA-Z0-9]+:[^\]]+\]", "", full_report_md
                    )
                    # Clean up multiple spaces but preserve newlines
                    clean_report_md = re.sub(r"[ \t]+", " ", clean_report_md)

                    # 3. Save Markdown File
                    md_path = report_dir / f"{base_filename}.md"
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(clean_report_md)

                    # 4. Save JSON Data (Raw + Sections)
                    json_data = {
                        "timestamp": datetime.now().isoformat(),
                        "input": user_input,
                        "calculation_result": calc_result.as_dict(),
                        "sections": {
                            "job_description": jd_text,
                            "job_analysis": ja_text,
                            "hazard_assessment": ha_text,
                            "redesign_suggestions": rs_text,
                        },
                    }

                    json_path = report_dir / f"{base_filename}.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)

                    # 5. User Feedback
                    console.print(f"\n[success]Multi-task analysis completed![/]")
                    console.print(f"[info]Report saved to: {md_path}[/]")
                    console.print(f"[info]Data saved to: {json_path}[/]")

                    # Show preview (clean)
                    console.print("\n[bold]Report Preview:[/]\n")
                    console.print(clean_report_md)
                else:
                    # Single-task normale
                    generate_single_model_report(
                        user_input,
                        generator,
                        calc,
                        ja_gen,
                        hz_gen,
                        rs_gen,
                        task_type=task_type,
                    )

        except KeyboardInterrupt:
            print("\n\n Interruzione utente. Arrivederci!\n")
            break
        except Exception as e:
            print(f"\n Errore: {e}\n")


def generate_reports_from_file(
    scenarios_file: str, num_examples: int = None, use_multi_model: bool = None
):
    """Genera automaticamente report NIOSH da file di scenari

    Args:
        scenarios_file: Path to file containing job descriptions (one per line)
        num_examples: Number of scenarios to process (None = all)
        use_multi_model: If True, generate reports with all models; if False, single model only; if None, ask user
    """

    if not os.path.exists(scenarios_file):
        console.print(f"[danger]ERROR: File {scenarios_file} non trovato[/]")
        return

    # Ask for model selection if not specified
    if use_multi_model is None:
        console.print(
            f"\n[bold cyan]SELECT MODEL MODE FOR BATCH PROCESSING[/bold cyan]"
        )
        console.print(
            "[info]Choose how you want to generate the analysis reports:[/]\n"
        )
        console.print(
            "1. [green]Single Model[/green] - Use only gemma3:12b (Ollama) - Faster processing"
        )
        console.print(
            "2. [yellow]Multi-Model[/yellow] - Generate reports with all available models - Slower but comprehensive\n"
        )

        while True:
            choice = console.input("[info]Select mode (1-2): [/]").strip()
            if choice == "1":
                use_multi_model = False
                break
            elif choice == "2":
                use_multi_model = True
                break
            else:
                console.print("[danger]Invalid choice. Please select 1-2.[/]")

        mode_text = "Multi-Model Mode" if use_multi_model else "Single Model Mode"
        console.print(f"[info]Selected: {mode_text}[/]\n")

    console.print(f"[info]Reading scenarios from {scenarios_file}...[/]")

    try:
        with open(scenarios_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Filtra solo le righe che sembrano descrizioni di task (non vuote, non commenti)
        scenarios = [
            line for line in lines if not line.startswith("#") and len(line) > 10
        ]

        if num_examples:
            scenarios = scenarios[:num_examples]

        console.print(f"[info]Found {len(scenarios)} scenarios to analyze[/]")

        # Initialize components (standard mode)
        generator = NIOSHJobDescriptionGenerator()
        calc = NIOSHCalculator()
        ja_gen = NIOSHJobAnalysisGenerator()
        hz_gen = NIOSHHazardAssessmentGenerator()
        rs_gen = NIOSHRedesignSuggestionsGenerator()

        os.makedirs("report", exist_ok=True)
        success_count = 0
        error_count = 0

        for i, scenario in enumerate(scenarios, 1):
            try:
                console.print(
                    f"\n[info]Processing scenario {i}/{len(scenarios)}: {scenario[:50]}...[/]"
                )

                # Classify task type
                task_type = classify_sentence(scenario)
                console.print(f"[info]Detected task type: [bold]{task_type}[/]")

                if task_type == "repetitive":
                    console.print(
                        "[info]Repetitive task detected → treating as SINGLE-TASK (RNLE rule)"
                    )
                    # Keep task_type as "repetitive" for folder organization

                if use_multi_model:
                    # Multi-model mode: generate reports with all available models
                    console.print(
                        f"[cyan]Multi-Model processing for scenario {i}...[/]"
                    )
                    generate_multi_model_reports(
                        scenario,
                        generator,
                        calc,
                        ja_gen,
                        hz_gen,
                        rs_gen,
                        task_type=task_type,
                    )
                    success_count += 1
                else:
                    # Single model mode: original behavior with comprehensive report
                    console.print(
                        f"[info]Single-Model processing for scenario {i}...[/]"
                    )
                    generate_single_model_report(
                        scenario,
                        generator,
                        calc,
                        ja_gen,
                        hz_gen,
                        rs_gen,
                        auto_save=True,
                        task_type=task_type,
                    )
                    success_count += 1

            except Exception as e:
                console.print(f"[danger]ERROR: Error processing scenario {i}: {e}[/]")
                error_count += 1
                continue

        console.print(f"\n[success]Report generation completed![/]")
        console.print(
            f"[info]Mode used: {'Multi-Model' if use_multi_model else 'Single Model'}[/]"
        )
        console.print(f"[info]Successfully processed scenarios: {success_count}[/]")
        console.print(f"[info]Scenarios with errors: {error_count}[/]")

        if use_multi_model:
            console.print(
                f"[info]Each scenario generated reports with all available models[/]"
            )
            console.print(
                f"[info]Check 'report/' directory for model-specific files (gemma3-12b_*, llama3.2_*, llama3.1-8b_*, gemini-2.5_*)[/]"
            )
        else:
            console.print(
                f"[info]Each scenario generated comprehensive single-model reports[/]"
            )
            console.print(
                f"[info]Check 'report/' directory for niosh_single_*.json/md files[/]"
            )

    except Exception as e:
        console.print(f"[danger]ERROR: Error reading file: {e}[/]")


def batch_process(inputs: List[str]):
    """Processa multiple descrizioni in batch"""

    # TODO: Implementare la pipeline completa anche per il batch se necessario
    generator = NIOSHJobDescriptionGenerator()
    return generator.batch_process(inputs)


if __name__ == "__main__":
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            # Help mode
            print("NIOSH Ergonomic Analysis Tool - Enhanced CLI")
            print("=" * 50)
            print()
            print("USAGE OPTIONS:")
            print("  python cli.py                    # Interactive mode with menu")
            print("  python cli.py --batch <file>    # Direct batch processing")
            print("  python cli.py --help            # Show this help")
            print()
            print("BATCH PROCESSING:")
            print(
                "  python cli.py --batch <scenarios_file> [num_examples] [--multi-model]"
            )
            print("  python cli.py -b <scenarios_file> [num_examples] [--multi-model]")
            print()
            print("OPTIONS:")
            print(
                "  scenarios_file  : Path to text file with job descriptions (one per line)"
            )
            print(
                "  num_examples    : Number of scenarios to process (optional, default = all)"
            )
            print(
                "  --multi-model   : Use multi-model mode (optional, default = ask user)"
            )
            print()
            print("EXAMPLES:")
            print("  python cli.py --batch scenarios.txt")
            print("  python cli.py --batch scenarios.txt 5 --multi-model")
            print("  python cli.py -b scenarios.txt 3")
            print()
            print("INTERACTIVE MODE FEATURES:")
            print("  - Choose between Interactive or Batch processing")
            print("  - Select Single Model or Multi-Model analysis")
            print("  - Real-time ergonomic analysis with immediate feedback")
            print("  - File validation and error handling")
            print()
            print("MODELS AVAILABLE:")
            print("  - gemma3:12b (Ollama)")
            print("  - llama3.2:latest (Ollama)")
            print("  - llama3.1:8b (Ollama)")
            print("  - gemini-2.5 (Google, if available)")
            sys.exit(0)

        elif sys.argv[1] == "--batch" or sys.argv[1] == "-b":
            # Batch mode: python cli.py --batch <scenarios_file> [num_examples] [--multi-model]
            if len(sys.argv) < 3:
                print(
                    "Usage: python cli.py --batch <scenarios_file> [num_examples] [--multi-model]"
                )
                print(
                    "   or: python cli.py -b <scenarios_file> [num_examples] [--multi-model]"
                )
                print("\nUse 'python cli.py --help' for full help")
                sys.exit(1)

            scenarios_file = sys.argv[2]
            num_examples = None
            use_multi_model = None  # Ask user

            # Parse optional arguments
            for arg in sys.argv[3:]:
                if arg.isdigit():
                    num_examples = int(arg)
                elif arg == "--multi-model":
                    use_multi_model = True

            console.print("[bold cyan]NIOSH BATCH PROCESSING MODE[/bold cyan]")
            console.print(f"[info]Scenarios file: {scenarios_file}[/]")
            if num_examples:
                console.print(f"[info]Number of scenarios: {num_examples}[/]")
            if use_multi_model is True:
                console.print(f"[info]Mode: Multi-Model[/]")
            console.print()

            generate_reports_from_file(scenarios_file, num_examples, use_multi_model)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use 'python cli.py --help' for available options")
    else:
        # Interactive mode (default)
        interactive_chat()
