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
from gemini_judge import GeminiJudge, JudgeInput
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
from task_classifier import classify_sentence
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

judge = GeminiJudge()


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
    max_workers: int = 8,
):
    """
    Versione DEFINITIVA con:
    - JD specific/combined per single/repetitive/multi
    - JA/HA/RS specific/combined anche per multi-task
    - JSON strutturato per analisi successive
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

        try:
            # -------------------------------------------------------
            # 1) CLASSIFICAZIONE LLM (single / repetitive / multi)
            # -------------------------------------------------------
            task_type = classify_sentence(text)
            is_multi = task_type == "multi"

            # -------------------------------------------------------
            # 2) JOB DESCRIPTION via PIPELINE NIOSH
            # -------------------------------------------------------
            result = jd_gen.analyze_job(text, task_type)

            jd_text = (
                result.get("description_with_tags")
                or result.get("job_description_with_tags")
                or result.get("description")
            )
            if not jd_text:
                print(f"[ERROR] Job analysis failed for scenario {idx}: missing description")
                return idx, None, None

            jd_text_clean = remove_tags(jd_text)

            # -------------------------------------------------------
            # 2B) JD SPECIFICA + COMBINATA
            # -------------------------------------------------------
            if not is_multi:
                # SINGLE / REPETITIVE → dual version classica
                dual_versions = jd_gen.generate_dual_version(
                    user_input=text,
                    unit_system="imperial",
                    task_type=task_type,
                    parameters=None,
                )

                jd_specific = remove_tags(dual_versions.get("specific_version", ""))
                jd_combined = remove_tags(dual_versions.get("combined_version", ""))

                calc_result = None  # lo calcoliamo dopo

            else:
                # MULTI-TASK → estraggo i task, mappo e calcolo prima il multi-task
                raw_tasks = extract_multi_task_tags(jd_text)
                tasks_data = map_extractor_to_calculator(raw_tasks)
                calc_result = calc.compute_multi_task(
                    tasks_data, job_description=jd_text_clean
                )

                dual_mt = jd_gen.generate_multi_task_dual_job_description(
                    calc_result, task_type
                )

                jd_specific = remove_tags(dual_mt["specific_version"])
                jd_combined = remove_tags(dual_mt["combined_version"])

            # -------------------------------------------------------
            # 3) PARAMETRI, CALCOLI & SEZIONI ANALITICHE
            # -------------------------------------------------------
            params_obj = None
            ja_text = ha_text = rs_text = ""
            ja_specific = ja_combined = ""
            ha_specific = ha_combined = ""
            rs_specific = rs_combined = ""

            if not is_multi:
                # --- SINGLE / REPETITIVE ---
                params_raw = extract_tagged_params(jd_text) or {}
                params_full = intelligent_llm_prevalidator(params_raw, jd_text)
                params_obj = NIOSHParameters(**params_full)

                # Calcolo NIOSH singolo task
                calc_result = calc.compute(params_obj)

                # JOB ANALYSIS (testo “normale” + dual specific/combined)
                ja_text = remove_tags(
                    ja_gen.generate_job_analysis(params_obj, task_type)
                )
                ja_dual = ja_gen.generate_dual_job_analysis(params_obj, task_type)
                ja_specific = remove_tags(ja_dual["ja_specific"])
                ja_combined = remove_tags(ja_dual["ja_combined"])

                # HAZARD ASSESSMENT
                hz_input = HazardAssessmentInput(
                    task_description=jd_text_clean,
                    weight_lbs=params_obj.weight,
                    rwl_origin_lbs=calc_result.rwl_origin_lbs,
                    rwl_dest_lbs=calc_result.rwl_destination_lbs,
                    li_origin=calc_result.li_origin,
                    li_dest=calc_result.li_destination,
                    significant_control=params_obj.significant_control,
                )

                ha_text = remove_tags(
                    hz_gen.generate_hazard_assessment(hz_input, task_type)
                )
                ha_dual = hz_gen.generate_dual_hazard_assessment(hz_input, task_type)
                ha_specific = remove_tags(ha_dual["ha_specific"])
                ha_combined = remove_tags(ha_dual["ha_combined"])

                # REDESIGN SUGGESTIONS
                rs_input = RedesignSuggestionsInput(
                    task_description=jd_text_clean,
                    h_origin=params_obj.horizontal_origin,
                    h_dest=params_obj.horizontal_destination,
                    v_origin=params_obj.vertical_origin,
                    v_dest=params_obj.vertical_destination,
                    a_origin=params_obj.asymmetry_angle,
                    a_dest=params_obj.asymmetry_angle,
                    frequency_lifts_per_min=params_obj.frequency,
                    duration_class=params_obj.duration,
                    coupling=params_obj.coupling,
                    significant_control=params_obj.significant_control,
                    weight_lbs=params_obj.weight,
                    rwl_origin_lbs=calc_result.rwl_origin_lbs,
                    rwl_dest_lbs=calc_result.rwl_destination_lbs,
                    li_origin=calc_result.li_origin,
                    li_dest=calc_result.li_destination,
                    multipliers_origin=calc_result.multipliers_origin,
                    multipliers_destination=calc_result.multipliers_destination,
                    risk_category=calc_result.risk_category,
                    risk_comment=calc_result.risk_comment,
                )

                rs_dual = rs_gen.generate_dual_redesign_suggestions(
                    rs_input, task_type
                )
                rs_specific = remove_tags(rs_dual["rs_specific"])
                rs_combined = remove_tags(rs_dual["rs_combined"])

                report_type = "single_task"

            else:
                # --- MULTI-TASK ---
                # Testi “normali”
                ja_text = remove_tags(
                    ja_gen.generate_multi_task_job_analysis(calc_result, task_type)
                )
                ha_text = remove_tags(
                    hz_gen.generate_multi_task_hazard_assessment(calc_result)
                )
                rs_text = remove_tags(
                    rs_gen.generate_multi_task_redesign(calc_result)
                )

                # Dual multi-task (specific + combined)
                ja_dual = ja_gen.generate_multi_task_dual_job_analysis(
                    calc_result, task_type
                )
                ja_specific = remove_tags(ja_dual["ja_specific"])
                ja_combined = remove_tags(ja_dual["ja_combined"])

                ha_dual = hz_gen.generate_multi_task_dual_hazard_assessment(
                    calc_result, task_type
                )
                ha_specific = remove_tags(ha_dual["ha_specific"])
                ha_combined = remove_tags(ha_dual["ha_combined"])

                rs_dual = rs_gen.generate_multi_task_dual_redesign_suggestions(
                    calc_result, task_type
                )
                rs_specific = remove_tags(rs_dual["rs_specific"])
                rs_combined = remove_tags(rs_dual["rs_combined"])

                report_type = "multi_task"

            # -------------------------------------------------------
            # 3C-1) VALUTAZIONE GEMINI — SPECIFIC VERSIONS
            # -------------------------------------------------------
            eval_jd_specific = judge.evaluate(
                JudgeInput(
                    section_name="job_description_specific",
                    human_text=jd_gen.get_reference_examples(task_type),
                    ai_text=jd_specific,
                    reference_examples=jd_gen.get_reference_examples(task_type),
                )
            )

            eval_ja_specific = judge.evaluate(
                JudgeInput(
                    section_name="job_analysis_specific",
                    human_text=ja_gen.get_reference_examples(task_type),
                    ai_text=ja_specific,
                    reference_examples=ja_gen.get_reference_examples(task_type),
                )
            )

            eval_ha_specific = judge.evaluate(
                JudgeInput(
                    section_name="hazard_assessment_specific",
                    human_text=hz_gen.get_reference_examples(task_type),
                    ai_text=ha_specific,
                    reference_examples=hz_gen.get_reference_examples(task_type),
                )
            )

            eval_rs_specific = judge.evaluate(
                JudgeInput(
                    section_name="redesign_suggestions_specific",
                    human_text=rs_gen.get_reference_examples(task_type),
                    ai_text=rs_specific,
                    reference_examples=rs_gen.get_reference_examples(task_type),
                )
            )

            # -------------------------------------------------------
            # 3C-2) VALUTAZIONE GEMINI — COMBINED VERSIONS
            # -------------------------------------------------------
            eval_jd_combined = judge.evaluate(
                JudgeInput(
                    section_name="job_description_combined",
                    human_text=jd_gen.get_reference_examples(task_type),
                    ai_text=jd_combined,
                    reference_examples=jd_gen.get_reference_examples(task_type),
                )
            )

            eval_ja_combined = judge.evaluate(
                JudgeInput(
                    section_name="job_analysis_combined",
                    human_text=ja_gen.get_reference_examples(task_type),
                    ai_text=ja_combined,
                    reference_examples=ja_gen.get_reference_examples(task_type),
                )
            )

            eval_ha_combined = judge.evaluate(
                JudgeInput(
                    section_name="hazard_assessment_combined",
                    human_text=hz_gen.get_reference_examples(task_type),
                    ai_text=ha_combined,
                    reference_examples=hz_gen.get_reference_examples(task_type),
                )
            )

            eval_rs_combined = judge.evaluate(
                JudgeInput(
                    section_name="redesign_suggestions_combined",
                    human_text=rs_gen.get_reference_examples(task_type),
                    ai_text=rs_combined,
                    reference_examples=rs_gen.get_reference_examples(task_type),
                )
            )

            # -------------------------------------------------------
            # 4) SALVATAGGI FILE
            # -------------------------------------------------------
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            safe_name = "".join(c if c.isalnum() else "_" for c in text[:40])

            type_dir = Path(output_dir) / task_type
            type_dir.mkdir(parents=True, exist_ok=True)

            json_path = type_dir / f"{safe_name}_{idx}.json"
            md_path = type_dir / f"{safe_name}_{idx}.md"
            specific_md_path = type_dir / f"{safe_name}_{idx}_JD_SPECIFIC.md"
            combined_md_path = type_dir / f"{safe_name}_{idx}_JD_COMBINED.md"

            # -------------------------------------------------------
            # 5) JSON STRUCT COMPLETO (analisi + valutazioni)
            # -------------------------------------------------------
            json_data = {
                "input": text,
                "timestamp": timestamp,
                "task_type": task_type,
                "jd_specific": jd_specific,
                "jd_combined": jd_combined,
                "jd_final_clean": jd_text_clean,
                "analysis": {
                    "job_analysis": ja_text,
                    "hazard_assessment": ha_text,
                    "redesign_suggestions": rs_text,
                    "job_analysis_specific": ja_specific,
                    "job_analysis_combined": ja_combined,
                    "hazard_assessment_specific": ha_specific,
                    "hazard_assessment_combined": ha_combined,
                    "redesign_suggestions_specific": rs_specific,
                    "redesign_suggestions_combined": rs_combined,
                },
                "evaluation": {
                    "jd_specific_eval": eval_jd_specific,
                    "jd_combined_eval": eval_jd_combined,
                    "ja_specific_eval": eval_ja_specific,
                    "ja_combined_eval": eval_ja_combined,
                    "ha_specific_eval": eval_ha_specific,
                    "ha_combined_eval": eval_ha_combined,
                    "rs_specific_eval": eval_rs_specific,
                    "rs_combined_eval": eval_rs_combined,
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

            print(f"SALVO → {json_path}")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            # -------------------------------------------------------
            # 6) MARKDOWN – SPECIFICA / COMBINATA
            # -------------------------------------------------------
            jd_spec_md = f"""# Job Description – SPECIFIC VERSION

    **Scenario:** {text}  
    **Generated:** {timestamp}  
    **Model:** {model}  

    ## Job Description 
    {jd_specific}

    ## Job Analysis 
    {ja_specific}

    ## Hazard Assessment 
    {ha_specific}

    ## Redesign Suggestions
    {rs_specific}
    """
            with open(specific_md_path, "w", encoding="utf-8", buffering=65536) as mf:
                mf.write(jd_spec_md)

            jd_comb_md = f"""# Job Description – COMBINED VERSION

    **Scenario:** {text}  
    **Generated:** {timestamp}  
    **Model:** {model}  

    ## Job Description 
    {jd_combined}

    ## Job Analysis 
    {ja_combined}

    ## Hazard Assessment 
    {ha_combined}

    ## Redesign Suggestions
    {rs_combined}
    """
            with open(combined_md_path, "w", encoding="utf-8", buffering=65536) as mf:
                mf.write(jd_comb_md)

            # markdown principale (se vuoi mantenerlo compatibile)
            with open(md_path, "w", encoding="utf-8", buffering=65536) as mf:
                mf.write(jd_comb_md)

            return idx, json_path, md_path

        except Exception as e:
            print(f"[ERROR] Failed to process scenario {idx}: {e}")
            import traceback

            traceback.print_exc()
            return idx, None, None


    # ================================================
    #   PARALLEL EXECUTION
    # ================================================
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, text in enumerate(scenarios, start=1):
            futures.append(executor.submit(process_single_scenario, idx, text))

        for future in as_completed(futures):
            result = future.result()
            if result[1] is not None:
                idx, json_path, md_path = result
                print(f"✓ Scenario {idx} completed → {json_path}")
            else:
                print(f"✗ Scenario {result[0]} failed")

    print("\n=== Parallel Batch Processing Completed ===")



# Note: retry_logic import skipped to avoid import issues
def detect_multi_task(user_text: str) -> bool:
    """
    Classifica la frase come MULTI-TASK se contiene indicatori semantici,
    lessicali o strutturali di compiti multipli.
    Restituisce False solo quando è chiaramente single-task.
    """

    txt = user_text.lower().strip()

    # ============================================================
    # 1) PATTERN FORTI (quasi sempre multi-task)
    # ============================================================
    strong_multi = [
        r"\btier\b", r"\btiers\b",
        r"\blevels\b",
        r"\blayers\b",
        r"\bmultiple tasks?\b",
        r"\bdifferent tasks?\b",
        r"\bvarious tasks?\b",
        r"\bmulti[- ]?task\b",
        r"\bseveral tasks?\b",
        r"\bdistinct tasks?\b",
        r"\bdifferent heights\b",
        r"\bdifferent shelves\b",
        r"\bmultiple shelves\b",
        r"\btask\s*1\b", r"\btask\s*2\b", r"\btask\s*3\b",
        r"\bfirst task\b", r"\bsecond task\b", r"\bthird task\b",
    ]

    for pat in strong_multi:
        if re.search(pat, txt):
            return True

    # ============================================================
    # 2) PATTERN MEDIO-FORTI (strutture sequenziali fisiche)
    # ============================================================
    medium_multi = [
        r"\b(top|middle|bottom)\s+(shelf|tier|row)\b",
        r"\bfrom\b.+\bto\b.+\bto\b",          # da X a Y a Z
        r"\bthree shelves\b",
        r"\bfive tiers\b",
        r"\bstacked\b",
        r"\bstack of\b",
        r"\bunload(?:ing)? multiple\b",
    ]

    for pat in medium_multi:
        if re.search(pat, txt):
            return True

    # ============================================================
    # 3) RILEVATORE DI SEQUENZE (fondamentale!)
    # ============================================================
    sequence_markers = [
        " then ",
        " next ",
        " after that ",
        " followed by ",
        " and finally ",
        " first ",
        " secondly ",
        " third ",
    ]

    # Le sequenze sono il segnale più naturale di multi-task umano
    occurrences = sum(1 for m in sequence_markers if m in txt)
    if occurrences >= 1:
        return True

    # ============================================================
    # 4) TRIGRAMMI DI AZIONI → 3+ verbi dinamici diversi
    # ============================================================
    # Riconosce “solleva… sposta… deposita…”
    verbs = re.findall(
        r"\b(lift|move|carry|place|set|pick up|put|transfer|raise|lower|load|unload)\b",
        txt
    )

    # Se troviamo 3+ verbi d'azione diversi → quasi certamente multi-task
    if len(set(verbs)) >= 3:
        return True

    # ============================================================
    # 5) Pattern numerici → molte altezze diverse = multi-task implicito
    # ============================================================
    # Tre o più valori verticali → probabile sequenza multipla
    vertical_values = re.findall(r"\b(\d+)\s*(?:in|cm)\b", txt)
    if len(vertical_values) >= 3:
        return True

    # Molte destinazioni/origini menzionate
    if len(re.findall(r"\b(origin|destination)\b", txt)) >= 3:
        return True

    # Tre o più oggetti distinti → indica più fasi
    if len(re.findall(r"\bbox\b|\bcontainer\b|\broll\b|\bcan\b", txt)) >= 3:
        return True

    # ============================================================
    # 6) FALLBACK: se la frase è lunga e con molte azioni → multi
    # ============================================================
    if len(txt.split()) >= 25 and len(set(verbs)) >= 2:
        return True

    # ============================================================
    # DEFAULT: single-task
    # ============================================================
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
    default_file_path = "scenari_niosh_fps3.txt"
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
                        f"[green]Successfully analyzed job with {generator.model}[/]"
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
            console.print(f"Job analysis generated with {ja_gen.model}")

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
                f"Hazard assessment generated with {hz_gen.model}"
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
                f"Redesign suggestions generated with {rs_gen.model}"
            )

        except Exception as e:
            console.print(f"Failed to generate report sections: {e}")
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


def generate_multi_task_report_single_model(
    user_input, generator, calc, ja_gen, hz_gen, rs_gen, task_type="multi"
):
    """Generate multi-task report using single model pipeline."""

    console.print("[cyan]Detected MULTI-TASK → switching to multi-task pipeline[/]")

    # Qui usi la pipeline multi-task (la stessa del batch)
    job_desc_result = generator.analyze_job(user_input)
    jd_text = job_desc_result["description"]
    tasks_data_raw = extract_multi_task_tags(jd_text)
    tasks_data = map_extractor_to_calculator(tasks_data_raw)
    calc_result = calc.compute_multi_task(tasks_data, job_description=jd_text)

    ja_text = remove_tags(ja_gen.generate_multi_task_job_analysis(calc_result))
    ha_text = remove_tags(hz_gen.generate_multi_task_hazard_assessment(calc_result))
    rs_text = remove_tags(rs_gen.generate_multi_task_redesign(calc_result))
    # --- Save Report Logic ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports") / task_type
    report_dir.mkdir(parents=True, exist_ok=True)

    base_filename = f"niosh_multitask_{timestamp}"

    # 1. Construct Full Report (Markdown)
    full_report_md = f"# NIOSH Multi-Task Analysis Report\n\n"
    full_report_md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    full_report_md += f"**Job Description:** {jd_text}\n\n"
    full_report_md += f"## Job Analysis\n\n{ja_text}\n\n"
    full_report_md += f"## Hazard Assessment\n\n{ha_text}\n\n"
    full_report_md += f"## Redesign Suggestions\n\n{rs_text}\n\n"

    # 2. Clean Tags for Final Output (preserving newlines)
    # Remove [TAG:value] patterns
    clean_report_md = re.sub(r"\[[a-zA-Z0-9]+:[^\]]+\]", "", full_report_md)
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
    with open(json_path, "w", encoding="utf-8", buffering=65536) as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # 5. User Feedback
    console.print(f"\n[success]Multi-task analysis completed![/]")
    console.print(f"[info]Report saved to: {md_path}[/]")
    console.print(f"[info]Data saved to: {json_path}[/]")

    # Show preview (clean)
    console.print("\n[bold]Report Preview:[/]\n")
    console.print(clean_report_md)


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
    """
    Versione CORRETTA con:
    - Gestione task_type consistente
    - Fix valori di default
    - Migliore error handling
    """

    try:
        
        if task_type == "repetitive":
            console.print("[info]Repetitive task → using SINGLE-TASK RNLE[/]")
            task_type = "single"  # Normalizza per evitare confusione

        # Step 1: Analisi job
        result = generator.analyze_job(user_input)
        if not result:
            console.print("[danger]ERROR: Job analysis failed[/]")
            return None

        mode = result.get("mode", "single-task")
        if mode != "single-task":
            console.print("[danger]ERROR: Multi-task detected in single-task mode[/]")
            return "MULTI_DETECTED"

        params_raw = result.get("parameters")
        if not params_raw:
            console.print("[danger]ERROR: Parameters not extracted[/]")
            return None

        description = result.get("description", "")
        description_clean = remove_tags(description)

      
        params_filled = intelligent_llm_prevalidator(params_raw, description)

        # Default values NIOSH-compliant
        default_values = {
            "weight": 25.0,
            "horizontal_origin": 25.0,  # Standard 10" (25cm)
            "horizontal_destination": 25.0,
            "vertical_origin": 75.0,  # Standard 30" (75cm)
            "vertical_destination": 75.0,
            "asymmetry_angle": 0.0,
            "frequency": 0.2,  # 1 lift per 5 min
            "duration": "<1h",
            "coupling": "fair",
            "significant_control": False,
            "gender": "M",
            "age": 25,
            "one_limb_lifting": False,
            "two_operators_lifting": False,
        }

        # Applica defaults solo se valori mancanti
        for key, default_val in default_values.items():
            if key not in params_filled or params_filled[key] is None:
                params_filled[key] = default_val

        # Validazione parametri
        ok, err = validate_params(params_filled)
        if not ok:
            console.print(f"[yellow]Validation warning: {err}[/]")
            console.print("[info]Continuing with corrected parameters...[/]")

        # Costruzione parametri NIOSH
        try:
            p = NIOSHParameters(**params_filled)
        except Exception as e:
            console.print(f"[danger]Error creating parameters: {e}[/]")
            console.print("[yellow]Using safe defaults...[/]")
            p = NIOSHParameters(**default_values)

        # Step 2: Calcoli RNLE
        calc_result = calc.compute(p)

        # Step 3: Genera sezioni report
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
        redesign_suggestions = remove_tags(rs_gen.generate_redesign_suggestions(rs_input))

        # Step 4: Crea report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

        # Step 5: Salva files
        if auto_save or get_save_confirmation():
            report_dir = Path("report") / task_type
            report_dir.mkdir(parents=True, exist_ok=True)

            safe_name = "".join([c if c.isalnum() else "_" for c in user_input[:30]]).strip("_")
            timestamp_short = int(time.time())
            base_filename = f"niosh_single_{safe_name}_{timestamp_short}"

            
            json_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "analysis_type": task_type,
                    "input_phrase": user_input,
                },
                "description": description_clean,
                "parameters": params_filled,
                "niosh_calculation": {
                    "rwl_origin_lbs": calc_result.rwl_origin_lbs,
                    "rwl_destination_lbs": calc_result.rwl_destination_lbs,
                    "li_origin": calc_result.li_origin,
                    "li_destination": calc_result.li_destination,
                    "multipliers_origin": calc_result.multipliers_origin,
                    "multipliers_destination": calc_result.multipliers_destination,
                    "risk_category": calc_result.risk_category,
                    "risk_comment": calc_result.risk_comment,
                },
                "sections": {
                    "job_analysis": job_analysis,
                    "hazard_assessment": hazard_assessment,
                    "redesign_suggestions": redesign_suggestions,
                },
            }

            
            json_filename = report_dir / f"{base_filename}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            # Salva Markdown
            md_filename = report_dir / f"{base_filename}.md"
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(professional_report)

            console.print(f"[success]Report saved: {json_filename} and {md_filename}[/]")

        # Display summary
        console.print("\n[bold green]✓ Analysis Complete![/]")
        console.print(f"[info]LI Origin: {calc_result.li_origin:.2f}[/]")
        if calc_result.li_destination is not None:
            console.print(f"[info]LI Destination: {calc_result.li_destination:.2f}[/]")
        console.print(f"[info]Risk Category: {calc_result.risk_category}[/]")
        console.print(f"[info]RWL Origin: {calc_result.rwl_origin_lbs:.1f} lbs[/]")
        if calc_result.rwl_destination_lbs is not None:
            console.print(f"[info]RWL Destination: {calc_result.rwl_destination_lbs:.1f} lbs[/]")

        return "SUCCESS"

    except Exception as e:
        console.print(f"[danger]Error in analysis: {e}[/]")
        import traceback
        traceback.print_exc()
        return None



def interactive_chat():
    """
    Enhanced interactive chat with Rich formatting.
    VERSIONE CORRETTA con:
    - Rimozione duplicazione generators
    - Gestione task_type normalizzata
    - Logica multi-model ottimizzata
    - Error handling robusto
    """

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
            import traceback
            traceback.print_exc()

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

    # ===================================================================
    # INTERACTIVE MODE - Inizializzazione
    # ===================================================================
    
    # Interactive mode continues with model selection
    use_multi_model = get_model_selection()
    mode_text = "Multi-Model Mode" if use_multi_model else "Single Model Mode"
    console.print(f"\n[info]Selected: {mode_text}[/]\n")

    # Health check Ollama
    console.print("[info]Checking Ollama service health...[/]")
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            console.print("[success]✓ Ollama service is healthy![/]")
        else:
            console.print(
                f"[yellow]⚠ Warning: Ollama responded with status {response.status_code}[/]"
            )
            console.print("[yellow]Continuing anyway...[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Could not connect to Ollama: {e}[/]")
        console.print("[yellow]Continuing anyway...[/]")

    if use_multi_model:
        console.print("[info]Google Gemini status: Will be checked during execution[/]")

    
    console.print("[info]Initializing generators...[/]")
    try:
        # Single shared instance per ogni tipo di generator
        generator = NIOSHJobDescriptionGenerator()
        calc = NIOSHCalculator()
        ja_gen = NIOSHJobAnalysisGenerator()
        hz_gen = NIOSHHazardAssessmentGenerator()
        rs_gen = NIOSHRedesignSuggestionsGenerator()
        console.print("[success]✓ All generators ready![/]")
    except Exception as e:
        console.print(f"[danger]✗ Error initializing generators: {e}[/]")
        import traceback
        traceback.print_exc()
        return

    console.print("\n[bold green]════════════════════════════════════════[/]")
    console.print("[bold green]   SYSTEM READY FOR INPUT[/]")
    console.print("[bold green]════════════════════════════════════════[/]\n")

    # ===================================================================
    # MAIN INTERACTION LOOP
    # ===================================================================
    
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
                    "[info]Repetitive task → using SINGLE-TASK RNLE (per NIOSH guidelines)"
                )
                # Normalizza per la pipeline, ma mantieni l'originale per logging
                original_task_type = task_type
                task_type = "single" 
                console.print(f"[info]Task type normalized: {original_task_type} → {task_type}[/]")

            is_multi = (task_type == "multi")

            # ===================================================================
            # MULTI-MODEL MODE
            # ===================================================================
            
            if use_multi_model:
                console.print(f"\n[cyan]━━━ MULTI-MODEL PROCESSING ━━━[/cyan]")
                console.print(f"[info]Task type: {task_type}[/]")
                console.print(f"[info]Will generate reports with ALL available models[/]\n")
                
                
                generate_multi_model_reports(
                    user_input,
                    generator,
                    calc,
                    ja_gen,
                    hz_gen,
                    rs_gen,
                    task_type=task_type,
                )

            # ===================================================================
            # SINGLE-MODEL MODE
            # ===================================================================
            
            else:
                console.print(f"\n[green]━━━ SINGLE-MODEL PROCESSING ━━━[/green]")
                console.print(f"[info]Task type: {task_type}[/]")
                console.print(f"[info]Using default model (gemma3:12b)[/]\n")
                
                if is_multi:
             
                    console.print("[cyan]Using MULTI-TASK pipeline[/]")
                    generate_multi_task_report_single_model(
                        user_input,
                        generator,
                        calc,
                        ja_gen,
                        hz_gen,
                        rs_gen,
                        task_type=task_type,
                    )
                else:
                    # Single-task pipeline con fallback
                    console.print("[cyan]Using SINGLE-TASK pipeline[/]")
                    result = generate_single_model_report(
                        user_input,
                        generator,
                        calc,
                        ja_gen,
                        hz_gen,
                        rs_gen,
                        task_type=task_type,
                    )

        
                    if result == "MULTI_DETECTED":
                        console.print(
                            "[yellow]⚠ Multi-task detected during analysis![/]"
                        )
                        console.print("[info]Switching to MULTI-TASK pipeline...[/]")
                        generate_multi_task_report_single_model(
                            user_input,
                            generator,
                            calc,
                            ja_gen,
                            hz_gen,
                            rs_gen,
                            task_type="multi",
                        )
                    elif result is None:
                        console.print("[yellow]⚠ Analysis failed, please try again[/]")

            # Separator per nuova iterazione
            console.print("\n[dim]" + "─" * 60 + "[/dim]\n")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]⚠ Interruzione utente. Arrivederci![/]\n")
            show_goodbye_message()
            break
        except Exception as e:
            console.print(f"\n[danger]✗ Errore imprevisto: {e}[/]\n")
            import traceback
            console.print("[dim]Stack trace:[/dim]")
            traceback.print_exc()
            console.print("\n[info]Il sistema è ancora attivo. Puoi continuare.[/]\n")


def generate_reports_from_file(
    scenarios_file: str, num_examples: int = None, use_multi_model: bool = None
):
    """
    Genera automaticamente report NIOSH da file di scenari.
    
    VERSIONE CORRETTA con:
    - Normalizzazione task_type corretta
    - Gestione multi-task completa
    - Success tracking accurato
    - Error handling robusto
    - Statistiche dettagliate

    Args:
        scenarios_file: Path to file containing job descriptions (one per line)
        num_examples: Number of scenarios to process (None = all)
        use_multi_model: If True, generate reports with all models; 
                        If False, single model only; 
                        If None, ask user
    """

    # ===================================================================
    # VALIDAZIONE FILE
    # ===================================================================
    
    if not os.path.exists(scenarios_file):
        console.print(f"[danger]✗ ERROR: File '{scenarios_file}' not found[/]")
        return

    # ===================================================================
    # MODEL SELECTION
    # ===================================================================
    
    if use_multi_model is None:
        console.print(f"\n[bold cyan]SELECT MODEL MODE FOR BATCH PROCESSING[/bold cyan]")
        console.print("[info]Choose how you want to generate the analysis reports:[/]\n")
        console.print(
            "1. [green]Single Model[/green] - Use only gemma3:12b (Ollama)"
        )
        console.print("   Faster processing with single model output\n")
        console.print(
            "2. [yellow]Multi-Model[/yellow] - Generate reports with all available models"
        )
        console.print("   Slower but comprehensive (gemma3, llama3.2, llama3.1, gemini)\n")

        while True:
            choice = console.input("[info]Select mode (1-2): [/]").strip()
            if choice == "1":
                use_multi_model = False
                break
            elif choice == "2":
                use_multi_model = True
                break
            else:
                console.print("[danger]Invalid choice. Please select 1 or 2.[/]")

        mode_text = "Multi-Model Mode" if use_multi_model else "Single Model Mode"
        console.print(f"[info]✓ Selected: {mode_text}[/]\n")

    # ===================================================================
    # LETTURA FILE E PREPROCESSING
    # ===================================================================
    
    console.print(f"[info]Reading scenarios from '{scenarios_file}'...[/]")

    try:
        with open(scenarios_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Filtra righe valide (non commenti, lunghezza minima)
        scenarios = [
            line for line in lines 
            if not line.startswith("#") and len(line) > 10
        ]

        if not scenarios:
            console.print("[yellow]⚠ Warning: No valid scenarios found in file[/]")
            return

        if num_examples:
            scenarios = scenarios[:num_examples]
            console.print(
                f"[info]✓ Found {len(lines)} scenarios, processing first {num_examples}[/]"
            )
        else:
            console.print(f"[info]✓ Found {len(scenarios)} scenarios to analyze[/]")

    except Exception as e:
        console.print(f"[danger]✗ ERROR: Error reading file: {e}[/]")
        import traceback
        traceback.print_exc()
        return

    # ===================================================================
    # INIZIALIZZAZIONE COMPONENTS
    # ===================================================================
    
    console.print("\n[info]Initializing NIOSH analysis components...[/]")
    try:
        generator = NIOSHJobDescriptionGenerator()
        calc = NIOSHCalculator()
        ja_gen = NIOSHJobAnalysisGenerator()
        hz_gen = NIOSHHazardAssessmentGenerator()
        rs_gen = NIOSHRedesignSuggestionsGenerator()
        console.print("[success]✓ All components initialized[/]")
    except Exception as e:
        console.print(f"[danger]✗ ERROR: Failed to initialize components: {e}[/]")
        return

    os.makedirs("report", exist_ok=True)

    # ===================================================================
    # STATISTICHE PROCESSING
    # ===================================================================
    
    stats = {
        "success": 0,
        "failed": 0,
        "single_task": 0,
        "multi_task": 0,
        "repetitive_task": 0,
    }

    console.print("\n[bold green]═══════════════════════════════════════[/]")
    console.print("[bold green]   STARTING BATCH PROCESSING[/]")
    console.print("[bold green]═══════════════════════════════════════[/]\n")

    # ===================================================================
    # MAIN PROCESSING LOOP
    # ===================================================================
    
    for i, scenario in enumerate(scenarios, 1):
        try:
            console.print(f"\n[cyan]━━━ Scenario {i}/{len(scenarios)} ━━━[/cyan]")
            console.print(f"[info]Text: {scenario[:70]}{'...' if len(scenario) > 70 else ''}[/]")

            #  Classificazione task type
            task_type = classify_sentence(scenario)
            console.print(f"[info]Detected type: [bold]{task_type}[/]")

            # Normalizzazione di task_type
            original_task_type = task_type
            if task_type == "repetitive":
                console.print(
                    "[info]Repetitive task → using SINGLE-TASK RNLE (per NIOSH guidelines)"
                )
                task_type = "single" 
                stats["repetitive_task"] += 1
            
            # Traccia statistiche per tipo
            if task_type == "multi":
                stats["multi_task"] += 1
            elif task_type == "single":
                stats["single_task"] += 1

            # ===================================================================
            # PROCESSING: MULTI-MODEL MODE
            # ===================================================================
            
            if use_multi_model:
                console.print(f"[cyan]→ Multi-Model processing...[/]")
                
                try:
                    generate_multi_model_reports(
                        scenario,
                        generator,
                        calc,
                        ja_gen,
                        hz_gen,
                        rs_gen,
                        task_type=task_type,
                    )
                    stats["success"] += 1
                    console.print(f"[green]✓ Scenario {i} completed (multi-model)[/]")
                    
                except Exception as e:
                    stats["failed"] += 1
                    console.print(f"[red]✗ Scenario {i} failed: {e}[/]")
                    import traceback
                    console.print("[dim]" + traceback.format_exc() + "[/dim]")

            # ===================================================================
            # PROCESSING: SINGLE-MODEL MODE
            # ===================================================================
            
            else:
                console.print(f"[cyan]→ Single-Model processing...[/]")
                
                try:
                    #  multi-task
                    if task_type == "multi":
                        # Multi-task pipeline
                        console.print("[info]Using MULTI-TASK pipeline[/]")
                        generate_multi_task_report_single_model(
                            scenario,
                            generator,
                            calc,
                            ja_gen,
                            hz_gen,
                            rs_gen,
                            task_type=task_type,
                        )
                    else:
                        # Single-task pipeline
                        console.print("[info]Using SINGLE-TASK pipeline[/]")
                        result = generate_single_model_report(
                            scenario,
                            generator,
                            calc,
                            ja_gen,
                            hz_gen,
                            rs_gen,
                            auto_save=True,
                            task_type=task_type,
                        )
                        
                        #  Gestione fallback MULTI_DETECTED
                        if result == "MULTI_DETECTED":
                            console.print("[yellow]⚠ Multi-task detected during analysis[/]")
                            console.print("[info]Switching to MULTI-TASK pipeline...[/]")
                            generate_multi_task_report_single_model(
                                scenario,
                                generator,
                                calc,
                                ja_gen,
                                hz_gen,
                                rs_gen,
                                task_type="multi",
                            )
                            stats["multi_task"] += 1
                            stats["single_task"] -= 1  # Correggi conteggio
                        elif result is None:
                            raise Exception("Analysis returned None")
                    
                    # Success count SOLO se completato con successo
                    stats["success"] += 1
                    console.print(f"[green]✓ Scenario {i} completed (single-model)[/]")
                    
                except Exception as e:
                    stats["failed"] += 1
                    console.print(f"[red]✗ Scenario {i} failed: {e}[/]")
                    import traceback
                    console.print("[dim]" + traceback.format_exc() + "[/dim]")

        except Exception as e:
            # Catch-all per errori imprevisti
            stats["failed"] += 1
            console.print(f"[red]✗ Scenario {i} - Unexpected error: {e}[/]")
            import traceback
            console.print("[dim]" + traceback.format_exc() + "[/dim]")
            continue

    # ===================================================================
    # SUMMARY FINALE
    # ===================================================================
    
    console.print("\n[bold green]═══════════════════════════════════════[/]")
    console.print("[bold green]   BATCH PROCESSING COMPLETE[/]")
    console.print("[bold green]═══════════════════════════════════════[/]\n")

    # Statistiche generali
    console.print(f"[info]Mode used: {'Multi-Model' if use_multi_model else 'Single Model'}[/]")
    console.print(f"[success]✓ Successfully processed: {stats['success']}/{len(scenarios)}[/]")
    
    if stats["failed"] > 0:
        console.print(f"[yellow]✗ Failed scenarios: {stats['failed']}/{len(scenarios)}[/]")
    
    # Statistiche per tipo
    console.print(f"\n[bold]Task Type Breakdown:[/]")
    console.print(f"  • Single-task: {stats['single_task']}")
    console.print(f"  • Multi-task: {stats['multi_task']}")
    console.print(f"  • Repetitive (→single): {stats['repetitive_task']}")

    # Info sui file generati
    console.print(f"\n[bold]Output Location:[/]")
    if use_multi_model:
        console.print(f"  • Each scenario generated reports with all available models")
        console.print(f"  • Check 'report/' directory for model-specific files:")
        console.print(f"    - gemma3-12b_report_*.json/md")
        console.print(f"    - llama3.2_report_*.json/md")
        console.print(f"    - llama3.1-8b_report_*.json/md")
        console.print(f"    - gemini-2.5_report_*.json/md (if available)")
    else:
        console.print(f"  • Each scenario generated comprehensive single-model reports")
        console.print(f"  • Check 'report/' directory for:")
        console.print(f"    - niosh_single_*.json/md (single-task)")
        console.print(f"    - niosh_multitask_*.json/md (multi-task)")

    # Success rate
    success_rate = (stats["success"] / len(scenarios) * 100) if len(scenarios) > 0 else 0
    console.print(f"\n[bold]Success Rate: {success_rate:.1f}%[/]")
    
    if success_rate == 100:
        console.print("[green]🎉 All scenarios processed successfully![/]")
    elif success_rate >= 80:
        console.print("[yellow]⚠ Most scenarios processed, check failed ones[/]")
    else:
        console.print("[red]⚠ Many failures detected, review errors above[/]")



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
