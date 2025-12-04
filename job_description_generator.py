import json
from typing import Dict, List, Optional
import ollama
from jd_ja_consistency import check_jd_ja_consistency
from tag_extractor import extract_tagged_params
from multi_task_extractor import extract_multi_task_tags  # solo se multi-task
from niosh_calculator import NIOSHCalculator
from niosh_parameters import NIOSHParameters
from model_router import call_llm_with_system_prompt, is_gemini_model
import validators

import re

REQUIRED_TAGS = ["W", "H0", "H1", "V0", "V1", "A", "F", "DUR", "COUP", "CTRL"]


def fix_or_insert_missing_tags(text: str, params) -> str:
    """
    Garantisce che tutti i tag obbligatori siano presenti, corretti e unici.
    Se mancano, li inserisce usando i parametri estratti.
    Se sono formattati male, li corregge.
    """
    fixed = text

    def ensure(tag, val):
        nonlocal fixed
        # rimpiazza versione errata
        fixed = re.sub(rf"\[{tag}\s*:\s*[^\]]*\]", f"[{tag}:{val}]", fixed)
        # inserisce se manca
        if f"[{tag}:" not in fixed:
            fixed += f" [{tag}:{val}]"

    ensure("W", f"{params.weight:.1f}")
    ensure("H0", f"{params.horizontal_origin:.1f}")
    ensure("H1", f"{params.horizontal_destination:.1f}")
    ensure("V0", f"{params.vertical_origin:.1f}")
    ensure("V1", f"{params.vertical_destination:.1f}")
    ensure("A", f"{params.asymmetry_angle:.1f}")
    ensure("F", f"{params.frequency:.1f}")
    ensure("DUR", params.duration)
    ensure("COUP", params.coupling)
    ensure("CTRL", "true" if params.significant_control else "false")

    # elimina spazi indesiderati
    fixed = re.sub(r"\s+", " ", fixed).strip()

    return fixed


def detect_multi_task(user_text: str) -> bool:
    """
    Riconosce se l'input descrive un job multi-task secondo la logica NIOSH.
    """
    t = user_text.lower()

    strong = [
        r"\btier\b",
        r"\btiers\b",
        r"\blevels\b",
        r"\blayers\b",
        r"\bmulti[- ]?task\b",
        r"\bmultiple tasks\b",
        r"\bdifferent tasks\b",
        r"\bvarious tasks\b",
        r"\bseveral tasks\b",
        r"\bdistinct tasks\b",
    ]
    for p in strong:
        if re.search(p, t):
            return True

    medium = [
        r"\bfrom .* to .* to\b",  # es. from cart to shelf 1 to shelf 2
        r"\bthree shelves\b",
        r"\bfive tiers\b",
        r"\bmultiple shelves\b",
        r"\btop (shelf|tier)\b",
        r"\bbottom (shelf|tier)\b",
    ]
    for p in medium:
        if re.search(p, t):
            return True

    # euristica numerica: molte altezze diverse → multi-task probabile
    if len(re.findall(r"\b(\d+)\s*(?:in|cm)\b", t)) >= 3:
        return True

    return False


class NIOSHJobDescriptionGenerator:
    """Generatore di job descriptions NIOSH - sistema ibrido (Ollama + Structured Outputs)"""

    def __init__(self, model: str = "gemma3:12b"):
        self.model = model

    def _detect_units(self, user_input: str) -> str:
        """Rileva se l'utente preferisce metrico (kg/cm) o imperiale (lbs/in)."""
        text = user_input.lower()
        metric_keywords = [
            "kg",
            "kilogram",
            "chilogrammo",
            "cm",
            "centimeter",
            "centimetro",
            "metrico",
            "metric",
        ]
        if any(k in text for k in metric_keywords):
            return "metric"
        return "imperial"

    def _create_system_prompt(self) -> str:
        """Crea il system prompt per la generazione di job descriptions."""
        return """You are an expert in NIOSH lifting equation and occupational ergonomics.
Your task is to generate detailed, technical job descriptions for manual lifting tasks.
Follow the style of the NIOSH Applications Manual and include all required tags.
Use consistent measurements and avoid contradictions.

If task_type="repetitive", describe clearly that this is a repetitive lifting task, 
highlighting frequency, sustained duration and cumulative physical demand. 
Do not change the numerical values.

DURATION TAG RULES (CRITICAL – NIOSH RNLE COMPLIANT)

You MUST treat the duration category as a CLOSED SET.

The ONLY valid values for the duration tag [DUR:…] are:

- [DUR:<1h]   → for tasks lasting less than 1 hour
- [DUR:1-2h]  → for tasks lasting between 1 and 2 hours
- [DUR:2-8h]  → for tasks lasting between 2 and 8 hours (typical work periods or partial shifts)
- [DUR:>8h]   → for tasks performed more than 8 hours per day

You MUST NOT:
- invent new duration labels (e.g. [DUR:1-3h], [DUR:2-3h], [DUR:3-5h], [DUR:1h], [DUR:3h])
- write free-form text inside the tag (e.g. [DUR:between one and three hours])
- change units (no minutes, no “short/medium/long”)

Natural-language text may describe the duration in words
(e.g., “between one and two hours”, “for most of the shift”, “for a full 8-hour shift”),
but the TAG must ALWAYS be one of the four canonical NIOSH categories above.

When in doubt, you MUST discretize the narrative duration into the closest valid NIOSH category, and then:
- keep the narrative free text, BUT
- use ONLY one of: <1h, 1-2h, 2-8h, >8h inside [DUR:…].

Never produce any [DUR:…] tag that is not exactly one of:
[DUR:<1h], [DUR:1-2h], [DUR:2-8h], [DUR:>8h].

"""

    def generate_job_description(
    self,
    user_input: str,
    unit_system: str = "imperial",
    parameters: Optional[NIOSHParameters] = None,
    task_type: str = None,
) -> str:

        # ==========================================================
        # 1) SELEZIONE ESEMPI (FEW-SHOT) E CONTESTO IN BASE AL TIPO
        # ==========================================================

        # --- SINGLE-TASK PROMPT ---
        examples_single = """
    EXAMPLE — SINGLE-TASK
    USER: "The worker lifts boxes from a pallet and places them onto a table."

    ASSISTANT: "The worker lifts a box weighing approximately 32 pounds [W:32] from a pallet on the floor.
    At the origin, the hands are positioned at approximately 10 inches [V0:10] with a horizontal reach
    of about 18 inches [H0:18]. The box is placed onto a table at roughly 34 inches [V1:34] and 
    14 inches [H1:14]. The lift involves mild torso rotation of about 20 degrees [A:20]. The task is 
    performed at about 2 lifts/min [F:2] with duration <1h [DUR:<1h]. Coupling is fair [COUP:fair] 
    and significant control is not required [CTRL:false]."
    """

        # --- REPETITIVE LIFTING PROMPT ---
        examples_repetitive = """
    EXAMPLE — REPETITIVE SINGLE-TASK
    USER: "Lifting containers repeatedly from a low shelf to a higher shelf for inspection."

    ASSISTANT: "The worker repeatedly lifts compact containers weighing 26 pounds [W:26] from a low shelf.
    The hands start at about 22 inches [V0:22] with 10 inches horizontal reach [H0:10]. Containers
    are placed at a higher shelf at 59 inches [V1:59] and 20 inches reach [H1:20]. No trunk rotation
    occurs [A:0]. This repetitive lift occurs at 3 lifts/min [F:3] for 45 minutes [DUR:<1h]. Coupling
    is fair [COUP:fair], significant control is required [CTRL:true]."
    """

        # --- MULTI-TASK PROMPT ---
        examples_multi = """
    EXAMPLE — MULTI-TASK
    USER: "The worker removes boxes from a lower pallet tier, then lifts small components to chest height,
    and finally loads heavier containers onto a high shelf."

    ASSISTANT:
    "The job consists of several distinct lifting tasks performed at different heights and load weights.

    In the first task, the employee removes boxes weighing 25 pounds from the lower pallet tier. The hands
    start near floor level at approximately 12 inches vertically and 16 inches horizontally from the body
    and the boxes are lifted to chest height at about 36 inches vertically and 20 inches horizontally.
    The lift involves about 35 degrees of torso rotation and is performed at a rate of 3 lifts per minute
    with a fair coupling and no requirement for significant control at the destination.
    [TASK:1] [W1:25] [H01:16] [H11:20] [V01:12] [V11:36] [A1:35] [F1:3] [COUP1:fair] [CTRL1:false]

    In the second task, the employee lifts small components weighing 5 pounds from a comfortable working
    level to chest height at approximately 36 inches, with minimal torso rotation and a frequency of
    about 10 lifts per minute. The components have a good hand-to-object coupling and do not require
    significant control at the destination.
    [TASK:2] [W2:5] [H02:18] [H12:20] [V02:30] [V12:36] [A2:5] [F2:10] [COUP2:good] [CTRL2:false]

    In the third task, the employee loads heavier containers weighing 45 pounds onto a high shelf located
    at approximately 72 inches, with the hands reaching about 24 inches horizontally from the body and
    about 45 degrees of torso rotation. This task is performed at a lower rate of about 1 lift per minute
    with a fair coupling and without a requirement for significant control at the destination.
    [TASK:3] [W3:45] [H03:20] [H13:24] [V03:30] [V13:72] [A3:45] [F3:1] [COUP3:fair] [CTRL3:false]

    The overall lifting activity is performed for a substantial portion of the workday.
    [DUR:2-8h]"
    """

        # --- selezione few-shot + context line ---
        if task_type == "repetitive":
            examples_text = examples_repetitive
            context_line = "This is a repetitive lifting task performed at sustained frequency."
        elif task_type == "multi":
            examples_text = examples_multi
            context_line = "This is a multi-task lifting scenario involving distinct lifting subtasks."
        else:
            examples_text = examples_single
            context_line = ""
            task_type = "single"  # fallback

        # ==========================================================
        # 2) ISTRUZIONI UNITÀ
        # ==========================================================

        unit_instruction = (
            "Use IMPERIAL units (lbs, inches) for all measurements."
            if unit_system == "imperial"
            else "Use METRIC units (kg, cm) for all measurements."
        )

        # ==========================================================
        # 3) COERENZA PARAMETRI (SE SONO STATI PASSATI)
        # ==========================================================

        coherence_guidance = ""
        if parameters:
            coherence_guidance = f"""
    COHERENCE REQUIREMENTS (MANDATORY):
    - Use EXACT values: weight ≈ {parameters.weight:.1f} lbs
    - H0 ≈ {parameters.horizontal_origin:.1f} in, H1 ≈ {parameters.horizontal_destination:.1f} in
    - V0 ≈ {parameters.vertical_origin:.1f} in, V1 ≈ {parameters.vertical_destination:.1f} in
    - A ≈ {parameters.asymmetry_angle:.0f}°
    - F ≈ {parameters.frequency_lifts_per_min:.1f} lifts/min
    - Duration category: {parameters.duration}
    - Coupling: {parameters.coupling}
    - Significant control: {"required" if parameters.significant_control else "not required"}
    """

        # ==========================================================
        # 4) COSTRUZIONE USER PROMPT
        # ==========================================================

        if task_type == "multi":
            user_prompt = f"""
    STYLE EXAMPLES (FEW-SHOT):
    {examples_multi}

    {unit_instruction}

    TASK CONTEXT:
    This is a multi-task lifting scenario involving distinct lifting subtasks.

    Generate a NEW multi-task job description following the MULTI-TASK example.

    CONSTRAINTS:
    - Preserve user scenario and environment.
    - Do NOT add tools, equipment, or objects not mentioned by the user.
    - Identify each distinct lifting action as a separate task (in chronological order).
    - For each task i, append:
    [TASK:i] [Wi:value] [H0i:value] [H1i:value] [V0i:value] [V1i:value]
    [Ai:value] [Fi:value] [COUPi:good|fair|poor] [CTRLi:true|false]
    - After all tasks, append exactly ONE duration tag:
    [DUR:<1h] OR [DUR:1-2h] OR [DUR:2-8h] OR [DUR:>8h]
    - Use realistic ergonomic language and NIOSH style.

    USER INPUT:
    {user_input}
    """
        else:
            user_prompt = f"""
    STYLE EXAMPLES (FEW-SHOT):
    {examples_text}

    {unit_instruction}

    {coherence_guidance}

    TASK CONTEXT:
    {context_line}

    Generate a NEW job description following the examples.

    CONSTRAINTS:
    - Preserve user scenario and environment
    - DO NOT create new tools or change load type
    - Include ALL tags exactly once:
    [W:value] [H0:value] [H1:value] [V0:value] [V1:value]
    [A:value] [F:value] [DUR:value] [COUP:good|fair|poor] [CTRL:true|false]
    - Tags MUST appear next to the measurement they refer to
    - Use realistic ergonomic language and NIOSH style

    USER INPUT:
    {user_input}
    """

        # ==========================================================
        # 5) SELEZIONE SYSTEM PROMPT
        # ==========================================================

        system_single = self._create_system_prompt()

        system_repetitive = f"""
You are an expert in NIOSH lifting equation.
This is a REPETITIVE lifting task (LOCKED category).
You MUST emphasize cumulative loading, sustained frequency,
and repeated lifting cycles. Follow NIOSH Applications Manual style strictly.
You MUST NOT change the task type to single or multi-task.
"""

        system_multi = """
    You are an expert in the Revised NIOSH Lifting Equation.
TASK TYPE LOCK: MULTI-TASK.
You MUST generate a multi-task job description and MUST NOT reduce it to single-task
    Your task is to generate MULTI-TASK job descriptions in EXACT NIOSH Applications Manual style.

    You MUST:
    - identify each distinct lifting action as a separate task (TASK:1, TASK:2, TASK:3, …)
    - describe each task in natural language
    - append the correct machine-readable tags for each task

    TAG FORMAT (MANDATORY):
    - For each task i:
    [TASK:i] [Wi:value] [H0i:value] [H1i:value] [V0i:value] [V1i:value]
    [Ai:value] [Fi:value] [COUPi:good|fair|poor] [CTRLi:true|false]

    - AFTER all tasks, exactly ONE duration tag:
    [DUR:<1h] OR [DUR:1-2h] OR [DUR:2-8h] OR [DUR:>8h]

    You MUST NOT:
    - invent new heights, distances, or weights
    - change the scenario, tools, or environment
    - add or remove tasks
    - use any other bracket format

    You must imitate the style and tagging of the MULTI-TASK example in the user prompt.
    """

        if task_type == "repetitive":
            system_prompt = system_repetitive
        elif task_type == "multi":
            system_prompt = system_multi
        else:
            system_prompt = self._create_system_prompt() + f"""

TASK TYPE LOCK:
- You MUST treat this job as a {task_type} task.
- You MUST NOT reinterpret or change the task category based on wording.
- Ignore any linguistic cues that conflict with this classification.
"""

        # ==========================================================
        # 6) CHIAMATA LLM
        # ==========================================================

        jd_text = call_llm_with_system_prompt(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        #  INTEGRAZIONE ESTRATTORE MULTI-TASK: VALIDAZIONE
        if task_type == "multi":
            tasks = extract_multi_task_tags(jd_text)

            if not tasks:
                # QUI SCEGLIAMO COSA FARE:
                #  - versione "strict": solleva errore esplicito
                #  - oppure fai solo warning e lascia jd_text (ma il crash tornerà dopo)
                raise ValueError(
                    "Generated multi-task Job Description does not contain valid [TASK:i] tags "
                    "or could not be parsed by extract_multi_task_tags()."
                )

            # opzionale: potresti loggare/debuggare
            # print(f"[DEBUG] Parsed {len(tasks)} multi-task blocks from JD.")

        if parameters is None:
            return jd_text

        # ==========================================================
        # 7) FIX TAG (SINGLE-TASK / REPETITIVE)
        # ==========================================================

        jd_fixed = fix_or_insert_missing_tags(jd_text, parameters)
        return jd_fixed

    def extract_parameters(self, description: str):
        """
        Estrattore NIOSH robusto basato su metodologia MRI:
        1) Preprocessing
        2) Dizionario controllato
        3) Pattern matching estrattivo
        """

        import re
        from niosh_parameters import NIOSHParameters

        # ===============================
        # 1) PRE-PROCESSING
        # ===============================
        text = description.lower()

        # normalizza unità
        text = text.replace("in.", "in").replace("inch", "inches")
        text = text.replace("lbs", "lb").replace("pounds", "lb")
        text = text.replace("degrees", "deg").replace("degree", "deg")

        # normalizza spazi
        text = re.sub(r"\s+", " ", text)

        # ===============================
        # 2) CONTROLLED VOCABULARY
        # ===============================
        # (WordNet-style mapping)
        VOCAB = {
            "horizontal": "H",
            "reach": "H",
            "origin": "origin",
            "destination": "destination",
            "height": "V",
            "twist": "A",
            "rotation": "A",
            "rotating": "A",
            "rotational": "A",
            "frequency": "F",
            "coupling": "coupling",
            "control": "control",
            "shift": "duration",
            "hours": "duration",
        }

        def find_concept(token):
            return VOCAB.get(token, None)

        # ===============================
        # 3) PATTERN MATCHING (estrattivo)
        # ===============================

        params = {}

        # -------- PESO --------
        m = re.search(r"(\d+(?:\.\d+)?)\s*lb", text)
        if m:
            params["weight"] = float(m.group(1))

        # -------- H ORIGIN --------
        m = re.search(r"(?:horizontal|reach)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*in", text)
        if m:
            params["horizontal_origin"] = float(m.group(1))

        # -------- H DESTINATION --------
        m = re.search(
            r"(?:destination|stacking area)[^0-9]{0,30}(\d+(?:\.\d+)?)\s*in", text
        )
        if m:
            params["horizontal_destination"] = float(m.group(1))

        # -------- V ORIGIN --------
        m = re.search(r"(?:origin|from)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*in", text)
        if m:
            params["vertical_origin"] = float(m.group(1))

        # -------- V DESTINATION --------
        m = re.search(r"(?:destination|to)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*in", text)
        if m:
            params["vertical_destination"] = float(m.group(1))

        # -------- ASYMMETRY --------
        m = re.search(
            r"(?:twist|twisting|rotate|rotation)[^0-9]{0,10}(\d+(?:\.\d+)?)\s*deg",
            text,
        )
        if m:
            params["asymmetry_angle"] = float(m.group(1))

        # -------- FREQUENCY --------
        m = re.search(r"(\d+(?:\.\d+)?)\s*lifts per minute", text)
        if m:
            params["frequency"] = float(m.group(1))

        # -------- DURATION --------
        if "8-hour" in text or "8 hour" in text:
            params["duration"] = "2-8h"
        elif "2-8 hours" in text:
            params["duration"] = "2-8h"
        elif "1-2 hours" in text:
            params["duration"] = "1-2h"
        elif "less than 1 hour" in text:
            params["duration"] = "<1h"

        # -------- COUPLING --------
        m = re.search(r"\b(good|fair|poor|none)\b[^a-z]{0,10}coupling", text)
        if m:
            raw = m.group(1)
            if raw == "none":
                params["coupling"] = "poor"
                params["coupling_note"] = (
                    "none (treated as poor according to NIOSH Table 6)"
                )
            else:
                params["coupling"] = raw

        # -------- SIGNIFICANT CONTROL --------
        if "no significant control" in text:
            params["significant_control"] = False
        elif "significant control" in text:
            params["significant_control"] = True

        # -------- GENERE / ETÀ (ISO) --------
        params["gender"] = "W" if "female" in text else "M"
        m = re.search(r"(\d+)\s*years old", text)
        params["age"] = int(m.group(1)) if m else 25

        # -------- ONE-LIMB / TWO-OPERATORS --------
        params["one_limb_lifting"] = bool(re.search(r"one (hand|arm|limb)", text))
        params["two_operators_lifting"] = bool(
            re.search(r"two (workers|operators|people)", text)
        )

        # ===============================
        # COSTRUZIONE MODELLO
        # ===============================

        # Provide default values for missing required parameters
        default_params = {
            "weight": params.get("weight", 25.0),
            "horizontal_origin": params.get("horizontal_origin", 20.0),
            "horizontal_destination": params.get("horizontal_destination", 20.0),
            "vertical_origin": params.get("vertical_origin", 30.0),
            "vertical_destination": params.get("vertical_destination", 30.0),
            "asymmetry_angle": params.get("asymmetry_angle", 0.0),
            "frequency": params.get("frequency", 1.0),
            "duration": params.get("duration", "1-2h"),
            "coupling": params.get("coupling", "fair"),
            "significant_control": params.get("significant_control", False),
            "gender": params.get("gender", "M"),
            "age": params.get("age", 25),
            "one_limb_lifting": params.get("one_limb_lifting", False),
            "two_operators_lifting": params.get("two_operators_lifting", False),
        }

        # Override with any existing values from extracted params
        default_params.update(params)

        return NIOSHParameters(**default_params)

    def ensure_job_consistency(
        self, user_input: str, description: str, parameters: NIOSHParameters
    ) -> tuple:
        """
        Garantisce coerenza completa tra description e parameters.
        Ritorna description coerente e parameters finali.
        """
        # Prima passata: estrai parametri dalla description generata
        extracted_params = self.extract_parameters(description)

        if extracted_params:
            # Confronta e usa i parametri più affidabili
            # In genere, quelli estratti dal testo sono più coerenti
            final_params = extracted_params
        else:
            # Fallback ai parametri calcolati
            final_params = parameters

        # Seconda passata: rigenera description con parametri coerenti
        raw_desc = self.generate_job_description(user_input, "imperial", final_params)

        # applica sempre autocorrezione tag
        coherent_description = fix_or_insert_missing_tags(raw_desc, final_params)

        return coherent_description, final_params

    def analyze_job(self, user_input: str, task_type: str | None = None) -> Dict:
        """
        Pipeline completa con task_type LOCK:
        - Single task: JD → params → JA → HA
        - Multi task: JD_multi → extraction_multi → JA_multi → CLI → HA_multi
        """

        from tag_extractor import extract_tagged_params, remove_tags
        from multi_task_extractor import extract_multi_task_tags
        from validators import validate_params, intelligent_llm_prevalidator
        from niosh_parameters import NIOSHParameters
        from job_analysis import NIOSHJobAnalysisGenerator
        from hazard_assessment import NIOSHHazardAssessmentGenerator

        print(f"\n{'='*70}")
        print(f"INPUT UTENTE: {user_input}")
        print(f"{'='*70}\n")

        # ----------------------------------------------------------
        # 1 — TASK TYPE LOCK
        # ----------------------------------------------------------
        if task_type in ("single", "repetitive", "multi"):
            resolved_task_type = task_type
            print(f"Task type LOCK esterno: {resolved_task_type}")
        else:
            auto_multi = detect_multi_task(user_input)
            resolved_task_type = "multi" if auto_multi else "single"
            print(f"Riconoscimento automatico: {resolved_task_type.upper()}")

        is_multi = resolved_task_type == "multi"

        # ----------------------------------------------------------
        # 2 — Rilevazione unità
        # ----------------------------------------------------------
        unit_system = self._detect_units(user_input)
        print(f"Sistema unità: {unit_system}")

        # ----------------------------------------------------------
        # 3 — Generazione Job Description (LOCK)
        # ----------------------------------------------------------
        print("\nGenerazione Job Description (LOCKED)...")
        description_with_tags = self.generate_job_description(
            user_input,
            unit_system,
            parameters=None,
            task_type=resolved_task_type,
        )

        print(description_with_tags)
        print("\n" + "-" * 70)

        # ==========================================================
        # 4 — MULTI-TASK PIPELINE
        # ==========================================================
        if is_multi:
            print("→ Modalità MULTI-TASK attivata")

            print("Estrazione dei TASK dai TAG multi-task...")
            raw_tasks = extract_multi_task_tags(description_with_tags)

            if not raw_tasks:
                return {
                    "valid": False,
                    "error": "Impossibile estrarre multi-task data dai tag",
                    "description": description_with_tags,
                }

            # Conversione in formato NIOSHCalculator
            structured_tasks = []
            for t in raw_tasks:
                structured_tasks.append({
                    "task_id": t["task"],
                    # NIOSHCalculator richiede "weight", NON "weight_lbs"
                    "weight": float(t["weight"]) if t["weight"] is not None else 0.0,
                    "horizontal_origin": float(t["H0"]) if t["H0"] is not None else 0.0,
                    "horizontal_destination": float(t["H1"]) if t["H1"] is not None else 0.0,
                    "vertical_origin": float(t["V0"]) if t["V0"] is not None else 0.0,
                    "vertical_destination": float(t["V1"]) if t["V1"] is not None else 0.0,
                    "asymmetry_angle": float(t["A"]) if t["A"] is not None else 0.0,
                    "frequency": float(t["F"]) if t["F"] is not None else 0.0,
                    "coupling": t.get("coupling") or "fair",
                    "duration": t.get("duration") or "2-8h",
                })

            calc = NIOSHCalculator()
            calc_result = calc.compute_multi_task(structured_tasks)

            if not calc_result:
                return {
                    "valid": False,
                    "error": "Errore nel compute_multi_task",
                    "description": description_with_tags,
                }

            # -----------------------------
            # Job Analysis MULTI
            # -----------------------------
            ja_generator = NIOSHJobAnalysisGenerator(model=self.model)
            print("\nGenerazione JOB ANALYSIS MULTI-TASK...")
            ja_multi = ja_generator.generate_multi_task_job_analysis(calc_result)

            print(ja_multi)
            print("\n" + "-" * 70)

            # -----------------------------
            # Hazard Assessment MULTI
            # -----------------------------
            print("Generazione MULTI-TASK HAZARD ASSESSMENT...")
            ha_generator = NIOSHHazardAssessmentGenerator(model=self.model)
            hazard_text = ha_generator.generate_multi_task_hazard_assessment(calc_result)

            return {
                "input": user_input,
                "mode": "multi-task",
                "description": description_with_tags,
                "job_description_with_tags": description_with_tags,
                "job_analysis_with_tags": ja_multi,
                "hazard_assessment": hazard_text,
                "valid": True,
            }

        # ==========================================================
        # 5 — SINGLE-TASK PIPELINE
        # ==========================================================
        print("→ Modalità SINGLE-TASK attivata" +
            (" (REPETITIVE)" if resolved_task_type == "repetitive" else ""))

        raw_params = extract_tagged_params(description_with_tags) or {}
        if "A" not in raw_params and "asymmetry_angle" not in raw_params:
            raw_params["asymmetry_angle"] = 0.0

        try:
            params_filled = intelligent_llm_prevalidator(raw_params, description_with_tags)
        except Exception as e:
            print(f"[WARN] intelligent_llm_prevalidator fallito: {e}")
            params_filled = raw_params

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

        for k, v in default_values.items():
            if k not in params_filled or params_filled[k] is None:
                params_filled[k] = v

        ok, err = validate_params(params_filled)
        if not ok:
            print(f"[ERROR] validate_params fallita: {err}")
            return {
                "valid": False,
                "error": err,
                "description": description_with_tags,
                "parameters": params_filled,
            }

        final_params = NIOSHParameters(**params_filled)

        try:
            description_with_tags = fix_or_insert_missing_tags(
                description_with_tags, final_params
            )
        except Exception as e:
            print(f"[WARN] fix_or_insert_missing_tags fallita: {e}")

        description_clean = remove_tags(description_with_tags)

        ja_generator = NIOSHJobAnalysisGenerator(model=self.model)
        print("\nGenerazione JOB ANALYSIS SINGLE-TASK...")
        ja_text = ja_generator.generate_job_analysis(final_params, task_type=resolved_task_type)

        from hazard_assessment import HazardAssessmentInput
        ha_input = HazardAssessmentInput(
            task_description=description_clean,
            weight_lbs=final_params.weight,
            rwl_origin_lbs=None,
            rwl_dest_lbs=None,
            li_origin=None,
            li_dest=None,
            significant_control=final_params.significant_control,
        )

        ha_generator = NIOSHHazardAssessmentGenerator(model=self.model)
        hazard_text = ha_generator.generate_hazard_assessment(ha_input, task_type=resolved_task_type)

        return {
            "input": user_input,
            "mode": resolved_task_type,
            "description": description_clean,
            "description_clean": description_clean,
            "description_with_tags": description_with_tags,
            "parameters": final_params.model_dump(),
            "job_analysis_with_tags": ja_text,
            "hazard_assessment": hazard_text,
            "valid": True,
        }
