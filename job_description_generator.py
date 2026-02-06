import json
import re
from typing import Dict, List, Optional

import ollama  # usato solo se vuoi in futuro fare chiamate dirette
from jd_ja_consistency import check_jd_ja_consistency
from tag_extractor import extract_tagged_params, remove_tags
from multi_task_extractor import extract_multi_task_tags
from niosh_calculator import NIOSHCalculator
from niosh_parameters import NIOSHParameters
from model_router import call_llm_with_system_prompt, is_gemini_model
from validators import validate_params, intelligent_llm_prevalidator
from task_classifier import classify_sentence

REQUIRED_TAGS = ["W", "H0", "H1", "V0", "V1", "A", "F", "DUR", "COUP", "CTRL"]


# ==========================================================
#  TAG FIXER
# ==========================================================

def fix_or_insert_missing_tags(text: str, params: NIOSHParameters) -> str:
    """
    Garantisce che tutti i tag obbligatori siano presenti, corretti e unici.
    Se mancano, li inserisce usando i parametri estratti.
    Se sono formattati male, li corregge.
    """
    fixed = text

    def ensure(tag: str, val: str):
        nonlocal fixed
        # rimpiazza versione errata (spazi, ecc.)
        fixed = re.sub(rf"\[{tag}\s*:\s*[^\]]*\]", f"[{tag}:{val}]", fixed)
        # inserisce se manca del tutto
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

    # Normalizza spazi
    fixed = re.sub(r"\s+", " ", fixed).strip()
    return fixed


# ==========================================================
#  HEURISTIC: DETECT MULTI-TASK
# ==========================================================

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
        r"\bfrom .* to .* to\b",  # es. "from cart to shelf 1 to shelf 2"
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


# ==========================================================
#  MAIN GENERATOR
# ==========================================================

class NIOSHJobDescriptionGenerator:
    """Generatore di job descriptions NIOSH - sistema ibrido (LLM + parsing)"""

    # ---------- FEW-SHOT DI RIFERIMENTO (USATI ANCHE DAL JUDGE) ----------

    examples_single = """
EXAMPLE — SINGLE-TASK
USER: "The worker lifts boxes from a pallet and places them onto a table."

ASSISTANT: "The worker lifts a box weighing approximately 32 pounds [W:32] from a pallet on the floor.
At the origin, the hands are positioned at approximately 10 inches [V0:10] with a horizontal reach
of about 18 inches [H0:18]. The box is placed onto a table at roughly 34 inches [V1:34] and 
14 inches [H1:14]. The lift involves mild torso rotation of about 20 degrees [A:20]. The task is 
performed at about 2 lifts/min [F:2] with duration less than one hour [DUR:<1h]. Coupling is fair [COUP:fair] 
and significant control is not required [CTRL:false]."
"""

    examples_repetitive = """
EXAMPLE — REPETITIVE SINGLE-TASK
USER: "Lifting containers repeatedly from a low shelf to a higher shelf for inspection."

ASSISTANT: "The worker repeatedly lifts compact containers weighing 26 pounds [W:26] from a low shelf.
The hands start at about 22 inches [V0:22] with 10 inches horizontal reach [H0:10]. Containers
are placed at a higher shelf at 59 inches [V1:59] and 20 inches reach [H1:20]. No trunk rotation
occurs [A:0]. This repetitive lift occurs at 3 lifts/min [F:3] for 45 minutes [DUR:<1h]. Coupling
is fair [COUP:fair], and significant control is required at the destination [CTRL:true]."
"""

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

    def __init__(self, model: str = "gemma3:12b"):
        self.model = model

    # --------------------------------------------------
    # UNIT DETECTION
    # --------------------------------------------------
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

    # --------------------------------------------------
    # SYSTEM PROMPT BASE
    # --------------------------------------------------
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

    # --------------------------------------------------
    # MAIN GENERATION
    # --------------------------------------------------
    def generate_job_description(
        self,
        user_input: str,
        unit_system: str = "imperial",
        parameters: Optional[NIOSHParameters] = None,
        task_type: Optional[str] = None,
    ) -> str:
        """
        Genera la Job Description (single / repetitive / multi),
        con TAG inline in stile NIOSH.
        """

        # 1) Selezione few-shot e contesto
        if task_type == "repetitive":
            examples_text = self.examples_repetitive
            context_line = (
                "This is a repetitive lifting task performed at sustained frequency."
            )
        elif task_type == "multi":
            examples_text = self.examples_multi
            context_line = (
                "This is a multi-task lifting scenario involving distinct lifting subtasks."
            )
        else:
            examples_text = self.examples_single
            context_line = ""
            task_type = "single"

        # 2) Istruzioni unità
        unit_instruction = (
            "Use IMPERIAL units (lbs, inches) for all measurements."
            if unit_system == "imperial"
            else "Use METRIC units (kg, cm) for all measurements."
        )

        # 3) Coerenza parametri (se passati)
        coherence_guidance = ""
        if parameters:
            coherence_guidance = f"""
COHERENCE REQUIREMENTS (MANDATORY):
- Use EXACT values: weight ≈ {parameters.weight:.1f} lbs
- H0 ≈ {parameters.horizontal_origin:.1f} in, H1 ≈ {parameters.horizontal_destination:.1f} in
- V0 ≈ {parameters.vertical_origin:.1f} in, V1 ≈ {parameters.vertical_destination:.1f} in
- A ≈ {parameters.asymmetry_angle:.0f}°
- F ≈ {parameters.frequency:.1f} lifts/min
- Duration category: {parameters.duration}
- Coupling: {parameters.coupling}
- Significant control: {"required" if parameters.significant_control else "not required"}
"""

        # 4) Costruzione user_prompt
        if task_type == "multi":
            user_prompt = f"""
STYLE EXAMPLES (FEW-SHOT):
{self.examples_multi}

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

        # 5) System prompt con TASK TYPE LOCK
        if task_type == "repetitive":
            system_prompt = """
You are an expert in NIOSH lifting equation.
This is a REPETITIVE lifting task (LOCKED category).
You MUST emphasize cumulative loading, sustained frequency,
and repeated lifting cycles. Follow NIOSH Applications Manual style strictly.
You MUST NOT change the task type to single or multi-task.
"""
        elif task_type == "multi":
            system_prompt = """
You are an expert in the Revised NIOSH Lifting Equation.
TASK TYPE LOCK: MULTI-TASK.
You MUST generate a multi-task job description and MUST NOT reduce it to single-task.

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
        else:
            system_prompt = (
                self._create_system_prompt()
                + f"""

TASK TYPE LOCK:
- You MUST treat this job as a {task_type} task.
- You MUST NOT reinterpret or change the task category based on wording.
- Ignore any linguistic cues that conflict with this classification.
"""
            )

        # 6) Chiamata LLM
        jd_text = call_llm_with_system_prompt(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        # 7) Validazione multi-task
        if task_type == "multi":
            tasks = extract_multi_task_tags(jd_text)
            if not tasks:
                raise ValueError(
                    "Generated multi-task Job Description does not contain valid [TASK:i] tags "
                    "or could not be parsed by extract_multi_task_tags()."
                )

        # Se non ho parametri, ritorno così com'è (verrà parsato dopo)
        if parameters is None:
            return jd_text

        # 8) Fix tags per single/repetitive
        jd_fixed = fix_or_insert_missing_tags(jd_text, parameters)
        return jd_fixed

    # --------------------------------------------------
    # RIFERIMENTI PER IL JUDGE (few-shot umani)
    # --------------------------------------------------
    def get_reference_examples(self, task_type: str) -> str:
        """
        Restituisce i few-shot di riferimento per il GeminiJudge.
        """
        if task_type == "single":
            return self.examples_single
        elif task_type == "repetitive":
            return self.examples_repetitive
        elif task_type == "multi":
            return self.examples_multi
        else:
            return self.examples_single

    # --------------------------------------------------
    # ESTRATTORE PARAMETRI GREZZO DAL TESTO
    # --------------------------------------------------
    def extract_parameters(self, description: str) -> NIOSHParameters:
        """
        Estrattore NIOSH robusto basato su pattern matching.
        Ritorna sempre un NIOSHParameters con default sensati.
        """
        text = description.lower()

        # normalizza unità
        text = text.replace("in.", "in").replace("inch", "inches")
        text = text.replace("lbs", "lb").replace("pounds", "lb")
        text = text.replace("degrees", "deg").replace("degree", "deg")
        text = re.sub(r"\s+", " ", text)

        params: Dict[str, object] = {}

        # peso
        m = re.search(r"(\d+(?:\.\d+)?)\s*lb", text)
        if m:
            params["weight"] = float(m.group(1))

        # H origin
        m = re.search(r"(?:horizontal|reach)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*in", text)
        if m:
            params["horizontal_origin"] = float(m.group(1))

        # H dest
        m = re.search(
            r"(?:destination|stacking area)[^0-9]{0,30}(\d+(?:\.\d+)?)\s*in", text
        )
        if m:
            params["horizontal_destination"] = float(m.group(1))

        # V origin
        m = re.search(r"(?:origin|from)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*in", text)
        if m:
            params["vertical_origin"] = float(m.group(1))

        # V dest
        m = re.search(r"(?:destination|to)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*in", text)
        if m:
            params["vertical_destination"] = float(m.group(1))

        # Asimmetria
        m = re.search(
            r"(?:twist|twisting|rotate|rotation)[^0-9]{0,10}(\d+(?:\.\d+)?)\s*deg", text
        )
        if m:
            params["asymmetry_angle"] = float(m.group(1))

        # Frequenza
        m = re.search(r"(\d+(?:\.\d+)?)\s*lifts per minute", text)
        if m:
            params["frequency"] = float(m.group(1))

        # Durata
        if "8-hour" in text or "8 hour" in text:
            params["duration"] = "2-8h"
        elif "2-8 hours" in text:
            params["duration"] = "2-8h"
        elif "1-2 hours" in text:
            params["duration"] = "1-2h"
        elif "less than 1 hour" in text:
            params["duration"] = "<1h"

        # Coupling
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

        # Significant control
        if "no significant control" in text:
            params["significant_control"] = False
        elif "significant control" in text:
            params["significant_control"] = True

        # Genere / età
        params["gender"] = "W" if "female" in text else "M"
        m = re.search(r"(\d+)\s*years old", text)
        params["age"] = int(m.group(1)) if m else 25

        params["one_limb_lifting"] = bool(re.search(r"one (hand|arm|limb)", text))
        params["two_operators_lifting"] = bool(
            re.search(r"two (workers|operators|people)", text)
        )

        # Default
        default_params = {
            "weight": 25.0,
            "horizontal_origin": 20.0,
            "horizontal_destination": 20.0,
            "vertical_origin": 30.0,
            "vertical_destination": 30.0,
            "asymmetry_angle": 0.0,
            "frequency": 1.0,
            "duration": "1-2h",
            "coupling": "fair",
            "significant_control": False,
            "gender": "M",
            "age": 25,
            "one_limb_lifting": False,
            "two_operators_lifting": False,
        }

        default_params.update(params)
        return NIOSHParameters(**default_params)

    # --------------------------------------------------
    # COERENZA JD ↔ PARAMETRI
    # --------------------------------------------------
    def ensure_job_consistency(
        self, user_input: str, description: str, parameters: NIOSHParameters
    ) -> tuple[str, NIOSHParameters]:
        """
        Garantisce coerenza completa tra description e parameters.
        """
        extracted_params = self.extract_parameters(description)

        if extracted_params:
            final_params = extracted_params
        else:
            final_params = parameters

        raw_desc = self.generate_job_description(
            user_input, "imperial", final_params, task_type="single"
        )
        coherent_description = fix_or_insert_missing_tags(raw_desc, final_params)
        return coherent_description, final_params
    
    def get_combined_fewshot(self) -> str:
        """
        Few-shot completo che include:
        - example SINGLE
        - example REPETITIVE
        - example MULTI
        (NON sostituisce quelli originali, li completa)
        """
        return f"""
==================== FEW-SHOT — ALL TASK TYPES ====================

--- SINGLE-TASK EXAMPLE ---
{self.examples_single}

--- REPETITIVE-TASK EXAMPLE ---
{self.examples_repetitive}

--- MULTI-TASK EXAMPLE ---
{self.examples_multi}

====================================================================
"""

    def generate_dual_version(
        self,
        user_input: str,
        unit_system: str = "imperial",
        task_type: Optional[str] = None,
        parameters: Optional[NIOSHParameters] = None,
    ):
        """
        Genera DUE versioni:
        1) specific_version → usa il few-shot corrispondente al task_type
        2) combined_version → usa tutti gli esempi insieme
        """
        # --- Versione SPECIFICA (quella normale)
        specific = self.generate_job_description(
            user_input=user_input,
            unit_system=unit_system,
            parameters=parameters,
            task_type=task_type
        )

        # --- Versione COMBINATA
        # Costruisco un prompt identico, ma sostituisco il few-shot originale
        combined_fewshot = self.get_combined_fewshot()

        # Ricostruisco l'user_prompt della versione specifica,
        # sostituendo SOLO il blocco few-shot.
        # È più sicuro rigenerare il prompt manualmente:

        combined_user_prompt = f"""
    STYLE EXAMPLES (FULL MANIFOLD):
    {combined_fewshot}

    USER INPUT:
    {user_input}

    Generate a NEW job description using the FULL NIOSH manifold,
    while respecting the task_type="{task_type}" lock.
    """

        combined = call_llm_with_system_prompt(
            model=self.model,
            system_prompt=self._create_system_prompt(),
            user_prompt=combined_user_prompt,
            temperature=0.0,
        )

        return {
            "specific_version": specific,
            "combined_version": combined
        }
    
    def generate_multi_task_job_description(self, calc_result, task_type):
        """
        Generates ONLY the single multi-task job description text (non-dual).
        """
        dual = self.generate_multi_task_dual_job_description(calc_result, task_type)
        return dual["specific_version"]

    def generate_multi_task_dual_job_description(self, calc_result, task_type: str = "multi"):
        """
        Restituisce due versioni della JOB DESCRIPTION MULTI-TASK:
        - specific_version: JD multi-task generata dal modello con il few-shot MULTI
        - combined_version: JD multi-task generata aprendosi all’intero manifold (single + rep + multi)
        """

        # ======================================================
        # 1) COSTRUZIONE BLOCCHI TASK DA calc_result
        # ======================================================

        task_blocks = []
        for t in calc_result.tasks:
            task_blocks.append(
                f"[TASK:{t.task_id}] "
                f"[Wi:{t.weight_lbs}] "
                f"[H0i:{t.horizontal_origin}] "
                f"[H1i:{t.horizontal_destination}] "
                f"[V0i:{t.vertical_origin}] "
                f"[V1i:{t.vertical_destination}] "
                f"[Ai:{t.asymmetry_angle}] "
                f"[Fi:{t.frequency_lifts_per_min}] "
                f"[COUPi:{t.coupling}] "
                f"[CTRLi:{str(t.significant_control).lower()}]"
            )

        tasks_block_str = "\n".join(task_blocks)
        duration = getattr(calc_result, "duration", "2-8h")

        # ======================================================
        # 2) VERSIONE SPECIFICA (usa solo il few-shot MULTI)
        # ======================================================

        specific_prompt = f"""
    STYLE EXAMPLE — MULTI-TASK:
    {self.examples_multi}

    Generate a new multi-task Job Description.
    Respect ALL numeric values. NEVER invent new distances or weights.

    TASKS (DO NOT MODIFY):
    {tasks_block_str}

    Include exactly ONE duration tag: [DUR:{duration}]
    """

        system_prompt_specific = (
            "You are an expert in NIOSH multi-task Job Description writing. "
            "Follow EXACTLY the NIOSH Applications Manual style. "
            "NEVER modify numeric values, NEVER add tools, never change the scenario."
        )

        jd_specific = call_llm_with_system_prompt(
            model=self.model,
            system_prompt=system_prompt_specific,
            user_prompt=specific_prompt,
            temperature=0.0,
        )

        # ======================================================
        # 3) VERSIONE COMBINATA (full manifold JOB DESCRIPTIONS)
        # ======================================================

        full_fewshot = (
            "\n--- SINGLE JD ---\n" + self.examples_single +
            "\n--- REPETITIVE JD ---\n" + self.examples_repetitive +
            "\n--- MULTI JD ---\n" + self.examples_multi
        )

        combined_prompt = f"""
    FULL MANIFOLD OF NIOSH JD EXAMPLES:
    {full_fewshot}

    NOW GENERATE A MULTI-TASK JOB DESCRIPTION USING THE FULL STYLE SPACE.

    RULES:
    - task_type="multi" MUST be respected
    - NEVER modify numeric values
    - NEVER add/remove tasks
    - Use the provided TAGS exactly as they appear

    TASKS (FROM PYTHON):
    {tasks_block_str}

    Insert ONE duration tag: [DUR:{duration}]

    Composite Lifting Index (context only): {calc_result.cli}
    Risk category: {calc_result.risk_category}
    """

        system_prompt_combined = (
            "You write Job Descriptions using the complete NIOSH stylistic manifold. "
            "NEVER touch numeric values. NEVER change the scenario. "
            "Always produce a coherent multi-task JD with correct tags."
        )

        jd_combined = call_llm_with_system_prompt(
            model=self.model,
            system_prompt=system_prompt_combined,
            user_prompt=combined_prompt,
            temperature=0.0,
        )

        # ======================================================
        # 4) RITORNO FINALE
        # ======================================================

        return {
            "specific_version": jd_specific,
            "combined_version": jd_combined,
        }


    def analyze_job(self, user_input: str, task_type: Optional[str] = None) -> Dict:
        """
        Pipeline completa con task_type LOCK.
        Versione completamente corretta:
        - JD generata UNA SOLA VOLTA
        - Multi-task generato correttamente DOPO l'estrazione dei task
        - Nessuna ricorsione di retta o indiretta
        - Nessun doppio flusso JD→dual→reanalyze
        """

        print(f"\n{'='*70}")
        print(f"INPUT UTENTE: {user_input}")
        print(f"{'='*70}\n")

        # ================================================================
        # 1 — TASK TYPE RESOLUTION (detect_multi + classify_sentence)
        # ================================================================
        forced_multi = detect_multi_task(user_input)
        classified = classify_sentence(user_input)

        print(f"[DEBUG] detect_multi_task → {forced_multi}")
        print(f"[DEBUG] classify_sentence → {classified}")

        # LOCK esterno
        if task_type in ("single", "repetitive", "multi"):
            resolved_task_type = task_type
            print(f"Task type LOCK esterno: {resolved_task_type}")
        else:
            if forced_multi:
                resolved_task_type = "multi"
            else:
                resolved_task_type = classified
            print(f"Riconoscimento automatico: {resolved_task_type.upper()}")

        is_multi = resolved_task_type == "multi"

        # ================================================================
        # 2 — Rilevazione unità
        # ================================================================
        unit_system = self._detect_units(user_input)
        print(f"Sistema unità: {unit_system}")

        # ================================================================
        # 3 — JD (LOCK sul task type)
        # ================================================================
        print("\nGenerazione Job Description (LOCKED)...")

        jd_initial = self.generate_job_description(
            user_input=user_input,
            unit_system=unit_system,
            parameters=None,
            task_type=resolved_task_type
        )

        # JD pulita
        jd_clean_initial = remove_tags(jd_initial)

        # =====================================================================
        # =========================== MULTI-TASK ==============================
        # =====================================================================
        if is_multi:
            print("→ Modalità MULTI-TASK attivata")
            print("Estrazione dei TASK dai TAG multi-task...")

            raw_tasks = extract_multi_task_tags(jd_initial)

            if not raw_tasks:
                return {
                    "valid": False,
                    "error": "Impossibile estrarre dati multi-task dai tag",
                    "description": jd_initial,
                }

            structured = []
            for t in raw_tasks:
                structured.append({
                    "task_id": t["task"],
                    "weight": float(t["weight"] or 0.0),
                    "horizontal_origin": float(t["H0"] or 20.0),
                    "horizontal_destination": float(t["H1"] or 20.0),
                    "vertical_origin": float(t["V0"] or 30.0),
                    "vertical_destination": float(t["V1"] or 40.0),
                    "asymmetry_angle": float(t["A"] or 0.0),
                    "frequency": float(t["F"] or 0.2),
                    "duration": t.get("duration") or "2-8h",
                    "coupling": t.get("coupling") or "fair",
                    "significant_control": True,
                })

            calc = NIOSHCalculator()
            calc_result = calc.compute_multi_task(structured)

            if not calc_result:
                return {
                    "valid": False,
                    "error": "Errore nel compute_multi_task",
                    "description": jd_initial,
                }

            # JD SPECIFICA + COMBINATA (MULTI TASK CORRETTO)
            dual_mt = self.generate_multi_task_dual_job_description(
                calc_result, resolved_task_type
            )

            jd_specific = remove_tags(dual_mt["specific_version"])
            jd_combined = remove_tags(dual_mt["combined_version"])

            return {
                "input": user_input,
                "mode": "multi",
                "description": jd_initial,
                "description_clean": jd_clean_initial,
                "jd_specific": jd_specific,
                "jd_combined": jd_combined,
                "calculations": calc_result.as_dict(),
                "valid": True,
            }

        # =====================================================================
        # ======================= SINGLE / REPETITIVE =========================
        # =====================================================================
        print("→ Modalità SINGLE-TASK attivata"
            + (" (REPETITIVE)" if resolved_task_type == "repetitive" else ""))

        # ================================================================
        # 4 — Estrai parametri dai TAG
        # ================================================================
        raw_params = extract_tagged_params(jd_initial) or {}

        if "A" not in raw_params:
            raw_params["asymmetry_angle"] = 0.0

        try:
            params_filled = intelligent_llm_prevalidator(raw_params, jd_initial)
        except Exception as e:
            print(f"[WARN] intelligent_llm_prevalidator fallito: {e}")
            params_filled = raw_params

        # Default robusti
        defaults = {
            "weight": 25.0,
            "horizontal_origin": 20.0,
            "horizontal_destination": 20.0,
            "vertical_origin": 30.0,
            "vertical_destination": 30.0,
            "asymmetry_angle": 0.0,
            "frequency": 0.2,
            "duration": "<1h",
            "coupling": "fair",
            "significant_control": False,
            "gender": "M",
            "age": 25,
            "one_limb_lifting": False,
            "two_operators_lifting": False,
        }

        for k, v in defaults.items():
            if k not in params_filled or params_filled[k] is None:
                params_filled[k] = v

        ok, err = validate_params(params_filled)
        if not ok:
            print(f"[ERROR] validate_params fallita: {err}")
            return {
                "valid": False,
                "error": err,
                "description": jd_initial,
                "parameters": params_filled,
            }

        final_params = NIOSHParameters(**params_filled)

        try:
            jd_fixed = fix_or_insert_missing_tags(jd_initial, final_params)
        except Exception as e:
            print(f"[WARN] fix_or_insert_missing_tags fallita: {e}")
            jd_fixed = jd_initial

        # JD SPECIFICA/COMBINATA = quelle già generate sopra (UNA SOLA VOLTA)
        dual_versions = self.generate_dual_version(
            user_input=user_input,
            unit_system=unit_system,
            task_type=resolved_task_type,
            parameters=final_params
        )

        jd_specific = remove_tags(dual_versions["specific_version"])
        jd_combined = remove_tags(dual_versions["combined_version"])

        return {
            "input": user_input,
            "mode": resolved_task_type,
            "description": jd_fixed,
            "description_clean": remove_tags(jd_fixed),
            "jd_specific": jd_specific,
            "jd_combined": jd_combined,
            "parameters": final_params.model_dump(),
            "valid": True,
        }
