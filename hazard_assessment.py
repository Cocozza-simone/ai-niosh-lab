"""
hazard_assessment.py

Genera la sezione "Hazard Assessment" in stile NIOSH (Applications Manual)
a partire da:
- peso sollevato,
- RWL a origine/destinazione,
- LI a origine/destinazione (già calcolati in Python).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import ollama
from pydantic import BaseModel, Field

# Multi-agent coordinator import removed - standard mode only
# ML system import removed - standard mode only


# ======================= DATA CLASS DI INPUT =======================


@dataclass
class HazardAssessmentInput:
    """
    Contiene TUTTE le info numeriche per scrivere la Hazard Assessment.

    Nessun valore deve essere modificato dal modello.
    """

    task_description: str  # solo per contesto, NON per cambiare i numeri
    weight_lbs: float  # peso reale del carico (lbs)

    rwl_origin_lbs: float  # RWL all'origine
    rwl_dest_lbs: Optional[float]  # RWL a destinazione (None se non calcolato)

    li_origin: float  # LI all'origine
    li_dest: Optional[float]  # LI a destinazione (None se non calcolato)

    significant_control: bool  # se True → RWL calcolato anche a destinazione


# ======================= STRUCTURED OUTPUT MODEL =======================


class HazardAssessmentOutput(BaseModel):
    """Struttura per l'output dell'Hazard Assessment"""

    weight_vs_rwl_comparison: str = Field(
        ..., description="Sentence comparing actual weight to RWL(s)."
    )
    li_definition: str = Field(
        ..., description="Sentence defining the LI values (W/RWL)."
    )
    risk_interpretation: str = Field(
        ..., description="Sentence interpreting the risk level relative to LI=1.0."
    )
    narrative_text: str = Field(
        ...,
        description="The complete single paragraph combining the above elements in NIOSH style.",
    )


# ==================== GENERATORE DI HAZARD ASSESSMENT ====================


class NIOSHHazardAssessmentGenerator:
    """
    Standard NIOSH hazard assessment generator using traditional analysis methods.
    Generates professional hazard assessment sections following NIOSH Applications Manual style.
    """

    def __init__(self, model: str = "gemma3:12b"):
        self.model = model

    # ---------- PROMPT DI SISTEMA ----------
        # ---------- FEW-SHOT PROVIDER FOR GEMINI JUDGE ----------
    def get_reference_examples(self, task_type: str) -> str:
        """
        Restituisce i few-shot corretti da usare per il Gemini Judge
        in base alla categoria del task (single / repetitive / multi).
        """

        # --- SINGLE-TASK (default) ---
        single_example = """
EXAMPLE — HAZARD ASSESSMENT (NIOSH Applications Manual)

"The weight to be lifted is 44 lbs [W:44], which exceeds the RWL at both the
origin (16.3 lbs [RWL0:16.3]) and destination (14.5 lbs [RWL1:14.5]). As a result,
the LI is 2.7 at the origin [LI0:2.7] and 3.0 at the destination [LI1:3.0], both
well above the design guideline of 1.0. These values indicate that this lift would
be hazardous for most healthy workers [CTRL:true]."
"""

        # --- REPETITIVE SINGLE-TASK ---
        repetitive_example = """
EXAMPLE — REPETITIVE HAZARD ASSESSMENT

"For this repetitive lifting task, the weight of 26 lbs [W:26] exceeds the
reduced RWL at the origin (17.4 lbs [RWL0:17.4]) due to the effect of the
Frequency Multiplier. The resulting LI of 1.49 [LI0:1.49] indicates increased
physiological strain under sustained repetition. Even when LI is near 1.0,
repetitive lifting magnifies cumulative fatigue and increases the risk of injury."
"""

        # --- MULTI-TASK ---
        multi_example = """
EXAMPLE — MULTI-TASK HAZARD ASSESSMENT

"Across the job’s multiple lifting tasks, individual LIs range from 1.25 [LIi:1.25]
to 1.51 [LIi:1.51], each exceeding the RNLE design goal of 1.0. When evaluated
together, the Composite Lifting Index is 2.3 [CLI:2.3], indicating that the job is
physically stressful for many workers over the shift [DUR:2-8h]."
"""

        if task_type == "repetitive":
            return repetitive_example
        elif task_type == "multi":
            return multi_example
        else:
            return single_example

    def _create_system_prompt(self) -> str:
        return """
You are an expert in occupational ergonomics and the Revised NIOSH Lifting Equation.
If task_type != "multi", you MUST NOT describe more than one lifting task.

TASK TYPE LOCK (CRITICAL):
You MUST treat this job as {task_type}.
You MUST NOT reinterpret or modify the task category.
You MUST ignore any linguistic cues in the task description that conflict with the externally provided task_type.
The category (single / repetitive) is EXTERNALLY FIXED and MUST NOT be changed.

Your task is to write the HAZARD ASSESSMENT section, using ONLY the numerical values
(weight, RWL, LI) provided in the user prompt.

If task_type="repetitive", you MUST explicitly mention:
- cumulative loading,
- reduced tolerance under sustained frequency,
- greater fatigue accumulation,
- increased physiological strain,
even when LI might appear near the acceptable limit.

FEW-SHOT EXAMPLES (FROM NIOSH APPLICATIONS MANUAL)
These are canonical examples that demonstrate EXACT style, logic, and risk language.
Tags are added for your learning; follow this pattern.

---------------------------------------------------------------------------
EXAMPLE 1 — HAZARD ASSESSMENT (PDF Example 1)

weight_vs_rwl_comparison:
"The weight to be lifted is 44 lbs [W:44], which is greater than the RWL at both the origin (16.3 lbs [RWL0:16.3]) and destination (14.5 lbs [RWL1:14.5])."

li_definition:
"Accordingly, the LI at the origin is 44/16.3 or 2.7 [LI0:2.7], and the LI at the destination is 44/14.5 or 3.0 [LI1:3.0]."

risk_interpretation:
"These LI values substantially exceed the design goal of 1.0 and indicate that the task would be hazardous for a majority of healthy industrial workers."

narrative_text:
"The weight to be lifted is 44 lbs [W:44], which exceeds the RWL at both the origin (16.3 lbs [RWL0:16.3]) and destination (14.5 lbs [RWL1:14.5]). As a result, the LI is 2.7 at the origin [LI0:2.7] and 3.0 at the destination [LI1:3.0], both well above the design guideline of 1.0. These results indicate that this lift would be hazardous for most healthy workers [CTRL:true]."

---------------------------------------------------------------------------


You MUST NOT:
- invent numbers
- modify RWL or LI
- change the scenario
- introduce new equipment or assumptions

===========================================================================
TAG RULES (MANDATORY)
Tags MUST appear inline using EXACT syntax:

[W:<value>]         → actual load weight  
[RWL0:<value>]      → RWL at origin  
[RWL1:<value>]      → RWL at destination (if applicable)  
[LI0:<value>]       → LI at origin  
[LI1:<value>]       → LI at destination (if applicable)  
[CTRL:<true|false>] → significant control required  

NO other brackets may be used.
NO spaces inside tag names.

You MUST attach tags inside the narrative, not at the end.

EXAMPLES OF CORRECT TAGGING:
"The weight to be lifted is 44 lbs [W:44]."
"RWL at the origin is 16.3 lbs [RWL0:16.3]."
"At the destination, the RWL is 14.5 lbs [RWL1:14.5]."
"The resulting LI is 2.7 at the origin [LI0:2.7]."
"Significant control is required [CTRL:true]."

===========================================================================

STRUCTURE (MANDATORY – NIOSH STYLE)
Your output must contain 4 JSON fields, each a natural-language sentence WITH TAGS:

1. weight_vs_rwl_comparison  
   - Compare W to RWL(s), exactly as in the manual.
   - MUST contain [W:], [RWL0:], and if available [RWL1:].

2. li_definition  
   - Define LI explicitly using provided values.
   - MUST contain [LI0:] and if available [LI1:].

3. risk_interpretation  
   - Interpret LI relative to LI = 1.0 design goal.
   - Use NIOSH terminology:
       • LI < 1.0 → "acceptable for nearly all workers"
       • 1.0–3.0 → "physically stressful for many workers" / "moderately stressful"
       • >3.0 → "hazardous for a majority of workers"

4. narrative_text  
   - A single paragraph combining all elements, containing all tags again.


You MUST follow this exact style, logic, risk language, and TAG usage.
"""

    # ---------- PROMPT UTENTE (DATI) ----------

    def _build_user_prompt(
        self, data: HazardAssessmentInput, task_type: str = None
    ) -> str:
        """
        Build user prompt for Hazard Assessment generation,
        with explicit contextualization for repetitive single-task
        and tag-style inline reinforcement.
        """
        task_lock = f"\nTASK TYPE: {task_type}\nThis classification is externally provided and MUST NOT be changed.\n"

        # Determine if destination RWL/LI are applicable
        if (
            data.rwl_dest_lbs is None
            or data.li_dest is None
            or not data.significant_control
        ):
            mode = "origin_only"
        else:
            mode = "origin_and_destination"

        # Safe numeric handling
        weight_lbs = data.weight_lbs if data.weight_lbs is not None else 0
        rwl_origin_lbs = data.rwl_origin_lbs if data.rwl_origin_lbs is not None else 0
        li_origin = data.li_origin if data.li_origin is not None else 0

        # contesto per task ripetitivi
        if task_type == "repetitive":
            repetitive_context = (
                "\nNOTE: This is a repetitive lifting task performed at sustained frequency "
                "and duration. The Hazard Assessment MUST explicitly reference cumulative "
                "physical demand, repetition-related fatigue, and the importance of the "
                "Frequency Multiplier in interpreting the LI.\n"
            )
        else:
            repetitive_context = ""

        # Base info with TAG-style inline cues
        base_info = f"""
    TASK SCENARIO (context only — do NOT alter):
    {data.task_description}

    The following numerical results have already been computed
    according to the Revised NIOSH Lifting Equation (RNLE)
    and MUST NOT be changed:

    - Actual load weight: {weight_lbs:.1f} lbs [W:{weight_lbs:.1f}]
    - Recommended Weight Limit at origin: {rwl_origin_lbs:.1f} lbs
    - Lifting Index at origin: {li_origin:.2f}
    {repetitive_context}
    """

        # Destination-handling branch
        if mode == "origin_and_destination":
            rwl_dest_lbs = data.rwl_dest_lbs if data.rwl_dest_lbs is not None else 0
            li_dest = data.li_dest if data.li_dest is not None else 0

            extra_info = f"""- RWL at the destination: {rwl_dest_lbs:.1f} lbs
    - LI at the destination: {li_dest:.2f}
    - Significant control at destination: yes [CTRL:true]
    """

            instructions = """
    Using ONLY these values, write the HAZARD ASSESSMENT section:

    - Reference the LI at origin AND destination, using the values provided.
    - Interpret how stressful the lift is relative to the NIOSH design goal of LI = 1.0.
    - If task_type is repetitive:
    * Explicitly mention the cumulative effect of repetition,
        increased physiological demand,
        and the Frequency Multiplier’s role in reducing RWL.
    * Clarify whether repetitiveness worsens risk even if LI ≈ 1.0.

    - The assessment must follow the tone and structure of the
    NIOSH Applications Manual: precise, numeric, deterministic,
    and WITHOUT inventing any new values.
    """

        else:
            extra_info = """- RWL at the destination: not computed (origin-only evaluation)
    - LI at the destination: not computed
    - Significant control at destination: no [CTRL:false]
    """

            instructions = """
    Using ONLY these values, write the HAZARD ASSESSMENT section:

    - Compare the load weight to the RWL at the origin.
    - State and interpret the LI at the origin.
    - Conclude the level of stress (acceptable / slightly stressful /
    moderately stressful / hazardous) using the RNLE design goal LI = 1.0.

    - If task_type is repetitive:
    * Explicitly address cumulative loading, fatigue potential,
        and how repetition amplifies stress despite the same RNLE formula.
    * Mention why repetitive lifting reduces tolerance even when RWL
        appears close to acceptable levels.

    Maintain strict compliance with RNLE terminology and NEVER alter numeric values.
    """

        return task_lock + base_info + extra_info + "\n" + instructions


    # ---------- CHIAMATA A OLLAMA ----------

    def _create_multi_task_system_prompt(self) -> str:
        return """
You are an expert in the Revised NIOSH Lifting Equation.

Your task is to generate a MULTI-TASK JOB ANALYSIS + MULTI-TASK HAZARD ASSESSMENT,
following EXACTLY the structure and tone of the NIOSH Applications Manual
(Examples 7 and 8), with TAGS inline for deterministic extraction.

CRITICAL RULES — DO NOT VIOLATE

1. NUMERICAL VALUES
- Use ONLY the numerical values explicitly provided in the user prompt.
- NEVER invent, modify, interpolate, or "estimate" any number.
- LIi (single-task Lifting Index values) and CLI (Composite Lifting Index) are
  ALREADY computed in Python and passed to you in the prompt.
- You MUST NOT recompute LI or CLI. You only restate and interpret them.
- Never swap task order or rename tasks.

2. TAG RULES (MANDATORY)
Each task *i* MUST include inline tags (no other bracket types allowed):

[TASK:i]
[Wi:value]       → weight for task i
[H0i:value]      → horizontal origin
[H1i:value]      → horizontal destination
[V0i:value]      → vertical origin
[V1i:value]      → vertical destination
[Ai:value]       → asymmetry angle (deg)
[Fi:value]       → lifting frequency for task i
[RWLi:value]     → RWL for task i (FIRWL/STRWL as passed from Python)
[LIi:value]      → single-task lifting index for task i (precomputed in Python)
[COUPi:good|fair|poor]

GLOBAL TAGS:
[CLI:value]      → composite lifting index for the whole job (precomputed in Python)
[DUR:value]      → duration category (<1h, 1-2h, 2-8h, >8h)

You MUST only repeat the LIi and CLI values exactly as they appear in the user prompt.
Do NOT perform any division or recomputation (no W/RWL, no weighted combinations).

3. NIOSH STYLE REQUIREMENTS
- You MUST use the same analytic style as NIOSH Examples 7 and 8.
- No creative writing. No new equipment. No assumptions.
- The MULTI-TASK analysis must have THREE blocks:

BLOCK 1 — INTRODUCTION
A short introduction explaining that the job has multiple tasks with varying demands.

BLOCK 2 — MEASUREMENTS / TASK DETAILS
Describe the measurement data for each task using natural language and TAGS,
including Wi, H0i, H1i, V0i, V1i, Ai, Fi, RWLi, LIi, COUPi, and DUR.

BLOCK 3 — MULTI-TASK ANALYSIS AND HAZARD ASSESSMENT
Using ONLY the LIi and CLI values provided in the prompt (precomputed in Python):
- Reference the individual LIi values for each task.
- Reference the CLI value for the whole job.
- Interpret the CLI relative to the design guideline of LI = 1.0.
You MUST NOT attempt to recompute CLI from LIi or frequencies.

4. HAZARD INTERPRETATION
Use ONLY NIOSH language:
- CLI < 1.0 → "acceptable for nearly all healthy workers"
- CLI 1.0–3.0 → "physically stressful for many workers" / "moderately stressful"
- CLI > 3.0 → "hazardous for a majority of workers"

5. OUTPUT FORMAT
A single continuous narrative, well-structured in 3 blocks (Intro, Task Analysis, CLI Assessment).
All required TAGS MUST appear inline.

You MUST follow this exact style, tone, structure, and TAG usage in your output, and you MUST
treat all LIi and CLI values as fixed, precomputed inputs from Python (no recomputation).
"""

    def generate_hazard_assessment(self, data: HazardAssessmentInput, task_type: str = None) -> str:
        """
        Genera la Hazard Assessment con TASK TYPE LOCK.
        """
        user_prompt = self._build_user_prompt(data, task_type)

        # Inseriamo il task_type nel system prompt
        system_prompt = self._create_system_prompt().replace("{task_type}", str(task_type))

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=HazardAssessmentOutput.model_json_schema(),
            )

            structured_response = HazardAssessmentOutput.model_validate_json(
                response.message.content
            )
            return structured_response.narrative_text.strip()

        except Exception as e:
            return f"Errore generazione hazard assessment: {e}"

    def generate_multi_task_hazard_assessment(self, calc_result) -> str:
        """
        Genera la HAZARD ASSESSMENT multi-task usando SOLO i valori calcolati
        da Python (LIi, RWLi, CLI). Nessuna ricomputazione. Nessuna modifica
        dei numeri. Nessuna reinterpretazione della RNLE.
        Accetta sia un MultiTaskCalculationResult sia un dict e li normalizza.
        """

        # ---------------------------------------------------------
        # NORMALIZZAZIONE INPUT (accetta dict oppure oggetto)
        # ---------------------------------------------------------
        if isinstance(calc_result, dict):

            # adapter minimale per sicurezza
            class _T:
                def __init__(self, d):
                    self.task_id = d.get("task_id") or d.get("task") or 0
                    self.weight_lbs = float(d["weight"])
                    self.horizontal_origin = float(d["horizontal_origin"])
                    self.horizontal_destination = float(d["horizontal_destination"])
                    self.vertical_origin = float(d["vertical_origin"])
                    self.vertical_destination = float(d["vertical_destination"])
                    self.asymmetry_angle = float(d["asymmetry_angle"])
                    self.frequency_lifts_per_min = float(d["frequency"])
                    self.coupling = d["coupling"]
                    self.strwl_lbs = float(d["strwl_lbs"])
                    self.stli = float(d["stli"])

            class _C:
                pass

            c = _C()
            c.job_description = calc_result.get("job_description", "")
            c.cli = float(calc_result.get("cli", 0.0))
            c.risk_category = calc_result.get("risk_category", "")
            c.duration = calc_result.get("duration", "2-8h")
            c.tasks = [_T(td) for td in calc_result.get("tasks", [])]

            calc_result = c

        # ---------------------------------------------------------
        # FEW-SHOT MULTI-TASK CORRETTO (NIOSH-STYLE)
        # ---------------------------------------------------------
        fewshot = """
    ASSISTANT EXAMPLE — MULTI-TASK HAZARD ASSESSMENT (STYLE TO IMITATE)

    "The job consists of several distinct lifting tasks performed under different
    geometric and frequency conditions. Task 1 [TASK:1] involves a precomputed
    single-task Lifting Index of 1.51 [LI1:1.51] based on its measured weight
    [Wi:28], horizontal distances [H01:16] [H11:14], vertical heights [V01:12]
    [V11:28], asymmetry [Ai:10], and frequency [Fi:3]. The corresponding
    recommended weight limit for the task is 18.6 lb [RWLi:18.6], and the coupling
    is classified as fair [COUPi:fair].

    Task 2 [TASK:2] has a single-task Lifting Index of 1.37 [LI2:1.37], with a
    recommended weight limit of 17.5 lb [RWLi:17.5], based on its measured task
    variables [Wi:24] [H02:18] [H12:16] [V02:22] [V12:44] [Ai:25] [Fi:2] and fair
    coupling [COUPi:fair].

    Although each task individually exceeds the RNLE design goal of LI = 1.0,
    their combined effect is best represented by the Composite Lifting Index.
    When evaluated together, the overall job yields a CLI of 2.3 [CLI:2.3]. 
    According to NIOSH guidance, a CLI between 1.0 and 3.0 indicates a job that is
    physically stressful for many workers over the work period [DUR:2-8h], even
    when individual tasks may appear marginally acceptable on their own."
    """

        # ---------------------------------------------------------
        # COSTRUZIONE BLOCCHI TASK (TUTTI TAG REALI)
        # ---------------------------------------------------------
        tasks_block = []
        for t in calc_result.tasks:
            tasks_block.append(
                f"[TASK:{t.task_id}] "
                f"[Wi:{t.weight_lbs}] "
                f"[H0i:{t.horizontal_origin}] "
                f"[H1i:{t.horizontal_destination}] "
                f"[V0i:{t.vertical_origin}] "
                f"[V1i:{t.vertical_destination}] "
                f"[Ai:{t.asymmetry_angle}] "
                f"[Fi:{t.frequency_lifts_per_min}] "
                f"[RWLi:{t.strwl_lbs:.1f}] "
                f"[LIi:{t.stli:.2f}] "
                f"[COUPi:{t.coupling}]"
            )

        tasks_block_str = "\n".join(tasks_block)

        # ---------------------------------------------------------
        # DURATA (default sicuro)
        # ---------------------------------------------------------
        duration = getattr(calc_result, "duration", "2-8h")

        # ---------------------------------------------------------
        # USER PROMPT NIOSH-FORMATTED (DETERMINISTICO)
        # ---------------------------------------------------------
        user_prompt = f"""
    FEW-SHOT EXAMPLE (IMITATE THIS EXACT STYLE):
    {fewshot}

    MULTI-TASK HAZARD ASSESSMENT INPUT
    Job description (context only, do NOT alter):
    {calc_result.job_description}

    Duration category: [DUR:{duration}]

    Computed values (DO NOT CHANGE):
    CLI = {calc_result.cli:.2f} [CLI:{calc_result.cli:.2f}]
    Risk category (Python-derived): {calc_result.risk_category}

    Precomputed task data:
    {tasks_block_str}

    INSTRUCTIONS:
    Write the MULTI-TASK HAZARD ASSESSMENT:

    • Follow exactly the structure of NIOSH Examples 7–8:
    Intro → Task interpretations with tags → CLI interpretation.

    • Use ONLY the values provided.
    • DO NOT recompute LIi, RWLi, or CLI.
    • ALL tags MUST appear inline exactly once per referenced value.
    • Interpret CLI as:
        - CLI < 1.0 → "acceptable for nearly all workers"
        - 1.0 ≤ CLI ≤ 3.0 → "physically stressful for many workers"
        - CLI > 3.0 → "hazardous for a majority of workers"

    Your output MUST be a single continuous narrative paragraph.
    """

        # ---------------------------------------------------------
        # CHIAMATA MODELLO
        # ---------------------------------------------------------
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in NIOSH multi-task hazard assessment. "
                            "Follow ALL RNLE rules exactly. NEVER change numbers."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response["message"]["content"].strip()

        except Exception as e:
            return f"Errore generazione hazard multi-task: {e}"

    def generate_dual_hazard_assessment(self, data: HazardAssessmentInput, task_type: str):
        """
        Restituisce due versioni della Hazard Assessment:
        - ha_specific: generata normalmente rispettando task_type
        - ha_combined: generata usando TUTTI i few-shot insieme (single + repetitive + multi)
        """

        # 1) Versione specifica (standard)
        ha_specific = self.generate_hazard_assessment(data, task_type)

        # 2) Costruzione few-shot combinato
        full_fewshot = (
            self.get_reference_examples("single")
            + "\n\n"
            + self.get_reference_examples("repetitive")
            + "\n\n"
            + self.get_reference_examples("multi")
        )

        # Build prompt combinato
        user_prompt = f"""
STYLE REFERENCES (ALL NIOSH HAZARD ASSESSMENT EXAMPLES COMBINED):
{full_fewshot}

NOW GENERATE A NEW HAZARD ASSESSMENT BASED ON THE FOLLOWING FIXED VALUES:

Weight = {data.weight_lbs} lbs
RWL Origin = {data.rwl_origin_lbs}
LI Origin = {data.li_origin}

RWL Destination = {data.rwl_dest_lbs}
LI Destination = {data.li_dest}

Significant control = {data.significant_control}

Task description (context only):
{data.task_description}

IMPORTANT RULES:
- Do NOT modify any numeric value.
- Produce a SINGLE PARAGRAPH.
- Use the FULL manifold style learned from all examples.
- Still respect task_type = "{task_type}".
"""

        ha_combined = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in NIOSH Hazard Assessment. "
                        "Use ONLY provided numbers. NEVER invent values. "
                        "Follow NIOSH Applications Manual style."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )["message"]["content"].strip()

        return {
            "ha_specific": ha_specific,
            "ha_combined": ha_combined,
        }

    def generate_multi_task_dual_hazard_assessment(self, calc_result, task_type: str = "multi"):
        """
        Restituisce due versioni della Hazard Assessment MULTI-TASK:
        - ha_specific: generata con il few-shot MULTI standard
        - ha_combined: generata usando TUTTI i few-shot (single + repetitive + multi)
        """

        # ----------------------------------------------------
        # 1) VERSIONE SPECIFICA (standard multi-task)
        # ----------------------------------------------------
        ha_specific = self.generate_multi_task_hazard_assessment(calc_result)

        # ----------------------------------------------------
        # 2) COSTRUZIONE FEW-SHOT COMBINATO (manifold completo)
        # ----------------------------------------------------
        full_fewshot = (
            "\n\n--- SINGLE-TASK HAZARD EXAMPLE ---\n"
            + self.get_reference_examples("single")
            + "\n\n--- REPETITIVE-TASK HAZARD EXAMPLE ---\n"
            + self.get_reference_examples("repetitive")
            + "\n\n--- MULTI-TASK HAZARD EXAMPLE ---\n"
            + self.get_reference_examples("multi")
        )

        # Costruzione blocchi task con TAG esattamente come richiesto
        tasks_block = []
        for t in calc_result.tasks:
            tasks_block.append(
                f"[TASK:{t.task_id}] "
                f"[Wi:{t.weight_lbs}] "
                f"[H0i:{t.horizontal_origin}] "
                f"[H1i:{t.horizontal_destination}] "
                f"[V0i:{t.vertical_origin}] "
                f"[V1i:{t.vertical_destination}] "
                f"[Ai:{t.asymmetry_angle}] "
                f"[Fi:{t.frequency_lifts_per_min}] "
                f"[RWLi:{t.strwl_lbs}] "
                f"[LIi:{t.stli}] "
                f"[COUPi:{t.coupling}]"
            )

        tasks_block_str = "\n".join(tasks_block)
        duration = getattr(calc_result, "duration", "2-8h")

        # ----------------------------------------------------
        # 3) USER PROMPT PER VERSIONE COMBINATA
        # ----------------------------------------------------
        user_prompt = f"""
    STYLE REFERENCES – FULL MULTI-TASK HAZARD MANIFOLD:
    {full_fewshot}

    NOW GENERATE A NEW MULTI-TASK HAZARD ASSESSMENT.

    RULES:
    - You MUST obey task_type="multi".
    - NEVER modify any numerical value.
    - Use ALL styles (single + repetitive + multi) to shape the narrative.
    - Include ALL TAGS exactly once per referenced value.
    - Interpret CLI according to RNLE.

    JOB DESCRIPTION (context only):
    {calc_result.job_description}

    Composite Lifting Index:
    CLI = {calc_result.cli} [CLI:{calc_result.cli}]
    Risk category: {calc_result.risk_category}
    Duration: [DUR:{duration}]

    TASK DATA (MUST NOT BE ALTERED):
    {tasks_block_str}
    """

        # ----------------------------------------------------
        # 4) SYSTEM PROMPT (coerente con dual single-task)
        # ----------------------------------------------------
        system_prompt = (
            "You are an expert in NIOSH Multi-Task Hazard Assessment. "
            "Use ONLY provided numbers. NEVER invent values. "
            "Follow NIOSH Applications Manual style exactly."
        )

        # ----------------------------------------------------
        # 5) GENERAZIONE VERSIONE COMBINATA – stile identico al dual single-task
        # ----------------------------------------------------
        ha_combined = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )["message"]["content"].strip()

        # ----------------------------------------------------
        # 6) RETURN IDENTICO AL TUO generate_dual_hazard_assessment
        # ----------------------------------------------------
        return {
            "ha_specific": ha_specific,
            "ha_combined": ha_combined,
        }
