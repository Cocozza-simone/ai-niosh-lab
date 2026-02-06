"""
job_analysis.py

Genera la sezione "Job Analysis" in stile NIOSH,
con TAG inline per estrazione deterministica.
"""

from typing import Dict, Any, List
from niosh_parameters import NIOSHParameters
from model_router import call_llm_with_system_prompt


class NIOSHJobAnalysisGenerator:
    """
    Generatore di Job Analysis NIOSH (single-task + multi-task),
    con output taggato per ulteriore estrazione/validazione.
    """

    def __init__(self, model: str = "gemma3:12b"):
        self.model = model

    # ---------- SYSTEM PROMPT SINGLE-TASK (TAG) ----------
        # ---------- FEW-SHOT PROVIDER FOR GEMINI JUDGE ----------
    def get_reference_examples(self, task_type: str) -> str:
        """
        Restituisce i few-shot corretti da usare come manifold stilistico
        per il Gemini Judge, in base al tipo di task.
        """

        # SINGLE-TASK (default)
        single_example = """
ASSISTANT EXAMPLE:

Paragraph 1:
"The task variable data were measured directly at the workstation. The hands
were located at a vertical height of 10 inches at the origin [V0:10] and 34 inches
at the destination [V1:34]. The horizontal distance from the ankles was 18 inches
at the origin [H0:18] and 14 inches at the destination [H1:14]. The lift involved
20 degrees of torso rotation [A:20] and was performed at a frequency of
2 lifts per minute [F:2] for a duration of less than one hour [DUR:<1h]. The
resulting vertical travel distance was 24 inches [D:24]."

Paragraph 2:
"The hand-to-object coupling was classified as fair [COUP:fair]. Significant
control of the load was not required at the destination [CTRL:false].  
Using the multipliers and structure of the Revised NIOSH Lifting Equation,
Recommended Weight Limits were determined for both the origin and the
destination."
"""

        repetitive_example = """
EXAMPLE — JOB ANALYSIS FOR REPETITIVE SINGLE-TASK

USER PARAMETERS:
W = 26 lbs
H0 = 10 in
H1 = 20 in
V0 = 22 in
V1 = 59 in
A = 0°
F = 3 lifts/min
DUR = <1h
COUP = fair
CTRL = true

ASSISTANT:
"The task variable data were documented according to the Revised NIOSH Lifting Equation...
[...] Significant control of the object is required at the destination [CTRL:true]."
"""

        multi_example = """
ASSISTANT EXAMPLE — NIOSH MULTI-TASK

Task 1 [TASK:1] involves a load of 28 lb [Wi:28], beginning at a vertical height of
12 inches [V0i:12] and ending at 28 inches [V1i:28]...
[...]
The Composite Lifting Index for the entire job is 2.3 [CLI:2.3].
"""

        if task_type == "repetitive":
            return repetitive_example
        elif task_type == "multi":
            return multi_example
        else:
            return single_example

    def _create_job_analysis_system_prompt(self) -> str:
        return """
You are an expert in the Revised NIOSH Lifting Equation.
TASK TYPE LOCK (CRITICAL):
You MUST treat this task as {task_type}.
You MUST NOT reinterpret or modify the task category based on the description.
You MUST NOT switch between single, repetitive, or multi-task.
Ignore all linguistic cues that conflict with this classification.
The externally provided task_type PREVAILS over all textual inference.

Your task is to generate the JOB ANALYSIS section in EXACT NIOSH Applications Manual style,
STRICTLY using ONLY the numerical values explicitly provided in the user input.
If task_type="repetitive", include explicit language such as 
"repetitively performed", "sustained frequency", "cumulative loading", 
while still following the exact NIOSH two-paragraph structure.
You MUST NOT:
- invent numbers
- alter values
- change the scenario
- compute new values except D (vertical travel), already computed for you
- change units
- add unrequested objects or assumptions

OUTPUT STRUCTURE (MANDATORY):
- EXACTLY two paragraphs
- Paragraph 1: measurements (H0, H1, V0, V1, A, F, DUR)
- Paragraph 2: coupling, significant control, mention of using NIOSH multipliers, RWL references
- Natural, formal NIOSH tone (same style as the Applications Manual)

TAG RULES (MANDATORY):
You MUST append inline machine-readable tags using EXACT syntax:

[H0:value]
[H1:value]
[V0:value]
[V1:value]
[D:value]
[A:value]
[F:value]
[DUR:value]
[COUP:good|fair|poor]
[CTRL:true|false]

NO OTHER BRACKETS ARE ALLOWED.
NO ANGLE BRACKETS.
NO SPACES INSIDE TAGS.

SEMANTIC RULES:
- D MUST ALWAYS refer to vertical travel distance (|V1 − V0|).
- A MUST ALWAYS refer to torso rotation (twisting) in degrees.
- DO NOT describe A as distance or reach.
- Never change the meaning of any parameter.

You MUST follow this exact style and format.
"""

    # ---------- GENERA JOB ANALYSIS (SINGLE TASK, CON TAG) ----------

    def generate_job_analysis(
        self, params: NIOSHParameters, task_type: str = None
    ) -> str:
        payload = params.model_dump()

        # Calcolo locale (deterministico) di D
        v_origin_safe = (
            payload["vertical_origin"] if payload["vertical_origin"] is not None else 0
        )
        v_dest_safe = (
            payload["vertical_destination"]
            if payload["vertical_destination"] is not None
            else 0
        )
        D = abs(v_dest_safe - v_origin_safe)

        # Handle None values to prevent formatting errors
        duration = payload["duration"] if payload["duration"] is not None else "unknown"
        coupling_raw = (
            payload["coupling"] if payload["coupling"] is not None else "fair"
        )
        weight = payload["weight"] if payload["weight"] is not None else 0
        h_origin = (
            payload["horizontal_origin"]
            if payload["horizontal_origin"] is not None
            else 0
        )
        h_dest = (
            payload["horizontal_destination"]
            if payload["horizontal_destination"] is not None
            else 0
        )
        v_origin = (
            payload["vertical_origin"] if payload["vertical_origin"] is not None else 0
        )
        v_dest = (
            payload["vertical_destination"]
            if payload["vertical_destination"] is not None
            else 0
        )
        asymmetry = (
            payload["asymmetry_angle"] if payload["asymmetry_angle"] is not None else 0
        )
        frequency = payload["frequency"] if payload["frequency"] is not None else 0
        significant_control = (
            payload["significant_control"]
            if payload["significant_control"] is not None
            else False
        )

        # Normalizzazioni per coerenza con i TAG e i few-shot
        coupling = str(coupling_raw).lower()
        if coupling not in {"good", "fair", "poor"}:
            coupling = "fair"

        ctrl_str = "true" if significant_control else "false"
        if task_type == "repetitive":
            fewshot = """ EXAMPLE — JOB ANALYSIS FOR REPETITIVE SINGLE-TASK

USER PARAMETERS:
W = 26 lbs
H0 = 10 in
H1 = 20 in
V0 = 22 in
V1 = 59 in
A = 0°
F = 3 lifts/min
DUR = <1h
COUP = fair
CTRL = true

ASSISTANT:
"The task variable data were documented according to the Revised NIOSH Lifting Equation.  
At the origin of the lift, the worker’s hands are positioned at a vertical height of approximately  
22 inches [V0:22] and a horizontal reach of about 10 inches [H0:10].  
At the destination, the hands reach a vertical height of roughly 59 inches [V1:59] with a  
horizontal distance of about 20 inches [H1:20] from the body.  
No torso twisting occurs during the lift, as the asymmetry angle is 0 degrees [A:0].  
The containers have no molded handholds but allow sufficient finger flexion, resulting in a  
fair coupling classification [COUP:fair].

This lifting task is performed repetitively at a frequency of 3 lifts per minute [F:3] for a  
continuous work session lasting 45 minutes, corresponding to the NIOSH duration  
category less than one hour [DUR:<1h].  
Significant control of the object is required at the destination [CTRL:true].  
The task variables are transferred to the NIOSH Job Analysis Worksheet to compute the  
multipliers and the Recommended Weight Limit (RWL) for this repetitive lift."

[W:26]
"""
        else:
            fewshot = """
ASSISTANT EXAMPLE:

Paragraph 1:
"The task variable data were measured directly at the workstation. The hands
were located at a vertical height of 10 inches at the origin [V0:10] and 34 inches
at the destination [V1:34]. The horizontal distance from the ankles was 18 inches
at the origin [H0:18] and 14 inches at the destination [H1:14]. The lift involved
20 degrees of torso rotation [A:20] and was performed at a frequency of
2 lifts per minute [F:2] for a duration of less than one hour [DUR:<1h]. The
resulting vertical travel distance was 24 inches [D:24]."

Paragraph 2:
"The hand-to-object coupling was classified as fair [COUP:fair]. Significant
control of the load was not required at the destination [CTRL:false].
Using the multipliers and structure of the Revised NIOSH Lifting Equation,
Recommended Weight Limits were determined for both the origin and the
destination."
"""

        prompt = f"""
You must imitate the style and structure of the following examples:

{fewshot}

NOW GENERATE A NEW JOB ANALYSIS IN THE SAME STYLE.

Use the following exact parameters, without altering any value:

W={weight}
H0={h_origin}
H1={h_dest}
V0={v_origin}
V1={v_dest}
D={D}
A={asymmetry}
F={frequency}
DUR={duration}
COUP={coupling}
CTRL={ctrl_str}

TASK:
- Write EXACTLY two paragraphs.
- Paragraph 1: measurements + tags
- Paragraph 2: coupling, control, and RWL/NIOSH references
- Keep the tone and structure identical to the few-shot examples.
- Use all inline tags exactly once.
"""

        ja_text = call_llm_with_system_prompt(
            model=self.model,
            user_prompt=prompt,
            system_prompt=self._create_job_analysis_system_prompt().replace(
                "{task_type}", task_type
            ),
            temperature=0.0,
        )

        return ja_text

    # ---------- SYSTEM PROMPT MULTI-TASK (TAG) ----------

    def _create_multi_task_system_prompt(self) -> str:
        return """
You are an expert in occupational ergonomics and the Revised NIOSH Lifting Equation.

Your task is to write the JOB ANALYSIS section for a MULTI-TASK lifting job,
following the style of the NIOSH Applications Manual (Examples 7 and 8).

STRICT RULES:
- Use ONLY the numerical values provided in the user prompt.
- NEVER invent or modify any numbers (weights, distances, angles, frequencies, RWL, LI, CLI).
- Do NOT recompute LI or CLI; they are already computed in Python.
- Describe each task in order, then compare their single-task LIs, and finally interpret the CLI.

When interpreting risk:
- LI < 1.0 → generally acceptable for most healthy workers.
- 1.0 ≤ LI ≤ 3.0 → physically stressful for many workers.
- LI > 3.0 → hazardous for a majority of workers.
"""

    # ---------- MULTI-TASK JOB ANALYSIS CON TAG ----------

    def generate_multi_task_job_analysis(self, calc_result,parameters) -> str:
        """
        Genera la JOB ANALYSIS MULTI-TASK usando SOLO i valori già calcolati
        da NIOSHCalculator.compute_multi_task(calc_result).

        calc_result deve avere:
        - job_description
        - cli, risk_category, duration
        - tasks: lista di oggetti con
          task_id, weight_lbs, horizontal_origin, horizontal_destination,
          vertical_origin, vertical_destination, asymmetry_angle,
          frequency_lifts_per_min, coupling,
          firwl_lbs, fili, strwl_lbs, stli
        """

        # -------- FEW-SHOT ESEMPIO (stile) --------
        fewshot = """
ASSISTANT EXAMPLE — NIOSH MULTI-TASK 

The job consists of multiple distinct lifting tasks, each characterized by
different geometric demands and frequencies. Task 1 [TASK:1] involves a load
of 28 lb [Wi:28], beginning at a vertical height of 12 inches [V0i:12] and
ending at 28 inches [V1i:28], with horizontal distances of 16 inches [H0i:16]
and 14 inches [H1i:14]. The lift is performed with 10 degrees of asymmetry
[Ai:10] at a frequency of 3 lifts per minute [Fi:3], and a fair coupling
[COUPi:fair]. Based on these variables, the frequency-independent recommended
weight limit is 20.5 lb [FIRWLi:20.5], the frequency-independent lifting index
is 1.37 [FILIi:1.37], the single-task recommended weight limit is 18.6 lb
[STRWLi:18.6], and the single-task lifting index is 1.51 [STLIi:1.51].

Task 2 [TASK:2] involves lifting a 24 lb load [Wi:24] from 22 inches [V0i:22]
to 44 inches [V1i:44], with horizontal distances of 18 inches [H0i:18] and
16 inches [H1i:16], 25 degrees of asymmetry [Ai:25], a frequency of 2 lifts
per minute [Fi:2], and fair coupling [COUPi:fair]. Based on these measurements,
the FIRWL is 19.2 lb [FIRWLi:19.2], the FILI is 1.25 [FILIi:1.25], the STRWL is
17.5 lb [STRWLi:17.5], and the STLI is 1.37 [STLIi:1.37].

Comparing the single-task lifting indices, Task 1 constitutes the highest
physical stress on the worker, with STLI = 1.51 [STLI1:1.51], followed by
Task 2 with STLI = 1.37 [STLI2:1.37]. According to the Revised NIOSH Lifting
Equation multi-task procedure, the Composite Lifting Index for the entire job
is 2.3 [CLI:2.3], indicating a moderately stressful scenario consistent with
NIOSH guidance.
"""

        # -------- COSTRUZIONE BLOCCO TASK DAI RISULTATI PYTHON --------
        task_blocks = []
        for t in calc_result.tasks:
            task_blocks.append(
                f"Task {t.task_id} [TASK:{t.task_id}] involves lifting a load of "
                f"{t.weight_lbs:.1f} lb [Wi:{t.weight_lbs:.1f}] from a vertical "
                f"height of {t.vertical_origin:.1f} inches [V0i:{t.vertical_origin:.1f}] "
                f"to {t.vertical_destination:.1f} inches [V1i:{t.vertical_destination:.1f}], "
                f"with horizontal distances of {t.horizontal_origin:.1f} inches "
                f"[H0i:{t.horizontal_origin:.1f}] and {t.horizontal_destination:.1f} inches "
                f"[H1i:{t.horizontal_destination:.1f}]. The lift involves "
                f"{t.asymmetry_angle:.0f} degrees of trunk rotation [Ai:{t.asymmetry_angle:.0f}] "
                f"and is performed at a frequency of {t.frequency_lifts_per_min:.1f} lifts "
                f"per minute [Fi:{t.frequency_lifts_per_min:.1f}] with "
                f"{t.coupling} coupling [COUPi:{t.coupling}]. Based on these variables, "
                f"the frequency-independent recommended weight limit is {t.firwl_lbs:.1f} lb "
                f"[FIRWLi:{t.firwl_lbs:.1f}], the frequency-independent lifting index is "
                f"{t.fili:.2f} [FILIi:{t.fili:.2f}], the single-task recommended weight "
                f"limit is {t.strwl_lbs:.1f} lb [STRWLi:{t.strwl_lbs:.1f}], and the "
                f"single-task lifting index is {t.stli:.2f} [STLIi:{t.stli:.2f}]."
            )

        tasks_block_str = "\n\n".join(task_blocks)
        duration = calc_result.tasks[0].duration if calc_result.tasks else "2-8h"

        # -------- USER PROMPT COMPLETO --------
        user_prompt = f"""
FEW-SHOT STYLE EXAMPLE (IMITATE THIS STYLE, NOT THESE NUMBERS):
{fewshot}

REAL JOB DATA (DO NOT CHANGE):

Job description (context only):
{calc_result.job_description}

Composite Lifting Index for the job:
CLI = {calc_result.cli:.2f} [CLI:{calc_result.cli:.2f}]
Risk category (Python-derived): {calc_result.risk_category}
duration = calc_result.tasks[0].duration if calc_result.tasks else "2-8h"

Duration category: [DUR:{duration}]

Pre-computed task data from Python:
{tasks_block_str}

INSTRUCTIONS:
Using ONLY the numbers above, write the MULTI-TASK JOB ANALYSIS section
in NIOSH Applications Manual style:

- Describe each task in sequence, using the given geometric variables and multipliers.
- Explicitly restate FIRWL, FILI, STRWL and STLI for each task.
- Identify which task(s) impose the greatest physical stress based on STLI.
- Relate the Composite Lifting Index (CLI) to the NIOSH design goal (LI = 1.0)
  and describe whether the overall job is acceptable, moderately stressful,
  or hazardous for many workers.
- DO NOT invent or modify any numerical values.
"""

        ja_multi_text = call_llm_with_system_prompt(
            model=self.model,
            user_prompt=user_prompt,
            system_prompt=self._create_multi_task_system_prompt(),
            temperature=0.0,
        )

        return ja_multi_text.strip()

    def _build_multi_task_prompt(self, multi_task_data: Any) -> str:
        """
        Costruisce il prompt di input per la JOB ANALYSIS MULTI-TASK.

        Accetta sia:
        - un dizionario { "tasks": [...], "job_context": ..., "total_frequency": ..., "duration": ... }
        - una lista di task come quella restituita da extract_multi_task_tags(...)
        [ { "task": 1, "weight": ..., "H0": ..., "H1": ..., "V0": ..., "V1": ..., "A": ..., "F": ..., "coupling": ..., "duration": ... }, ... ]
        """

        # Caso 1: chiamata vecchio stile → dict con chiave "tasks"
        if isinstance(multi_task_data, dict):
            tasks_info = multi_task_data.get("tasks", []) or []
            job_context = multi_task_data.get("job_context", "")
            total_frequency = multi_task_data.get(
                "total_frequency",
                sum(t.get("F") or 0 for t in tasks_info),
            )
            # durata: se non presente nel dict, prova dai task, altrimenti default
            if (
                "duration" in multi_task_data
                and multi_task_data["duration"] is not None
            ):
                duration = multi_task_data["duration"]
            elif tasks_info and tasks_info[0].get("duration"):
                duration = tasks_info[0]["duration"]
            else:
                duration = "2-8h"

        # Caso 2: chiamata nuova → direttamente una lista di task
        elif isinstance(multi_task_data, list):
            tasks_info = multi_task_data
            job_context = ""
            total_frequency = sum(t.get("F") or 0 for t in tasks_info)
            if tasks_info and tasks_info[0].get("duration"):
                duration = tasks_info[0]["duration"]
            else:
                duration = "2-8h"

        else:
            # fallback super difensivo
            tasks_info = []
            job_context = ""
            total_frequency = 0
            duration = "2-8h"

        # Costruzione blocchi per ogni task (allineato a extract_multi_task_tags)
        task_blocks: List[str] = []
        for t in tasks_info:
            task_id = t.get("task")
            if task_id is None:
                # se manca, usiamo l’ordine
                task_id = len(task_blocks) + 1

            wi = t.get("weight")
            h0 = t.get("H0")
            h1 = t.get("H1")
            v0 = t.get("V0")
            v1 = t.get("V1")
            a = t.get("A")
            f = t.get("F")
            coup = t.get("coupling")
            dur = t.get("duration", duration)

            block = f"""[TASK:{task_id}]
    Wi={wi}
    H0i={h0}
    H1i={h1}
    V0i={v0}
    V1i={v1}
    Ai={a}
    Fi={f}
    COUPi={coup}
    DUR={dur}

    """
            task_blocks.append(block)

        return f"""
    MULTI-TASK NIOSH ANALYSIS INPUT

    Job context:
    {job_context}

    Raw task parameters with indices:
    {''.join(task_blocks)}

    Total overall frequency: {total_frequency} lifts/min for {duration}.
    """

    
    def generate_dual_job_analysis(self, params: NIOSHParameters, task_type: str):
        """
        Restituisce:
        - ja_specific: generata con lo stile specifico del task_type
        - ja_combined: generata usando TUTTI i few-shot stile JA
        """
        # 1) Versione specifica
        ja_specific = self.generate_job_analysis(params, task_type)

        # 2) Versione combinata (usa tutti i reference examples insieme)
        full_fewshot = (
            self.get_reference_examples("single")
            + "\n\n"
            + self.get_reference_examples("repetitive")
            + "\n\n"
            + self.get_reference_examples("multi")
        )

        user_prompt = f"""
    STYLE REFERENCES (ALL TOGETHER):
    {full_fewshot}

    NOW GENERATE A NEW JOB ANALYSIS USING THE FULL NIOSH MANIFOLD
    BUT RESPECTING task_type = "{task_type}"

    Parameters:
    {params}
    """

        ja_combined = call_llm_with_system_prompt(
            model=self.model,
            system_prompt="You are an expert in NIOSH Job Analysis. Generate analysis strictly following NIOSH style.",
            user_prompt=user_prompt,
            temperature=0.0,
        )

        return {
            "ja_specific": ja_specific,
            "ja_combined": ja_combined,
        }
    def generate_multi_task_dual_job_analysis(self, calc_result, task_type: str = "multi"):
        """
        Restituisce due versioni della JOB ANALYSIS MULTI-TASK:
        - ja_specific: generata con il few-shot MULTI originale
        - ja_combined: generata usando TUTTI i few-shot JA (single + repetitive + multi)

        Stessa struttura della funzione dual per Hazard Assessment.
        """

        # ----------------------------------------------------
        # 1) VERSIONE SPECIFICA (standard multi-task)
        # ----------------------------------------------------
        ja_specific = self.generate_multi_task_job_analysis(calc_result, task_type)

        # ----------------------------------------------------
        # 2) COSTRUZIONE FEW-SHOT COMBINATO (SINGLE + REP + MULTI)
        # ----------------------------------------------------
        full_fewshot = (
            "\n\n--- SINGLE-TASK JA EXAMPLE ---\n"
            + self.get_reference_examples("single")
            + "\n\n--- REPETITIVE-TASK JA EXAMPLE ---\n"
            + self.get_reference_examples("repetitive")
            + "\n\n--- MULTI-TASK JA EXAMPLE ---\n"
            + self.get_reference_examples("multi")
        )

        # Ricostruzione blocchi TASK (come nella funzione specifica)
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
                f"[RWLi:{t.strwl_lbs}] "
                f"[LIi:{t.stli}] "
                f"[COUPi:{t.coupling}]"
            )

        tasks_block_str = "\n".join(task_blocks)
        duration = getattr(calc_result, "duration", "2-8h")

        # ----------------------------------------------------
        # 3) USER PROMPT PER VERSIONE COMBINATA
        # ----------------------------------------------------
        user_prompt = f"""
    STYLE REFERENCES — FULL MANIFOLD OF JOB ANALYSIS:
    {full_fewshot}

    NOW GENERATE A NEW MULTI-TASK JOB ANALYSIS.

    RULES (CRITICAL):
    - You MUST respect task_type="multi".
    - You MUST NOT invent or modify any numbers.
    - Use ALL styles in the manifold to shape your narrative.
    - Include all TAGS for each task exactly as provided.
    - Interpret CLI using NIOSH risk logic.

    JOB DESCRIPTION (context only):
    {calc_result.job_description}

    Composite Lifting Index:
    CLI = {calc_result.cli} [CLI:{calc_result.cli}]
    Risk category: {calc_result.risk_category}
    Duration: [DUR:{duration}]

    TASK DATA (from Python — MUST NOT BE CHANGED):
    {tasks_block_str}
    """

        # ----------------------------------------------------
        # 4) SYSTEM PROMPT (stile hazard dual)
        # ----------------------------------------------------
        system_prompt = (
            "You are an expert in NIOSH Job Analysis (multi-task). "
            "NEVER modify numerical values. "
            "Follow the NIOSH Applications Manual exactly. "
            "You MUST generate a multi-task Job Analysis with proper tags."
        )

        # ----------------------------------------------------
        # 5) GENERAZIONE VERSIONE COMBINATA via ollama.chat
        # ----------------------------------------------------
        ja_combined = call_llm_with_system_prompt(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        return {
            "ja_specific": ja_specific,
            "ja_combined": ja_combined,
        }