"""
redesign_suggestions.py

Genera la sezione "Redesign Suggestions" in stile NIOSH (Applications Manual)
a partire da:
- geometria del compito (H, V, A, F, durata, coupling, controllo),
- risultati NIOSH (RWL, LI, moltiplicatori, categoria di rischio).

Usa Structured Outputs per garantire che le proposte siano logicamente coerenti.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import ollama
from pydantic import BaseModel, Field
from niosh_calculator import NIOSHCalculator
from niosh_parameters import NIOSHParameters

# Multi-agent coordinator import removed - standard mode only
# ML system import removed - standard mode only


# ======================= DATA CLASS DI INPUT =======================


@dataclass
class RedesignSuggestionsInput:
    """
    Contiene TUTTE le info necessarie per scrivere le Redesign Suggestions.

    Nessun valore numerico deve essere modificato dal modello.
    """

    task_description: str  # job description generata (contesto)

    # Geometria / task variables
    h_origin: float
    h_dest: float
    v_origin: float
    v_dest: float
    a_origin: float
    a_dest: float
    frequency_lifts_per_min: float
    duration_class: str  # "<1h", "1-2h", "2-8h", ">8h"
    coupling: str  # "good", "fair", "poor"
    significant_control: bool  # True/False

    # Carico e risultati NIOSH
    weight_lbs: float
    rwl_origin_lbs: float
    rwl_dest_lbs: Optional[float]
    li_origin: float
    li_dest: Optional[float]

    # Moltiplicatori NIOSH
    multipliers_origin: Dict[str, float]
    multipliers_destination: Optional[Dict[str, float]]

    # Classificazione di rischio derivata (da NIOSHCalculationResult.compute_risk)
    risk_category: Optional[str] = None  # "acceptable", "slightly_stressful", ...
    risk_comment: Optional[str] = None  # frase corta tipo "Hazardous for ..."


# ======================= STRUCTURED OUTPUT MODEL =======================


class RedesignOutput(BaseModel):
    """Struttura per l'output delle suggestion"""

    smallest_multipliers: List[str] = Field(
        ...,
        description="List of multipliers with the smallest values (e.g. ['HM', 'VM'])",
    )
    proposed_changes: List[str] = Field(
        ...,
        description="List of specific proposed changes (e.g. 'Reduce horizontal distance')",
    )
    narrative_text: str = Field(
        ...,
        description="The final redesign suggestions section in plain English prose, following NIOSH style.",
    )


# ==================== GENERATORE DI REDESIGN SUGGESTIONS ====================


class NIOSHRedesignSuggestionsGenerator:
    """
    Standard NIOSH redesign suggestions generator using traditional analysis methods.
    Generates professional redesign suggestions following NIOSH Applications Manual style.
    """

    def __init__(self, model: str = "gemma3:12b"):
        self.model = model

    # ---------- UTILITY PER FORMATTARE I MOLTIPLICATORI CRITICI ----------

    def _create_improved_scenario(
        self, data: RedesignSuggestionsInput
    ) -> Optional[Dict]:
        """
        Crea uno scenario di redesign migliorato calcolando i nuovi RWL/LI.
        Identifica i moltiplicatori più bassi e propone modifiche realistiche.
        """
        try:
            # Parametri originali
            original_params = NIOSHParameters(
                weight=data.weight_lbs,
                horizontal_origin=data.h_origin,
                horizontal_destination=data.h_dest,
                vertical_origin=data.v_origin,
                vertical_destination=data.v_dest,
                asymmetry_angle=data.a_origin,
                frequency=data.frequency_lifts_per_min,
                duration=data.duration_class,
                coupling=data.coupling,
                significant_control=data.significant_control,
                gender="M",  # Default
                age=25,  # Default
                judgment="intermediate",  # Default
                one_limb_lifting=False,
                two_operators_lifting=False,
            )

            # Identifica i moltiplicatori più problematici
            smallest_multipliers = sorted(
                data.multipliers_origin.items(), key=lambda x: x[1]
            )[:3]

            # Applica miglioramenti basati sui moltiplicatori più bassi
            improved_params = NIOSHParameters(
                weight=data.weight_lbs,
                horizontal_origin=data.h_origin,
                horizontal_destination=data.h_dest,
                vertical_origin=data.v_origin,
                vertical_destination=data.v_dest,
                asymmetry_angle=data.a_origin,
                frequency=data.frequency_lifts_per_min,
                duration=data.duration_class,
                coupling=data.coupling,
                significant_control=data.significant_control,
                gender="M",
                age=25,
                judgment="intermediate",
                one_limb_lifting=False,
                two_operators_lifting=False,
            )

            improvements_made = []

            # Migliora HM (horizontal distance) - il più comune
            if any("HM" in k for k, v in smallest_multipliers if v < 0.94):
                # Riduci H del 25% ma calcola la percentuale reale per il testo
                new_h_origin = max(10, data.h_origin * 0.75)
                new_h_dest = max(10, data.h_dest * 0.75)
                improved_params.horizontal_origin = new_h_origin
                improved_params.horizontal_destination = new_h_dest
                # Calcola la percentuale effettiva di riduzione
                actual_reduction = (
                    (data.h_origin - new_h_origin) / data.h_origin
                ) * 100
                improvements_made.append(
                    f"reduced horizontal reach by {actual_reduction:.0f}%"
                )

            # Migliora FM (frequency)
            if any("FM" in k for k, v in smallest_multipliers if v < 0.94):
                # Riduci frequenza del 30%
                improved_params.frequency = max(0.5, data.frequency_lifts_per_min * 0.7)
                improvements_made.append("reduced lifting frequency by 30%")

            # Migliora VM (vertical location)
            if any("VM" in k for k, v in smallest_multipliers if v < 0.94):
                # Sposta V verso l'ottimale (30 pollici)
                if data.v_origin < 25:
                    improved_params.vertical_origin = min(30, data.v_origin + 5)
                elif data.v_origin > 35:
                    improved_params.vertical_origin = max(30, data.v_origin - 5)
                improvements_made.append(
                    "adjusted vertical height toward optimal range"
                )

            # Migliora AM (asymmetry)
            if any(
                "AM" in k
                for k, v in smallest_multipliers
                if v < 0.94 and data.a_origin > 15
            ):
                # Riduci asimmetria
                improved_params.asymmetry_angle = min(15, data.a_origin * 0.5)
                improvements_made.append("reduced asymmetry angle by 50%")

            # Migliora CM (coupling) - solo se non è già ottimale
            current_cm = data.multipliers_origin.get("CM", 1.0)
            if (
                any("CM" in k for k, v in smallest_multipliers if v < 1.0)
                and current_cm < 0.99  # Solo se non è già ~1.000
                and data.coupling != "good"
            ):
                improved_params.coupling = "good"
                improvements_made.append("improved hand-to-object coupling to 'good'")
            elif current_cm >= 0.99:
                # Il coupling è già ottimale, non suggerire miglioramenti
                pass

            # Calcola nuovi risultati NIOSH
            calculator = NIOSHCalculator()
            improved_result = calculator.compute(improved_params)

            return {
                "improvements": improvements_made,
                "new_rwl": improved_result.rwl_origin_lbs,
                "new_li": improved_result.li_origin,
                "improvement_percentage": (
                    (data.li_origin - improved_result.li_origin) / data.li_origin
                )
                * 100,
            }

        except Exception as e:
            print(f"Errore creazione scenario migliorato: {e}")
            return None

    def _format_smallest_multipliers(
        self, multipliers: Optional[Dict[str, float]]
    ) -> str:
        """
        Ritorna una stringa che evidenzia i moltiplicatori con valore più basso
        (cioè le penalizzazioni maggiori) in modo simile al manuale NIOSH.
        Esclude VM=1.000 in quanto rappresenta il valore ottimale, non una penalizzazione.
        """
        if not multipliers:
            return "not available"

        # Filtra i moltiplicatori per escludere VM=1.000 (valore ottimale)
        filtered_multipliers = {
            k: v
            for k, v in multipliers.items()
            if not (k == "VM" and abs(v - 1.0) < 0.001)
        }

        if not filtered_multipliers:
            return "all multipliers are optimal"

        # Trova il valore minimo tra i moltiplicatori filtrati
        min_value = min(filtered_multipliers.values())

        # Identifica i moltiplicatori più penalizzanti (sotto 0.94)
        penalizing_multipliers = [
            k for k, v in filtered_multipliers.items() if v < 0.94
        ]

        if penalizing_multipliers:
            # Usa i moltiplicatori penalizzanti come prioritari
            keys_str = ", ".join(sorted(penalizing_multipliers))
            values_str = ", ".join(
                [f"{filtered_multipliers[k]:.3f}" for k in penalizing_multipliers]
            )
            return f"{keys_str} = {values_str}"
        else:
            # Se non ci sono moltiplicatori penalizzanti, usa quelli con valori più bassi
            smallest = [
                k for k, v in filtered_multipliers.items() if abs(v - min_value) < 1e-6
            ]
            keys_str = ", ".join(sorted(smallest))
            return f"{keys_str} = {min_value:.3f}"

    # ---------- PROMPT DI SISTEMA ----------

        # ---------- PROMPT DI SISTEMA ----------

    def _create_system_prompt(self, task_type: str | None = None) -> str:
        task_lock = f"""
TASK TYPE LOCK (CRITICAL):
- You MUST treat this job as a '{task_type}' lifting task.
- You MUST NOT reinterpret or modify the task category (single vs repetitive).
- You MUST ignore any linguistic cues in the description that conflict with this externally provided task_type.
- The task_type comes from an external NIOSH pipeline and PREVAILS over any inference.
"""

        repetitive_block = ""
        if task_type == "repetitive":
            repetitive_block = """
If task_type="repetitive", you MUST explicitly prioritize redesign ideas focused on:
- reducing lifting frequency,
- introducing micro-pauses and rest breaks,
- alternating tasks to reduce cumulative loading,
- reducing exposure duration,
- implementing pacing strategies,
- using mechanical aids or partial automation when appropriate.

You MUST explicitly mention the cumulative physical demand characteristic of repetitive lifting,
and the fact that the Frequency Multiplier (FM) is often the primary limiting factor in such tasks.
Never modify or invent FM values; discuss ONLY conceptually.
"""

        return f"""
You are an expert in occupational ergonomics and the Revised NIOSH Lifting Equation.
You write the REDESIGN SUGGESTIONS section in the formal style of the NIOSH Applications Manual.

{task_lock}
{repetitive_block}

===========================================================
MANDATORY RULES
===========================================================
1. NEVER CHANGE NUMERIC VALUES.
2. Identify the smallest multipliers and use them as redesign targets.
3. Link redesign ideas ONLY to correct multipliers (HM, VM, DM, AM, FM, CM).
4. Use 1–2 technical paragraphs (no lists, no bullets).
5. Keep task context exactly as given.
6. Use clear NIOSH-style causal language:
   - Reducing horizontal reach increases HM.
   - Adjusting V toward ~30 inches increases VM.
   - Reducing asymmetry increases AM.
   - Improving coupling increases CM.
   - Reducing frequency/duration increases FM.
7. NEVER recompute formulas. NEVER invent new multipliers.
"""


    # ---------- PROMPT UTENTE (DATI) ----------
    def _create_system_prompt_multi_task(self) -> str:
        return """
You are an expert in occupational ergonomics and the Revised NIOSH Lifting Equation.
Your task is to write the MULTI-TASK REDESIGN SUGGESTIONS section, following EXACTLY
the tone and analytic style of NIOSH Applications Manual Examples 7 and 8.

You MUST strictly follow these rules:

======================================================================
CORE PRINCIPLES (DO NOT VIOLATE)
======================================================================

1. NUMBER USAGE
- Use ONLY the LIi, RWLi, and CLI values provided in the user prompt.
- NEVER recompute LIi, RWLi, or CLI.
- NEVER estimate new multipliers.
- NEVER invent new values (H, V, A, F, CM).
- You MUST restate numbers exactly as provided.

2. MULTI-TASK FOCUS
The redesign must follow the multi-task logic of NIOSH Examples 7 and 8:
- Identify which tasks have the highest single-task LI (LIi).
- Identify which tasks contribute most to the Composite Lifting Index (CLI).
- Explain how redesign priorities should focus on tasks with the greatest penalties.
- You MUST mention tasks by their IDs using the provided tags: [TASK:i], [LIi:x], [RWLi:x].

3. TAG RULES
Inline tags MUST appear in natural language as provided in the prompt:
- [TASK:i]
- [Wi:x]
- [RWLi:x]
- [LIi:x]
- [Fi:x]
- [CLI:x]
- [COUPi:good|fair|poor]
No other tags allowed.

4. NIOSH STYLE REQUIREMENTS
Your redesign suggestions MUST:
- Begin with a short introduction summarizing the multi-task penalties.
- Clearly identify tasks with the highest LI values:  
  (“Task 3 [LI3:1.7] and Task 4 [LI4:2.4] represent the major penalties…”)
- Explain that redesign efforts should focus on geometric variables that affect:
  - HM (horizontal distance)
  - VM (vertical location)
  - DM (vertical travel distance)
  - AM (asymmetry)
  - FM (frequency)
  - CM (coupling)
- Interpret the CLI relative to ≤1, 1–3, or ≥3:
  • CLI < 1 → acceptable  
  • CLI 1–3 → moderately / physically stressful  
  • CLI > 3 → hazardous  

5. REDESIGN LOGIC
You MUST:
- Target reductions in horizontal reach (improves HM).
- Target adjustments toward 30 inches for VM (if applicable).
- Target reduced asymmetry (improves AM).
- Target improved coupling (raises CM).
- Target reduced frequency when FM is small.
- For multi-task jobs, frame redesign as:
  “Because the CLI is driven primarily by the high LI values in Task i, redesign should prioritize…”

6. PROHIBITIONS
ABSOLUTELY FORBIDDEN:
- Recomputing LIi or CLI.
- Suggesting formulae such as 51 × HM × …
- Introducing new equipment unless explicitly allowed.
- Changing the scenario (weight, task type, load type).
- Numerical rounding inconsistencies.

======================================================================
FEW-SHOT TRAINING (NIOSH MULTI-TASK EXAMPLES)
======================================================================

EXAMPLE (based on NIOSH Example 8 — handling cans):
"The multi-task job includes subtasks with LI values of 1.6, 1.5, 1.7, and 2.4
([LI1:1.6] [LI2:1.5] [LI3:1.7] [LI4:2.4]). The highest LI occurs in Task 4, which
dominates the Composite Lifting Index of 2.9 [CLI:2.9]. Because this CLI is close
to 3.0, redesign should focus on reducing horizontal reach, improving coupling,
and decreasing lifting frequency, particularly for the more stressful tasks."

EXAMPLE (based on NIOSH Example 7 — depalletizing):
"Although no single task exceeds an LI of 1.0, the combined CLI of 1.4 [CLI:1.4]
indicates that cumulative exposure may be moderately stressful. Redesign should
prioritize tasks with lower HM and CM values and reduce vertical travel distances."

======================================================================
OUTPUT FORMAT
======================================================================

You MUST output:
- A short, technical paragraph (no bullets).
- Identification of tasks with highest LI.
- Clear redesign strategies tied to NIOSH multipliers.
- Use tags inline.
"""

    def _build_user_prompt(
        self,
        data: RedesignSuggestionsInput,
        improved_scenario: Optional[Dict] = None,
        task_type: str = None,
    ) -> str:

        control_text = "yes" if data.significant_control else "no"

        # 🔥 Se il task è ripetitivo, aggiungi un contesto speciale
        if task_type == "repetitive":
            repetitive_context = (
                "\nNOTE: This is a REPETITIVE lifting task performed at sustained frequency. "
                "The Hazard Assessment MUST explicitly reference cumulative physical loading, "
                "repetition-driven fatigue, and the limiting effect of the Frequency Multiplier (FM). "
                "Redesign suggestions should prioritize: reducing lifting frequency, pacing strategies, "
                "micro-pauses, alternating tasks to reduce cumulative exposure, and reducing duration.\n"
            )
        else:
            repetitive_context = ""

        smallest_origin = self._format_smallest_multipliers(data.multipliers_origin)
        smallest_dest = (
            self._format_smallest_multipliers(data.multipliers_destination)
            if data.multipliers_destination is not None
            else "not computed (RWL at origin only)"
        )

        rwl_dest_str = (
            f"{data.rwl_dest_lbs:.1f} lbs"
            if data.rwl_dest_lbs is not None
            else "not computed"
        )

        risk_line = ""
        if data.risk_category or data.risk_comment:
            risk_line = (
                f"- Overall risk classification: "
                f"{data.risk_category or 'n/a'}; {data.risk_comment or ''}\n"
            )

        li_dest_str = "not computed"
        if data.li_dest is not None:
            li_dest_str = f"{data.li_dest:.2f}"

        return f"""
TASK TYPE (LOCKED): {task_type}
This classification is externally provided by the NIOSH pipeline and MUST NOT be changed by the model.

You are given the following lifting task scenario (job description), which has already been
analyzed and MUST NOT be changed:

TASK SCENARIO (for context only, do NOT change it):
{data.task_description}

{repetitive_context}

    From this scenario and from the NIOSH calculation, the following data have been obtained
    (ALL VALUES ARE FINAL AND MUST NOT BE CHANGED):

    ACTUAL MULTIPLIER VALUES (use these exact numbers, do NOT invent or round them):
    - HM (Horizontal Multiplier): {data.multipliers_origin.get('HM', 'N/A')}
    - VM (Vertical Multiplier): {data.multipliers_origin.get('VM', 'N/A')}
    - DM (Distance Multiplier): {data.multipliers_origin.get('DM', 'N/A')}
    - AM (Asymmetry Multiplier): {data.multipliers_origin.get('AM', 'N/A')}
    - FM (Frequency Multiplier): {data.multipliers_origin.get('FM', 'N/A')}
    - CM (Coupling Multiplier): {data.multipliers_origin.get('CM', 'N/A')}

    GEOMETRY AND TASK VARIABLES:
    - Load weight: {data.weight_lbs:.1f} lbs
    - Vertical location at the origin, V_origin: {data.v_origin:.1f} inches
    - Vertical location at the destination, V_dest: {data.v_dest:.1f} inches
    - Horizontal location at the origin, H_origin: {data.h_origin:.1f} inches
    - Horizontal location at the destination, H_dest: {data.h_dest:.1f} inches
    - Asymmetry angle at the origin, A_origin: {data.a_origin:.1f} degrees
    - Asymmetry angle at the destination, A_dest: {data.a_dest:.1f} degrees
    - Lifting frequency: {data.frequency_lifts_per_min:.2f} lifts/min
    - Duration category: {data.duration_class}
    - Coupling classification: {data.coupling}
    - Significant control of the object required at destination: {control_text}

    NIOSH RESULTS:
    - Recommended Weight Limit (RWL) at the origin: {data.rwl_origin_lbs:.1f} lbs
    - Recommended Weight Limit (RWL) at the destination: {rwl_dest_str}
    - Lifting Index (LI) at the origin: {data.li_origin:.2f}
    - Lifting Index (LI) at the destination: {li_dest_str}

    MULTIPLIERS AT THE ORIGIN:
    - {data.multipliers_origin}

    MULTIPLIERS AT THE DESTINATION:
    - {data.multipliers_destination}

    SUMMARY OF SMALLEST MULTIPLIERS:
    - At the origin, the smallest multipliers (largest penalties) are: {smallest_origin}
    - At the destination, the smallest multipliers (largest penalties) are: {smallest_dest}

    {risk_line}

    NUMERICAL REDESIGN PROJECTION (integrate these specific numerical improvements into your main text):
    {self._format_numerical_projection(improved_scenario) if improved_scenario else "No quantitative projection available."}

    Using ONLY the information above, write the REDESIGN SUGGESTIONS section in the style of
    the NIOSH Applications Manual.
    """

    # ---------- CHIAMATA A OLLAMA ----------

        # ---------- CHIAMATA A OLLAMA ----------

    def generate_redesign_suggestions(
        self,
        data: RedesignSuggestionsInput,
        task_type: str | None = None,
    ) -> str:
        """
        Genera il testo di Redesign Suggestions includendo:
        - scenario migliorato numerico (calcolato in Python),
        - TASK TYPE LOCK (single / repetitive) passato dall'esterno.
        Nessun valore numerico viene modificato dal modello.
        """

        # Genera scenario migliorato (tutto deterministico in Python)
        improved_scenario = self._create_improved_scenario(data)

        # Prompt utente con contesto + numerica + info ripetitività
        prompt = self._build_user_prompt(data, improved_scenario, task_type)

        # Blocco di requisiti tecnici NIOSH (come avevi già)
        enhanced_prompt = (
            prompt
            + """

NIOSH TECHNICAL ACCURACY REQUIREMENTS:
- Coupling improvements INCREASE CM toward 1.0, improving RWL and DECREASING LI.
- Never say that coupling improvements "decrease the coupling multiplier".
- Moving heights toward 30 inches INCREASES VM (better), not decreases it.
- Reducing horizontal reach INCREASES HM (better), never worsens conditions.
- All improvements increase multipliers toward 1.0, reducing risk.
- Refer only conceptually to NIOSH multiplier tables: VM, CM, HM (no new numbers).

Return your response as JSON following the RedesignOutput schema exactly.
Use temperature 0.0 for deterministic output following Ollama best practices.
Ensure complete technical accuracy following NIOSH manual guidelines.
"""
        )

        # System prompt con TASK TYPE LOCK
        system_prompt = self._create_system_prompt(task_type)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enhanced_prompt},
                ],
                format=RedesignOutput.model_json_schema(),
                options={"temperature": 0.0},
            )

            structured_response = RedesignOutput.model_validate_json(
                response.message.content
            )
            base_suggestions = structured_response.narrative_text.strip()

            # Aggiungi appendice numerica se disponibile (calcolata in Python, NON dal modello)
            if improved_scenario:
                numerical_appendix = self._create_numerical_appendix(
                    data, improved_scenario
                )
                return base_suggestions + "\n\n" + numerical_appendix
            else:
                return base_suggestions

        except Exception as e:
            return f"Errore generazione redesign suggestions: {e}"

    def generate_multi_task_redesign(self, calc_result) -> str:
        """
        Genera REDESIGN SUGGESTIONS per job MULTI-TASK.
        Usa SOLO LIi, RWLi, CLI già calcolati da Python.
        Nessuna ricomputazione. Nessuna modifica numerica.
        Produce una narrativa tecnica nello stile NIOSH Examples 7–8.
        """

        # ---------------------------------------------------------
        # COSTRUZIONE BLOCCO TASK (TAG COERENTI CON TUTTO IL SISTEMA)
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

        # Durata sempre fornita dal calcolo Python
        duration = getattr(calc_result, "duration", "2-8h")

        # ---------------------------------------------------------
        # FEW-SHOT MULTI-TASK REDESIGN — versione corretta
        # ---------------------------------------------------------
        fewshot = """
    ASSISTANT EXAMPLE — MULTI-TASK REDESIGN (STYLE TO IMITATE)

    "In this multi-task job, the highest single-task lifting indices occur in 
    Task 3 [LI3:1.7] and Task 4 [LI4:2.4], which represent the greatest contributors 
    to the Composite Lifting Index of 2.9 [CLI:2.9]. Because CLI exceeds 2.0 and 
    approaches 3.0, the job would be considered physically stressful for a 
    substantial portion of workers over its duration [DUR:<1h].

    Redesign priorities should therefore focus on the geometric and task factors 
    that most strongly influence these elevated LI values. For the higher-stress 
    tasks, reducing horizontal reach (improving HM), improving hand-to-object 
    coupling (increasing CM), reducing torso rotation (increasing AM), and adjusting 
    vertical locations toward the optimal region near 30 inches (improving VM) would 
    provide the most meaningful reductions in physical stress. Where feasible, 
    lowering lifting frequency for the more demanding tasks would also improve the 
    Frequency Multiplier (FM) and reduce cumulative loading.

    Because Task 4 exhibits the highest LI [LI4:2.4], redesign efforts should 
    concentrate on improving HM, CM and AM for this task first, followed by 
    incremental improvements in Task 3. These modifications would reduce the 
    overall job demand and move the Composite Lifting Index closer to the NIOSH 
    design goal of 1.0."
    """

        # ---------------------------------------------------------
        # USER PROMPT MULTI-TASK COMPLETO
        # ---------------------------------------------------------
        user_prompt = f"""
    FEW-SHOT EXAMPLE (IMITATE THIS STYLE CLOSELY):
    {fewshot}

    MULTI-TASK REDESIGN INPUT
    Job description (context only — do NOT alter):
    {calc_result.job_description}

    Composite Lifting Index:
    CLI = {calc_result.cli:.2f} [CLI:{calc_result.cli:.2f}]
    Risk category: {calc_result.risk_category}
    Duration: [DUR:{duration}]

    Single-task LI values (do NOT change):
    {', '.join([f'[LI{t.task_id}:{t.stli:.2f}]' for t in calc_result.tasks])}

    TASK DETAILS (DO NOT MODIFY ANY VALUE):
    {tasks_block_str}

    INSTRUCTIONS:
    Write MULTI-TASK REDESIGN SUGGESTIONS in the style of NIOSH Examples 7–8.

    You MUST:
    • Identify WHICH TASKS have the highest LI values.
    • Explain WHY these tasks dominate the Composite Lifting Index.
    • Propose redesign strategies ONLY by referencing multipliers conceptually:
        – Reduce horizontal reach → increases HM  
        – Move V toward ~30 inches → increases VM  
        – Reduce asymmetry → increases AM  
        – Improve coupling → increases CM  
        – Reduce frequency (if applicable) → improves FM  

    Strict rules:
    • NEVER recompute LIi or CLI.
    • NEVER invent new heights, distances, angles, or multipliers.
    • Use ONLY the numerical values given above.
    • Include TAGS inline when mentioning task values.
    • Produce one continuous narrative paragraph in formal NIOSH style.
    """

        # ---------------------------------------------------------
        # CHIAMATA AL MODELLO
        # ---------------------------------------------------------
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._create_system_prompt_multi_task(),
                    },
                    {"role": "user", "content": user_prompt},
                ],
            )
            # Handle both object and dict response formats from Ollama
            if hasattr(response, "message"):
                return response.message.content
            elif isinstance(response, dict) and "message" in response:
                return response["message"]["content"]
            else:
                return str(response)
        except Exception as e:
            return f"Errore generazione redesign suggestions: {e}"

    def _format_numerical_projection(self, improved_scenario: Dict) -> str:
        """
        Formatta la proiezione numerica per il riferimento nel prompt.
        """
        improvements_text = ", ".join(improved_scenario["improvements"])
        return f"""Projected improvements: {improvements_text}
Expected RWL increase to {improved_scenario["new_rwl"]:.1f} lbs (from current)
Expected LI reduction to {improved_scenario["new_li"]:.2f} (from current)
Projected LI improvement: {improved_scenario["improvement_percentage"]:.1f}%"""

    def _create_numerical_appendix(
        self, original_data: RedesignSuggestionsInput, improved: Dict
    ) -> str:
        """
        Crea appendice numerica con confronto prima/dopo RWL/LI collegato a modifiche geometriche.
        """
        improvements_text = ", ".join(improved["improvements"])

        # Crea descrizione delle modifiche geometriche specifiche solo se ci sono miglioramenti reali
        geometric_changes = []
        if improved["improvements"]:
            if "horizontal" in improvements_text.lower():
                old_h = original_data.h_origin
                new_h = max(
                    10, original_data.h_origin * 0.75
                )  # Dalla logica di miglioramento
                geometric_changes.append(
                    f"horizontal reach reduced from {old_h:.0f} to {new_h:.0f} inches"
                )

            if "frequency" in improvements_text.lower():
                old_f = original_data.frequency_lifts_per_min
                new_f = max(0.5, original_data.frequency_lifts_per_min * 0.7)
                geometric_changes.append(
                    f"lifting frequency reduced from {old_f:.1f} to {new_f:.1f} lifts/min"
                )

            if (
                "vertical" in improvements_text.lower()
                or "height" in improvements_text.lower()
            ):
                old_v = original_data.v_origin
                if old_v < 25:
                    new_v = min(30, old_v + 5)
                    geometric_changes.append(
                        f"vertical height adjusted from {old_v:.0f} to {new_v:.0f} inches toward optimal range"
                    )
                elif old_v > 35:
                    new_v = max(30, old_v - 5)
                    geometric_changes.append(
                        f"vertical height adjusted from {old_v:.0f} to {new_v:.0f} inches toward optimal range"
                    )

            if "asymmetry" in improvements_text.lower():
                old_a = original_data.a_origin
                new_a = min(15, original_data.a_origin * 0.5)
                geometric_changes.append(
                    f"asymmetry angle reduced from {old_a:.0f}° to {new_a:.0f}°"
                )

            if "coupling" in improvements_text.lower():
                geometric_changes.append(
                    f"hand-to-object coupling improved from '{original_data.coupling}' to 'good'"
                )

        geometric_description = (
            "; ".join(geometric_changes) if geometric_changes else improvements_text
        )

        # Include destination values if available
        dest_info = ""
        if original_data.rwl_dest_lbs is not None and original_data.li_dest is not None:
            dest_info = f" At the destination, LI is {original_data.li_dest:.2f} (RWL = {original_data.rwl_dest_lbs:.1f} lbs)."

        # Verifica se ci sono miglioramenti reali da mostrare
        if not improved["improvements"] or improved["improvement_percentage"] <= 0:
            return f"""QUANTITATIVE IMPROVEMENT ASSESSMENT:
Original configuration: LI at the origin = {original_data.li_origin:.2f} (RWL = {original_data.rwl_origin_lbs:.1f} lbs).{dest_info}
The proposed redesign changes would increase the RWL at both the origin and destination while reducing the LI values. However, specific numerical projections should be calculated based on the exact parameter changes implemented. These improvements would move the task toward safer limits, although the task would likely remain above the ideal design goal of LI = 1.0."""
        else:
            return f"""QUANTITATIVE IMPROVEMENT ASSESSMENT:
Original configuration: LI at the origin = {original_data.li_origin:.2f} (RWL = {original_data.rwl_origin_lbs:.1f} lbs).{dest_info}
Proposed redesign improvements: {geometric_description}

New RWL calculation at the origin:
RWL_new = 51 × HM_new × VM_new × DM_new × AM_new × FM_new × CM_new = {improved["new_rwl"]:.1f} lbs
LI_new = {original_data.weight_lbs:.1f} / {improved["new_rwl"]:.1f} = {improved["new_li"]:.2f}

This represents a {improved["improvement_percentage"]:.1f}% improvement in the lifting index at the origin, moving the task closer to the design goal of LI = 1.0."""
