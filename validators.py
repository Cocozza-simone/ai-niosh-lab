from model_router import call_llm_with_system_prompt
import json
from threading import Semaphore

LLM_SEMAPHORE = Semaphore(2)

# ============================================================
#  LLM SEMANTIC GUESS
# ============================================================

def llm_semantic_guess(text: str, field: str, model: str = "gemma3:12b"):
    """
    Inferenza semantica robusta: restituisce un numero SOLO se è
    chiaramente deducibile dal testo.
    In caso di ambiguità → null (None).
    """

    system = """
You are an expert ergonomist specialized in the Revised NIOSH Lifting Equation.
Infer ONLY IF CLEAR AND CERTAIN a missing parameter.
Never guess randomly.

Rules:
- "on the floor", "from ground" → V0 = 0
- "waist height", "table", "bench", "shelf" → V around 30–36 inches
- "shoulder height", "chest height" → V around 48–55 inches
- "overhead", "above head" → V > 55 inches
- "close to body", "held near torso" → H 10–15 inches
- "reaching", "arms extended", "deep shelf" → H 20–30 inches
- "no twisting" → A = 0
- "turning", "rotating", "pivoting" → A = 30–60
- High frequency terms → F >= 2
- Low / occasional terms → F <= 0.5

If unclear → ALWAYS answer null.
Output exactly:
{"value": <number or null>}
"""

    user = f"""
Job description:
{text}

Infer the field: {field}

Respond ONLY with the JSON object:
{{"value": <number or null>}}
"""

    try:
        with LLM_SEMAPHORE:
            raw = call_llm_with_system_prompt(
                model=model,
                user_prompt=user,
                system_prompt=system,
                temperature=0.0
            )
        answer = json.loads(raw)
        return answer.get("value", None)
    except:
        return None



# ============================================================
#  INTELLIGENT LLM PREVALIDATOR
# ============================================================

def intelligent_llm_prevalidator(raw: dict, job_text: str, model="gemma3:12b") -> dict:
    """
    Combina inferenza semantica LLM + default NIOSH per ottenere
    un dizionario COMPLETO, coerente e pronto per il calcolo RNLE.
    """
    out = dict(raw) if raw else {}

    # ---------- 1) WEIGHT ----------
    out["weight"] = out.get("weight") or 25.0


    # ---------- 2) HORIZONTAL ORIGIN ----------
    if out.get("horizontal_origin") is None:
        guess = llm_semantic_guess(job_text, "horizontal_origin", model)
        out["horizontal_origin"] = guess or 20.0

    # ---------- 3) HORIZONTAL DESTINATION ----------
    if out.get("horizontal_destination") is None:
        guess = llm_semantic_guess(job_text, "horizontal_destination", model)
        out["horizontal_destination"] = guess or out["horizontal_origin"]

    # Clamp NIOSH
    out["horizontal_origin"] = max(10.0, out["horizontal_origin"])
    out["horizontal_destination"] = max(10.0, out["horizontal_destination"])


    # ---------- 4) VERTICAL ORIGIN ----------
    if out.get("vertical_origin") is None:
        guess = llm_semantic_guess(job_text, "vertical_origin", model)
        out["vertical_origin"] = guess if guess is not None else 30.0

    # ---------- 5) VERTICAL DESTINATION ----------
    if out.get("vertical_destination") is None:
        guess = llm_semantic_guess(job_text, "vertical_destination", model)
        out["vertical_destination"] = guess if guess is not None else out["vertical_origin"]

    # Clamp V (NIOSH max 70)
    out["vertical_origin"] = max(0.0, min(out["vertical_origin"], 70.0))
    out["vertical_destination"] = max(0.0, min(out["vertical_destination"], 70.0))


    # ---------- 6) ASYMMETRY ----------
    if out.get("asymmetry_angle") is None:
        guess = llm_semantic_guess(job_text, "asymmetry_angle", model)
        out["asymmetry_angle"] = guess if guess is not None else 0.0


    # ---------- 7) FREQUENCY ----------
    if out.get("frequency") is None:
        guess = llm_semantic_guess(job_text, "frequency", model)
        out["frequency"] = guess if guess is not None else 0.1


    # ---------- 8) DURATION ----------
    out["duration"] = out.get("duration") or "<1h"


    # ---------- 9) COUPLING ----------
    out["coupling"] = out.get("coupling") or "fair"


    # ---------- 10) SIGNIFICANT CONTROL ----------
    if out.get("significant_control") is None:
        guess = llm_semantic_guess(job_text, "significant_control", model)
        out["significant_control"] = bool(guess) if isinstance(guess, bool) else False

    return out



# ============================================================
#  VALIDATOR (POST-PREVALIDATION)
# ============================================================

def validate_params(params: dict) -> tuple:
    """
    Validazione dei parametri AFTER prevalidator (quindi niente missing).
    Controlla solo range RNLE + coerenze geometriche.
    Restituisce (ok, message)
    """

    # Controlla prima se ci sono valori None
    required_keys = ["weight", "asymmetry_angle", "frequency", "duration", "coupling",
                     "vertical_origin", "vertical_destination", 
                     "horizontal_origin", "horizontal_destination"]
    
    for key in required_keys:
        if key not in params or params[key] is None:
            return False, f"Missing or None value for {key}"

    # WEIGHT
    if not (0 < params["weight"] <= 200):
        return False, "Weight must be 1–200 lbs"

    # ASYMMETRY
    if not (0 <= params["asymmetry_angle"] <= 135):
        return False, "Asymmetry must be between 0° and 135°"

    # FREQUENCY
    if not (0 <= params["frequency"] <= 15):
        return False, "Frequency must be between 0 and 15 lifts/min"

    # DURATION CATEGORY
    if params["duration"] not in ["<1h", "1-2h", "2-8h", ">8h"]:
        return False, f"Invalid duration category: {params['duration']}"

    # COUPLING
    if params["coupling"] not in ["good", "fair", "poor"]:
        return False, "Coupling must be: good / fair / poor"

    # GEOMETRIA
    if params["vertical_origin"] < 0 or params["vertical_destination"] < 0:
        return False, "Vertical values cannot be negative"

    if params["horizontal_origin"] < 0 or params["horizontal_destination"] < 0:
        return False, "Horizontal values cannot be negative"

    # TUTTO OK
    return True, None