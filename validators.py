from model_router import call_llm_with_system_prompt
import json
import re

# ============================================================
#  NORMALIZZATORE DURATA NIOSH
# ============================================================

def normalize_duration_tag(raw_dur: str) -> str:
    """
    Normalizza qualsiasi formato del tag [DUR:...] alle 4 categorie RNLE:
    <1h, 1-2h, 2-8h, >8h.
    È tollerante a forme come '2h', '3h', '<4h', '90min', ecc.
    """

    if not raw_dur:
        return "2-8h"

    d = raw_dur.lower().strip()

    # già valido
    if d in ("<1h", "1-2h", "2-8h", ">8h"):
        return d

    # pattern tipo "3h", "2h"
    if d.endswith("h") and d[:-1].isdigit():
        h = float(d[:-1])
        if h < 1:
            return "<1h"
        if 1 <= h <= 2:
            return "1-2h"
        if 2 < h <= 8:
            return "2-8h"
        return ">8h"

    # pattern "<3h"
    if d.startswith("<") and d[1:-1].isdigit():
        h = float(d[1:-1])
        if h <= 1:
            return "<1h"
        if 1 < h <= 2:
            return "1-2h"
        if 2 < h <= 8:
            return "2-8h"
        return ">8h"

    # pattern "90min" o simili
    if "min" in d:
        m = re.search(r"(\d+)", d)
        if m:
            minutes = float(m.group(1))
            h = minutes / 60
            if h < 1:
                return "<1h"
            if h <= 2:
                return "1-2h"
            if h <= 8:
                return "2-8h"
            return ">8h"

    # fallback prudenziale
    return "2-8h"


# ============================================================
#  LLM SEMANTIC GUESS
# ============================================================

def llm_semantic_guess(text: str, field: str, model: str = "gemma3:12b"):
    """
    Inferenza semantica controllata. Restituisce un numero SOLO se chiaramente deducibile.
    Output: {"value": numero | null}
    """

    system = """
You are an expert ergonomist specialized in the Revised NIOSH Lifting Equation.
Infer ONLY IF CLEAR AND CERTAIN a missing parameter. Never guess randomly.

Rules:
- "on the floor", "from ground" → V0 = 0
- "waist height", "table", "bench", "shelf" → V ≈ 30–36 in
- "shoulder height", "chest height" → V ≈ 48–55 in
- "overhead", "above head" → V > 55 in

- "close to body" → H ≈ 10–15 in
- "reaching", "arms extended", "deep shelf" → H ≈ 20–30 in

- "no twisting" → A = 0
- "turning", "rotating", "pivoting" → A = 30–60°

- High frequency terms → F >= 2
- Low/occasional → F <= 0.5

If unclear → ALWAYS answer null.

Respond ONLY:
{"value": <number or null>}
"""

    user = f"""
Job description:
{text}

Infer the field: {field}

Return ONLY:
{{"value": <number or null>}}
"""

    try:
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
    Completa i parametri mancanti usando regole RNLE + inferenza LLM controllata.
    Prepara un dizionario coerente per il calcolo.
    """

    out = dict(raw) if raw else {}

    # ------- 1) WEIGHT -------
    out["weight"] = out.get("weight") or 25.0

    # ------- 2) H0 -------
    if out.get("horizontal_origin") is None:
        g = llm_semantic_guess(job_text, "horizontal_origin", model)
        out["horizontal_origin"] = g or 20.0
    out["horizontal_origin"] = max(10.0, out["horizontal_origin"])

    # ------- 3) H1 -------
    if out.get("horizontal_destination") is None:
        g = llm_semantic_guess(job_text, "horizontal_destination", model)
        out["horizontal_destination"] = g or out["horizontal_origin"]
    out["horizontal_destination"] = max(10.0, out["horizontal_destination"])

    # ------- 4) V0 -------
    if out.get("vertical_origin") is None:
        g = llm_semantic_guess(job_text, "vertical_origin", model)
        out["vertical_origin"] = g if g is not None else 30.0
    out["vertical_origin"] = max(0.0, min(out["vertical_origin"], 70.0))

    # ------- 5) V1 -------
    if out.get("vertical_destination") is None:
        g = llm_semantic_guess(job_text, "vertical_destination", model)
        out["vertical_destination"] = g if g is not None else out["vertical_origin"]
    out["vertical_destination"] = max(0.0, min(out["vertical_destination"], 70.0))

    # ------- 6) ASSEMBRIA -------
    if out.get("asymmetry_angle") is None:
        g = llm_semantic_guess(job_text, "asymmetry_angle", model)
        out["asymmetry_angle"] = g if g is not None else 0.0

    # ------- 7) FREQUENCY -------
    if out.get("frequency") is None:
        g = llm_semantic_guess(job_text, "frequency", model)
        out["frequency"] = g if g is not None else 0.1
    if "repeated" in job_text.lower() or "continuously" in job_text.lower():
        out["frequency"] = 3.0
    # CLAMP NIOSH – evita errori di validazione
    out["frequency"] = max(0.1, min(out["frequency"], 15.0))
    if out["frequency"] >= 15:
        print(f"[WARN] Frequency clamped from {out['frequency']} to 15.0")

    # ------- 8) DURATION -------
    dur_raw = out.get("duration") or "<1h"
    out["duration"] = normalize_duration_tag(dur_raw)

    # ------- 9) COUPLING -------
    out["coupling"] = (out.get("coupling") or "fair").lower()
    if out["coupling"] not in ["good", "fair", "poor"]:
        out["coupling"] = "fair"

    # ------- 10) SIGNIFICANT CONTROL -------
    sc = out.get("significant_control")
    if sc is None:
        g = llm_semantic_guess(job_text, "significant_control", model)
        out["significant_control"] = bool(g) if isinstance(g, bool) else False
    else:
        out["significant_control"] = bool(sc)

    return out


# ============================================================
#  VALIDATOR (POST-PREVALIDATION)
# ============================================================

def validate_params(params: dict) -> tuple:
    """
    Validazione finale RNLE.
    Controlla range numerici e categorie discrete.
    """

    required = [
        "weight","asymmetry_angle","frequency","duration","coupling",
        "vertical_origin","vertical_destination",
        "horizontal_origin","horizontal_destination"
    ]

    for k in required:
        if params.get(k) is None:
            return False, f"Missing or None value for {k}"

    # WEIGHT
    if not (0 < params["weight"] <= 200):
        return False, "Weight must be 1–200 lbs"

    # ASYMMETRY
    if not (0 <= params["asymmetry_angle"] <= 135):
        return False, "Asymmetry must be between 0° and 135°"

    # FREQUENCY
    if not (0 <= params["frequency"] <= 15):
        return False, "Frequency must be between 0 and 15 lifts/min"

    # DURATION
    if params["duration"] not in ["<1h","1-2h","2-8h",">8h"]:
        return False, f"Invalid duration category: {params['duration']}"

    # COUPLING
    if params["coupling"] not in ["good","fair","poor"]:
        return False, "Coupling must be: good / fair / poor"

    # GEOMETRIA
    if params["vertical_origin"] < 0 or params["vertical_destination"] < 0:
        return False, "Vertical values cannot be negative"

    if params["horizontal_origin"] < 10 or params["horizontal_destination"] < 10:
        return False, "Horizontal reach must be >= 10 in (RNLE constraint)"

    return True, None
