from model_router import call_llm_with_system_prompt


def classify_sentence(text: str, model: str = "llama3.2:latest") -> str:
    """
    Classifica uno scenario di sollevamento in:
    - "single"
    - "repetitive"
    - "multi"

    Logica:
    1) Euristica deterministica (parole chiave)
    2) Solo se necessario → LLM
    """

    t = text.lower()

    # 1) EURISTICA: REPETITIVE (marcatori di frequenza)
    repetitive_markers = [
        "repeatedly",
        "repeated",
        "continuously",
        "throughout the shift",
        "for most of the shift",
        "all morning",
        "all day",
        "throughout the packing operation",
        "for the entire shift",
    ]
    if any(m in t for m in repetitive_markers):
        return "repetitive"

    # 2) EURISTICA: MULTI (sequenze esplicite di azioni)
    sequence_markers = [
        " then ",
        " and then ",
        " next ",
        " and next ",
        " followed by ",
        " after that ",
        " and finally ",
        " finally ",
    ]
    if any(m in t for m in sequence_markers):
        return "multi"

    # 3) EURISTICA: MULTI per 3+ verbi d'azione distinti
    verbs = [
        "lift",
        "lifts",
        "carry",
        "carries",
        "push",
        "pushes",
        "pull",
        "pulls",
        "rotate",
        "rotates",
        "slide",
        "slides",
        "stack",
        "stacks",
        "place",
        "places",
        "set",
        "sets",
    ]
    found_verbs = [v for v in verbs if f" {v} " in t]
    if len(set(found_verbs)) >= 3:
        return "multi"

    # 4) FALLBACK: usa LLM come già facevi
    system = """
You are an expert in ergonomic task analysis and the Revised NIOSH Lifting Equation.

Classify the following lifting scenario into ONLY ONE category:

- multi
- repetitive
- single

Output ONLY one word.
"""
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
