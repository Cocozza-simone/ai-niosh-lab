# consistency_checks.py

from typing import Tuple, List, Optional, Dict
import re

from niosh_parameters import NIOSHParameters
from niosh_calculator import NIOSHCalculator


def _fix_numeric_field_in_text(
    text: str,
    pattern: str,
    true_value: float,
    tolerance: float = 1.0,
    label: str = "",
) -> Tuple[str, List[str]]:
    """
    Cerca pattern con un singolo gruppo numerico, per esempio:
        r"H\\s*=\\s*(\\d+(?:\\.\\d+)?)\\s*inches"

    Se il numero trovato differisce da true_value più della `tolerance`,
    lo sostituisce nel testo.

    Ritorna:
        nuovo_testo, lista_warning
    """
    warnings: List[str] = []
    if not text:
        return text, warnings

    def repl(match: re.Match) -> str:
        nonlocal warnings
        old_str = match.group(1)
        try:
            old_val = float(old_str)
        except ValueError:
            return match.group(0)

        if abs(old_val - true_value) <= tolerance:
            return match.group(0)

        warnings.append(
            f"Correzione nel testo{(' per ' + label) if label else ''}: "
            f"{old_val} → {true_value}."
        )

        start, end = match.span(1)
        return (
            match.string[match.start() : start]
            + f"{true_value:.0f}"
            + match.string[end : match.end()]
        )

    new_text = re.sub(pattern, repl, text)
    return new_text, warnings


def ensure_job_consistency(job: Dict) -> Tuple[Optional[Dict], List[str]]:
    """
    Controllo + correzione di coerenza su un singolo job NIOSH.

    Input: dict nello stesso formato che salvi in JSON, ad es.:

        {
          "input_phrase": ...,
          "description": ...,
          "parameters": {...},
          "niosh_calculation": {...},
          "simplified_calculation_lbs": ...,
          "job_analysis_text": ...,
          "hazard_assessment_text": ...,
          "redesign_suggestions_text": ...,
          "_text": ...
        }

    Output:
        - job_corretto (dict) oppure None se i parametri sono fuori range / non validi
        - lista di warning / correzioni applicate
    """
    warnings: List[str] = []

    # 1) Recupero e controllo struttura parametri
    params_raw = job.get("parameters")
    if not isinstance(params_raw, dict):
        warnings.append("Struttura 'parameters' mancante o non valida.")
        return None, warnings

    try:
        params = NIOSHParameters(**params_raw)
    except Exception as e:
        warnings.append(
            f"Errore nella costruzione di NIOSHParameters (Validazione): {e}"
        )
        return None, warnings

    # 2) Validazione range NIOSH (Pydantic ha già validato)
    # ok, errors = params.validate() -> Ritorna sempre True ora
    # if not ok: ...

    description = job.get("description", "") or ""
    desc_lower = description.lower()

    # 3) Coerenza torsione (A) ↔ testo
    negative_twist_phrases = [
        "no asymmetric lifting",
        "no asymmetric lift",
        "no twisting is involved",
        "does not twist",
        "doesn't twist",
        "does not have to twist",
        "does not twist significantly",
        "non ruota",
        "non ruota in modo significativo",
        "senza torsione",
        "senza rotazione del tronco",
    ]
    if (
        any(p in desc_lower for p in negative_twist_phrases)
        and params.asymmetry_angle > 0
    ):
        warnings.append(
            f"Asimmetria A={params.asymmetry_angle}° "
            "corretta a 0° perché il testo dichiara assenza di torsione."
        )
        params.asymmetry_angle = 0.0

    # 4) Coerenza 'significant control' ↔ testo
    pos_control_phrases = [
        "significant control of the load",
        "significant control of the object",
        "controllo significativo del carico",
        "controllo significativo dell'oggetto",
    ]
    neg_control_phrases = [
        "no significant control",
        "significant control of the load is not required",
        "significant control of the object is not required",
        "controllo significativo non è richiesto",
        "non è richiesto un controllo significativo",
    ]

    if (
        any(p in desc_lower for p in pos_control_phrases)
        and not params.significant_control
    ):
        warnings.append(
            "Flag significant_control portato a True perché il testo "
            "dichiara che è richiesto un controllo significativo."
        )
        params.significant_control = True

    if any(p in desc_lower for p in neg_control_phrases) and params.significant_control:
        warnings.append(
            "Flag significant_control portato a False perché il testo "
            "dichiara che NON è richiesto un controllo significativo."
        )
        params.significant_control = False

    # 5) Allineo H_origin e H_destination ai valori espliciti nel testo (se presenti)
    # Cerco tutte le occorrenze di "H=... inches" o "H=approximately ... inches"
    h_matches = re.findall(
        r"H\s*=\s*(?:approximately\s*)?(\d+(?:\.\d+)?)\s*inches",
        description,
    )

    # Se trovo almeno un H → lo interpreto come H_origin
    if len(h_matches) >= 1:
        try:
            h_text_origin = float(h_matches[0])
            if abs(h_text_origin - params.horizontal_origin) > 1.0:
                warnings.append(
                    f"horizontal_origin corretto da {params.horizontal_origin} "
                    f"a {h_text_origin} in base al Job Description."
                )
                params.horizontal_origin = h_text_origin
        except ValueError:
            warnings.append(
                f"Impossibile interpretare H_origin dal testo: '{h_matches[0]}'"
            )

    # Se trovo un secondo H → lo interpreto come H_destination
    if len(h_matches) >= 2:
        try:
            h_text_dest = float(h_matches[1])
            if abs(h_text_dest - params.horizontal_destination) > 1.0:
                warnings.append(
                    f"horizontal_destination corretto da {params.horizontal_destination} "
                    f"a {h_text_dest} in base al Job Description."
                )
                params.horizontal_destination = h_text_dest
        except ValueError:
            warnings.append(
                f"Impossibile interpretare H_destination dal testo: '{h_matches[1]}'"
            )

    # 6) Ricalcolo NIOSH dai parametri aggiornati
    calc = NIOSHCalculator()
    calc_result = calc.compute(params)
    job["niosh_calculation"] = calc_result.as_dict()

    # 7) Aggiorno anche il blocco 'parameters' con i valori eventualmente corretti
    job["parameters"] = {
        **params_raw,
        "horizontal_origin": params.horizontal_origin,
        "horizontal_destination": params.horizontal_destination,
        "asymmetry_angle": params.asymmetry_angle,
        "significant_control": params.significant_control,
    }

    return job, warnings
