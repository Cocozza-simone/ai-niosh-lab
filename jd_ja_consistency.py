def compare_parameters(jd_params: dict, ja_params: dict):
    """
    Confronta parametri JD ↔ JA.
    Restituisce (ok:bool, discrepancies:list).
    """

    discrepancies = []

    # Crea l’unione di tutte le chiavi
    all_keys = set(jd_params.keys()) | set(ja_params.keys())

    for k in all_keys:
        jd_val = jd_params.get(k)
        ja_val = ja_params.get(k)

        # Se uno manca → errore
        if jd_val is None and ja_val is not None:
            discrepancies.append(f"JA contains '{k}' but JD does not.")
            continue
        if jd_val is not None and ja_val is None:
            discrepancies.append(f"JD contains '{k}' but JA does not.")
            continue

        # Numeri diversi?
        if isinstance(jd_val, (int, float)) and isinstance(ja_val, (int, float)):
            if abs(float(jd_val) - float(ja_val)) > 0.0001:
                discrepancies.append(f"Mismatch on {k}: JD={jd_val}, JA={ja_val}")
        else:
            # String mismatch
            if jd_val != ja_val:
                discrepancies.append(f"Mismatch on {k}: JD={jd_val}, JA={ja_val}")

    return (len(discrepancies) == 0, discrepancies)


def check_jd_ja_consistency(jd_text_with_tags: str, ja_text_with_tags: str, extractor):
    """
    Validatore completo che:
    - estrae parametri da JD e JA tramite extractor()
    - confronta i due set di parametri
    - ritorna (ok, discrepancies)
    """

    jd_params = extractor(jd_text_with_tags)
    ja_params = extractor(ja_text_with_tags)

    ok, discrepancies = compare_parameters(jd_params, ja_params)

    return ok, jd_params, ja_params, discrepancies
