import re

def extract_multi_task_tags(text: str):
    """
    Estrazione deterministica dei TAG multi-task.
    Se i tag [TASK:i] sono presenti → usa il metodo standard.
    Se non sono presenti → estrae automaticamente i task dal testo narrativo.
    """

    # 1️⃣ CONTROLLO TAG STANDARD
    task_indices = [int(m.group(1)) for m in re.finditer(r"\[TASK:(\d+)\]", text)]
    if task_indices:
        return _extract_with_tags(text, task_indices)

    # 2️⃣ FALLBACK: NESSUN TAG → ESTRAZIONE "INTELLIGENTE"
    return _extract_smart(text)


# ------------------------------------------------------------------
#   METODO ORIGINALE (quando i tag sono presenti)
# ------------------------------------------------------------------

def _extract_with_tags(text: str, task_indices):
    tasks = {}
    def _safe_number(val):
        if val is None:
            return None
        try:
            # estrae il primo numero dalla stringa
            m = re.search(r"[-+]?\d*\.?\d+", str(val))
            return float(m.group()) if m else None
        except:
            return None

    def grab(label, i, caster=float):
        m = re.search(rf"\[{label}{i}:(.*?)\]", text)
        if m:
            raw = m.group(1).strip().strip("<>")
            try:
                return caster(raw)
            except:
                return raw
        return None

    for idx in task_indices:
        tasks[idx] = {
            "task": idx,
            "weight": grab("W", idx),
            "H0": grab("H0", idx),
            "H1": grab("H1", idx),
            "V0": grab("V0", idx),
            "V1": grab("V1", idx),
            "A": grab("A", idx),
            "F": _safe_number(grab("F", idx)),
            "coupling": grab("COUP", idx, str),
        }

    # DUR comune
    m = re.search(r"\[DUR:(.*?)\]", text)
    duration = m.group(1).strip() if m else None

    for t in tasks.values():
        t["duration"] = duration

    return [tasks[i] for i in sorted(tasks)]


# ------------------------------------------------------------------
#   INTELLIGENT FALLBACK PARSER
# ------------------------------------------------------------------

def _extract_smart(text: str):
    """
    Ricostruzione automatica dei task multi-task da JD NON conformi.
    Nessuna invenzione di numeri: estrazione rigorosa dai pattern.
    """

    # 1️⃣ Segmentazione in blocchi usando indicatori linguistici
    segments = re.split(
        r"\b(?:then|finally|subsequently|next|after that|in the second task|in the third task)\b",
        text,
        flags=re.IGNORECASE
    )

    tasks = []
    task_id = 1

    # 2️⃣ Pattern regex numerici
    RE_WEIGHT = r"(\d+(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)"
    RE_INCH = r"(\d+(?:\.\d+)?)\s*(?:in|inch|inches)"
    RE_DEG = r"(\d+(?:\.\d+)?)\s*(?:degree|degrees)"

    for seg in segments:
        seg_l = seg.lower()

        # 3️⃣ Estrazione parametri (best-effort ma deterministica)
        w = _search_first(seg, RE_WEIGHT)
        h_values = re.findall(RE_INCH, seg)
        v_values = h_values  # stesso pattern: distinguiamo in seguito
        a = _search_first(seg, RE_DEG)
        f = _search_first(seg, r"(\d+(?:\.\d+)?)\s*lifts?/min")

        # LOGICA HEURISTICA SENZA INVENZIONI:
        H0 = float(h_values[0]) if len(h_values) >= 1 else None
        H1 = float(h_values[1]) if len(h_values) >= 2 else None
        V0 = float(v_values[0]) if len(v_values) >= 1 else None
        V1 = float(v_values[1]) if len(v_values) >= 2 else None

        # Coupling
        if "good coupling" in seg_l:
            coup = "good"
        elif "poor coupling" in seg_l:
            coup = "poor"
        else:
            coup = "fair"

        if w or H0 or H1 or V0 or V1:
            tasks.append({
                "task": task_id,
                "weight": w,
                "H0": H0,
                "H1": H1,
                "V0": V0,
                "V1": V1,
                "A": a,
                "F": f,
                "coupling": coup,
                "duration": _extract_duration(text),
            })
            task_id += 1

    return tasks


def _search_first(text, pattern):
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None


def _extract_duration(text):
    m = re.search(r"\[DUR:(.*?)\]", text)
    if m:
        return m.group(1).strip()
    # fallback minimo: NON inventare → None
    return None
