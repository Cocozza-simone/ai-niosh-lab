import re
from typing import List, Dict, Any, Optional


def extract_multi_task_tags(text: str) -> List[Dict[str, Any]]:
    """
    Estrae i TAG multi-task da una Job Description MULTI-TASK.

    Supporta sia i tag stile:
        [TASK:1] [W1:25] [H01:16] [H11:20] [V01:12] [V11:36] [A1:35] [F1:3] [COUP1:fair] [CTRL1:false] [DUR:2-8h]

    sia i tag stile Application Manual:
        [TASK:1] [Wi:28] [H0i:16] [H1i:14] [V0i:12] [V1i:28] [Ai:10] [Fi:3] [COUPi:fair] [CTRLi:false] [DUR:2-8h]

    Ritorna:
        [
          {
            "task": 1,
            "weight": 25.0,
            "H0": 16.0,
            "H1": 20.0,
            "V0": 12.0,
            "V1": 36.0,
            "A": 35.0,
            "F": 3.0,
            "coupling": "fair",
            "significant_control": False,
            "duration": "2-8h"
          },
          ...
        ]
    """

    if not text:
        return []

    txt = text

    # ---- 1) Durata globale (valida per tutte le task) ----
    dur_match = re.search(r"\[DUR:([^\]]+)\]", txt, flags=re.I)
    global_duration: Optional[str] = None
    if dur_match:
        global_duration = dur_match.group(1).strip()

    # ---- 2) Individua tutti i TASK: i ----
    task_ids = re.findall(r"\[TASK:(\d+)\]", txt, flags=re.I)
    tasks: List[Dict[str, Any]] = []

    for tid_str in task_ids:
        i = int(tid_str)

        task_data: Dict[str, Any] = {"task": i}

        def find_numeric(tag_base: str) -> Optional[float]:
            """
            Cerca tag numerici in due forme:
                [TAGi:val]   es. [Wi:28]
                [TAG{i}:val] es. [W1:28], [H01:16], [V01:12]
            """
            # forma con indice numerico esplicito (H01, V11, W1, ecc.)
            patt_num = rf"\[{tag_base}{i}:([^\]]+)\]"
            m = re.search(patt_num, txt, flags=re.I)
            if m:
                val_str = m.group(1).strip()
                try:
                    return float(val_str)
                except ValueError:
                    return None

            # forma con suffisso 'i' (Wi, V0i, V1i, ecc.)
            patt_i = rf"\[{tag_base}i:([^\]]+)\]"
            m = re.search(patt_i, txt, flags=re.I)
            if m:
                val_str = m.group(1).strip()
                try:
                    return float(val_str)
                except ValueError:
                    return None

            return None

        def find_str(tag_base: str) -> Optional[str]:
            """
            Cerca tag stringa:
                [TAGi:val] oppure [TAG{i}:val]
            es. [COUP1:fair], [COUPi:good], [CTRL1:true]
            """
            patt_num = rf"\[{tag_base}{i}:([^\]]+)\]"
            m = re.search(patt_num, txt, flags=re.I)
            if m:
                return m.group(1).strip()

            patt_i = rf"\[{tag_base}i:([^\]]+)\]"
            m = re.search(patt_i, txt, flags=re.I)
            if m:
                return m.group(1).strip()

            return None

        # ---- 3) Mappa i tag numerici ----
        w = find_numeric("W")
        h0 = find_numeric("H0")
        h1 = find_numeric("H1")
        v0 = find_numeric("V0")
        v1 = find_numeric("V1")
        a = find_numeric("A")
        f = find_numeric("F")

        if w is not None:
            task_data["weight"] = w
        if h0 is not None:
            task_data["H0"] = h0
        if h1 is not None:
            task_data["H1"] = h1
        if v0 is not None:
            task_data["V0"] = v0
        if v1 is not None:
            task_data["V1"] = v1
        if a is not None:
            task_data["A"] = a
        if f is not None:
            task_data["F"] = f

        # ---- 4) Coupling & Control ----
        coup_raw = find_str("COUP")
        if coup_raw:
            task_data["coupling"] = coup_raw.lower()

        ctrl_raw = find_str("CTRL")
        if ctrl_raw:
            c = ctrl_raw.strip().lower()
            task_data["significant_control"] = c in ("true", "yes", "1")

        # ---- 5) Durata ----
        if global_duration:
            task_data["duration"] = global_duration

        tasks.append(task_data)

    return tasks


def _extract_single_task_as_multi(text: str):
    """Estrae un singolo task dai tag standard e lo restituisce come lista di 1 task."""

    def grab(label, caster=float):
        m = re.search(rf"\[{label}:(.*?)\]", text)
        if m:
            raw = m.group(1).strip().strip("<>")
            try:
                return caster(raw)
            except:
                return raw
        return None

    # Estrazione Duration
    m = re.search(r"\[DUR:(.*?)\]", text)
    duration = m.group(1).strip() if m else None

    task = {
        "task": 1,
        "weight": grab("W"),
        "H0": grab("H0"),
        "H1": grab("H1"),
        "V0": grab("V0"),
        "V1": grab("V1"),
        "A": grab("A"),
        "F": grab("F"),
        "coupling": grab("COUP", str),
        "duration": duration,
    }

    return [task]


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
        flags=re.IGNORECASE,
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
            tasks.append(
                {
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
                }
            )
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
