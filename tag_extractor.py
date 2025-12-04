import re


def extract_tagged_params(description: str):
    """Estrae i parametri NIOSH dai TAG inline generati nella Job Description."""
    params = {}

    def grab(label, caster=float):
        m = re.search(rf"\[{label}:(.*?)\]", description)
        if m:
            raw = m.group(1).strip()
            try:
                return caster(raw)
            except:
                return None
        return None

    params["weight"] = grab("W")
    params["horizontal_origin"] = grab("H0")
    params["horizontal_destination"] = grab("H1")
    params["vertical_origin"] = grab("V0")
    params["vertical_destination"] = grab("V1")
    params["asymmetry_angle"] = grab("A")
    params["frequency"] = grab("F")
    params["duration"] = grab("DUR", str)
    params["coupling"] = grab("COUP", str)
    params["significant_control"] = grab("CTRL", lambda x: x.lower() == "true")

    return params


def remove_tags(text: str) -> str:
    """Rimuove tutti i TAG dalla descrizione finale."""
    text = re.sub(r"\[[a-zA-Z0-9]+:[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
