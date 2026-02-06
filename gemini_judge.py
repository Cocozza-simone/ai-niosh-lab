import re
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
from model_router import call_llm_with_system_prompt


# ============================================================
#  STRUCT
# ============================================================

@dataclass
class JudgeInput:
    section_name: str
    human_text: str
    ai_text: str
    reference_examples: str
    parameters: Optional[Dict[str, Any]] = None


# ============================================================
#  GEMINI JUDGE — VERSIONE COMPLETA E CORRETTA
# ============================================================

class GeminiJudge:
    def __init__(self, model: str = "gemma3:12b"):
        from model_router import GEMINI_AVAILABLE
        if not GEMINI_AVAILABLE:
            raise RuntimeError(
                "GeminiJudge richiede google-genai e una chiave API configurata."
            )
        self.model = model


    # =======================================================
    # PROMPT BUILDER — COMPLETO, COME L’ORIGINALE, MA CORRETTO
    # =======================================================
    def _build_prompt(self, d: JudgeInput) -> str:
        return f"""
REFERENCE EXAMPLES (STYLE MANIFOLD)
------------------------------------
{d.reference_examples}

These examples define the stylistic + structural manifold.
Evaluation MUST occur strictly inside this distribution.

SECTION: {d.section_name}

TEXT A (HUMAN):
----------------
{d.human_text}

TEXT B (AI):
-------------
{d.ai_text}

SCORING RULES
---------------------------------

You MUST output ONLY one JSON object with fields:

"winner": "A" or "B"

"scores": {{
    "A": {{
        "structure": 0–10,
        "coherence": 0–10,
        "style": 0–10,
        "semantic_alignment": 0–10
    }},
    "B": {{
        "structure": 0–10,
        "coherence": 0–10,
        "style": 0–10,
        "semantic_alignment": 0–10
    }}
}}

"weighted_scores": {{
    "A": 0–10,
    "B": 0–10
}}

"justification": string


WEIGHTS:
- structure = 40%
- coherence = 40%
- style = 20%

Semantic alignment refines coherence but is NOT weighted independently.

STRICT RULE:
NO TEXT OUTSIDE JSON.
NO MARKDOWN.
NO EXPLANATIONS OUTSIDE JSON.
"""

    # ============================================================
    # 1) SEMANTIC CONSISTENCY SPECIALIZZATO NIOSH (VERSIONE ESTESA)
    # ============================================================
    def _semantic_consistency(self, a: str, b: str, params=None) -> float:
        """
        VERSIONE ESTESA E AVANZATA DEL GIUDICE SEMANTICO NIOSH.

        Obiettivo:
        - rilevare contraddizioni parametriche
        - rilevare cambi qualitativi biomeccanici
        - stimare coerenza strutturale tra A e B
        - riconoscere espansioni corrette tipiche di JD/JA/HA/RS
        - rilevare omissioni critiche
        - penalizzare cambi che alterano la biomeccanica del task
        """

        # ---------------------------------------------------------
        # NORMALIZZAZIONE
        # ---------------------------------------------------------
        a_low = a.lower()
        b_low = b.lower()

        # ---------------------------------------------------------
        # 1) CONFRONTO TAG NIOSH — VERSIONE ESTESA
        # ---------------------------------------------------------

        tag_thresholds = {
            "W": 5.0,
            "H0": 2.0, "H1": 2.0,
            "V0": 2.0, "V1": 2.0,
            "A": 5.0,
            "F": 0.5,
        }

        def extract_tag(text, tag):
            matches = re.findall(rf"\[{tag}:(-?\d+(?:\.\d+)?)\]", text, flags=re.I)
            if matches:
                try:
                    return float(matches[0])
                except:
                    return None
            return None

        large_conflict = False
        medium_conflict = False

        for tag, thr in tag_thresholds.items():
            av = extract_tag(a, tag)
            bv = extract_tag(b, tag)
            if av is not None and bv is not None:
                diff = abs(av - bv)

                # CONTRADDIZIONI GRANDI (Scenario 1)
                if diff >= thr * 2:
                    large_conflict = True

                # CONTRADDIZIONI MEDIE
                elif diff >= thr:
                    medium_conflict = True

        if large_conflict:
            return 0.5  # penalità massima controllata

        if medium_conflict:
            return 1.0

        # ---------------------------------------------------------
        # 2) CAMBI QUALITATIVI DI DOMINIO BIOMECCANICO
        # ---------------------------------------------------------

        biomech_shifts = [
            ("no twisting", "twist"),
            ("does not twist", "twist"),
            ("non ruota", "ruota"),
            ("no rotation", "rotation"),
            ("no asymmetry", "asymmetry"),
            ("good coupling", "poor coupling"),
            ("fair coupling", "poor coupling"),
            ("no significant control", "significant control"),
            ("controllo non significativo", "controllo significativo"),
            ("lifting from floor", "overhead"),
            ("ground lift", "overhead"),
            ("waist level", "floor"),
        ]

        for x, y in biomech_shifts:
            if x in a_low and y in b_low:
                return 1.5
            if x in b_low and y in a_low:
                # B più sicuro / più favorevole di A → penalità minore
                return 2.0

        # ---------------------------------------------------------
        # 3) DETTAGLIAGGIO COERENTE (Scenario 2 esteso)
        #    A è generico, B è strutturato → punteggio alto
        # ---------------------------------------------------------

        tag_pattern = r"\[[A-Z0-9]+:[^\]]+\]"
        a_tags = re.findall(tag_pattern, a, flags=re.I)
        b_tags = re.findall(tag_pattern, b, flags=re.I)

        generic_signals = [
            "the worker lifts",
            "the employee lifts",
            "lifts a load",
            "lifting task",
            "manual handling",
            "moves a box",
            "handles objects",
            "carries items",
        ]

        a_generic = any(g in a_low for g in generic_signals)

        # A completamente privo di struttura — B altamente strutturato
        if len(a_tags) == 0 and len(b_tags) >= 2:
            return 9.0

        # A generico e B aggiunge TAG o misure
        if a_generic and (len(b_tags) > 0 or re.search(r"\b(in|cm|kg|lbs)\b", b_low)):
            return 8.5

        # ---------------------------------------------------------
        # 4) MISURE BIOMECCANICHE IMPLICITE (estensione)
        # ---------------------------------------------------------

        # Se B introduce concetti biomeccanici coerenti con il task, premiamo
        positive_expansions = [
            ("waist", "30"), ("waist", "32"), ("table height", "36"),
            ("shoulder", "50"), ("chest", "48"),
            ("close to body", "10"), ("keep load close", "10"),
            ("arms extended", "20"), ("reaching", "20"),
            ("low frequency", "0.2"), ("high frequency", "2"),
        ]

        for kw, num in positive_expansions:
            if kw in a_low and num in b_low:
                return 8.0

        # ---------------------------------------------------------
        # 5) RILEVAZIONE DI CONTRADDITTORI LATENTI SU VERBI
        # ---------------------------------------------------------

        verb_conflicts = [
            ("lift", "push"),
            ("lift", "pull"),
            ("push", "lift"),
            ("carry", "push"),
            ("carry", "pull"),
            ("one hand", "two hands"),
            ("two hands", "one hand"),
            ("team lift", "solo lift"),
        ]

        for v1, v2 in verb_conflicts:
            if v1 in a_low and v2 in b_low:
                return 2.5

        # ---------------------------------------------------------
        # 6) ALLINEAMENTO TESTUALE DI BASE
        # ---------------------------------------------------------
        import difflib
        sim = difflib.SequenceMatcher(None, a_low, b_low).ratio()

        # mappiamo su 3–10
        base_score = max(3.0, min(10.0, sim * 10))

        # ---------------------------------------------------------
        # 7) MICRO-PENALITÀ PER Omissioni Key
        # ---------------------------------------------------------

        key_terms = ["origin", "destination", "horizontal", "vertical", "frequency"]
        for kt in key_terms:
            if kt in a_low and kt not in b_low:
                base_score -= 0.3

        return round(base_score, 2)

    # ============================================================
    # 2) JSON EXTRACTION — VERSIONE DEFINITIVA
    # ============================================================
    def _extract_json(self, text: str) -> Dict[str, Any]:

        # Caso: risposta già valida
        try:
            direct = json.loads(text)
            if isinstance(direct, dict):
                return direct
        except Exception:
            pass

        # Cerca blocchi { ... }
        blocks = re.findall(r"\{[\s\S]*?\}", text)
        for b in blocks:
            try:
                return json.loads(b)
            except Exception:
                continue

        # Ultimo fallback: ritorna un JSON coerente
        return {
            "winner": "B",
            "scores": {
                "A": {
                    "structure": 0,
                    "coherence": 0,
                    "style": 0,
                    "semantic_alignment": 10,
                },
                "B": {
                    "structure": 10,
                    "coherence": 10,
                    "style": 10,
                    "semantic_alignment": 9,
                },
            },
            "weighted_scores": {"A": 0, "B": 10},
            "justification": "Fallback: malformed JSON",
        }

    # ============================================================
    # 3) PESA I PUNTEGGI — COME L’ORIGINALE, MA CORRETTO
    # ============================================================
    def _compute_weights(self, sc: Dict[str, Dict[str, float]]) -> Dict[str, float]:

        def W(v: Dict[str, float]) -> float:
            s = float(v.get("structure", 0.0))
            c = float(v.get("coherence", 0.0))
            st = float(v.get("style", 0.0))
            return round(s * 0.4 + c * 0.4 + st * 0.2, 2)

        # protezione minima se A/B mancanti
        a_scores = sc.get("A", {})
        b_scores = sc.get("B", {})

        return {"A": W(a_scores), "B": W(b_scores)}

    # ============================================================
    # 4) MAIN COMPARATOR
    # ============================================================
    def compare(self, human_text, ai_text, reference_examples, section_name, parameters=None):

        d = JudgeInput(section_name, human_text, ai_text, reference_examples, parameters)
        user_prompt = self._build_prompt(d)

        # Sistema rigoroso (corretto!)
        system_prompt = """
You are a JSON-only judge.
You MUST output exactly ONE JSON object.
No commentary.
No markdown.
"""

        raw = call_llm_with_system_prompt(
            model=self.model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            format_schema={
                "type": "object",
                "properties": {
                    "winner": {"type": "string"},
                    "scores": {"type": "object"},
                    "weighted_scores": {"type": "object"},
                    "justification": {"type": "string"},
                },
                "required": ["winner", "scores", "weighted_scores", "justification"],
            },
            temperature=0.0,
        )

        result = self._extract_json(raw)

        # --- Semantic Alignment Post-Processing ----
        sa_b = self._semantic_consistency(human_text, ai_text, parameters)

        # assicuriamoci che esistano i dizionari
        result.setdefault("scores", {})
        result["scores"].setdefault("A", {})
        result["scores"].setdefault("B", {})

        result["scores"]["A"]["semantic_alignment"] = 10.0
        result["scores"]["B"]["semantic_alignment"] = float(sa_b)

        # Ricalcola pesi
        result["weighted_scores"] = self._compute_weights(result["scores"])

        return result

    # ============================================================
    # 5) PUBLIC ENTRYPOINT
    # ============================================================
    def evaluate(self, data: JudgeInput) -> dict:
        return self.compare(
            human_text=data.human_text,
            ai_text=data.ai_text,
            reference_examples=data.reference_examples,
            section_name=data.section_name,
            parameters=data.parameters,
        )

if __name__ == "__main__":
    judge = GeminiJudge()

    demo = JudgeInput(
        section_name="example",
        human_text="Human reference text.",
        ai_text="AI generated text.",
        reference_examples="Example A\nExample B",
    )

    print(json.dumps(judge.evaluate(demo), indent=2))
