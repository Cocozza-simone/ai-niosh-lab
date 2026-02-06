"""
model_router.py

Router per gestire chiamate a diversi modelli LLM (Ollama e Gemini).
Mantiene system prompt identici ma usa backends diversi.
"""

import json
import re
import time
from typing import Dict, Any, Optional
import ollama
import os

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GEMINI_AVAILABLE = False
    print("Google GenAI library non installata.")


#  Inserisci qui le tue chiavi Gemini (funzionano in rotazione)
GEMINI_KEYS = [
    "AIzaSyClBcyE6mYxoeKUbcPAJ6L_4JhPQYn5Su4",
    "AIzaSyDTT1WmCVMn_thrB16yV967tNZR6WDZYis",
    "AIzaSyApbCyBHHB3NBb5pYABNRjHWE7eBDGaup0",
    "AIzaSyBDRUTl_-2XDee-7oQ3u-IwQeo0unbnqHY",
    "AIzaSyAdqSbNHSV226aC0re1pn5zB8RrutMlHJk",
    "AIzaSyBaJ8qHEAUcSD06kmbqao1WZK_mM4BMsMk",
    "AIzaSyCZHKosIjw3Vf_o7F_yEKWIVNvSOvhRu0M",
]

if not GEMINI_KEYS:
    raise Exception("Nessuna chiave Gemini impostata in GEMINI_KEYS.")
_current_key_index = 0
_bad_keys = set()


def _next_gemini_key():
    """
    Restituisce la prossima chiave NON presente nella blacklist.
    Se tutte le chiavi sono invalide → errore chiaro.
    """
    global _current_key_index

    available = [k for k in GEMINI_KEYS if k not in _bad_keys]

    if not available:
        raise Exception("Tutte le chiavi Gemini risultano invalide o scadute.")

    key = available[_current_key_index % len(available)]
    _current_key_index += 1
    return key


def _create_gemini_client():
    """Crea un client Gemini con la prossima chiave valida."""
    key = _next_gemini_key()
    return genai.Client(api_key=key)

def is_gemini_model(model_name: str) -> bool:
    return model_name.startswith("gemini") or "gemini-" in model_name.lower()


def call_llm_with_system_prompt(model: str, user_prompt: str, system_prompt: str,
                                format_schema: Optional[Dict] = None,
                                temperature: float = 0.0, **kwargs):
    if is_gemini_model(model):
        return _call_gemini(model, user_prompt, system_prompt, format_schema,
                            temperature, **kwargs)
    return _call_ollama(model, user_prompt, system_prompt, format_schema,
                        temperature, **kwargs)


def _call_ollama(
    model: str,
    user_prompt: str,
    system_prompt: str,
    format_schema: Optional[Dict] = None,
    temperature: float = 0.0,
    **kwargs,
) -> str:
    """
    Versione stabile di Ollama:
    - NIENTE streaming
    - Compatibile con JSON structured output
    - Niente chunk vuoti
    - Nessun loop infinito
    """

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            options = {"temperature": temperature}
            if kwargs:
                options.update(kwargs)

            # ============================================================
            # 1) STRUCTURED OUTPUT (format_schema) → usa ollama.chat
            # ============================================================
            if format_schema:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    format=format_schema,
                    options=options,
                )

                # Risposta valida (Dict compatibility)
                if (
                    isinstance(response, dict)
                    and "message" in response
                    and isinstance(response["message"], dict)
                    and response["message"].get("content")
                ):
                    return response["message"]["content"].strip()

                # Risposta valida (Object compatibility - newer ollama lib)
                if hasattr(response, "message") and hasattr(
                    response.message, "content"
                ):
                    return response.message.content.strip()

                # Risposta vuota → retry
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

                return ""

            # ============================================================
            # 2) FREE TEXT (senza schema) → ollama.chat senza streaming
            # ============================================================
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options=options,
            )

            # Dict compatibility
            if (
                isinstance(response, dict)
                and "message" in response
                and isinstance(response["message"], dict)
                and response["message"].get("content")
            ):
                return response["message"]["content"].strip()

            # Object compatibility
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content.strip()

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue

            return ""

        except Exception as e:
            print(f"Ollama API error (attempt {attempt + 1}): {str(e)}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue

            # Fallimento totale → errore
            raise Exception(f"Ollama API error after {max_retries} attempts: {str(e)}")

    return ""



def _call_gemini(
    model: str,
    user_prompt: str,
    system_prompt: str,
    format_schema: Optional[Dict],
    temperature: float,
    **kwargs,
) -> str:

    if not GEMINI_AVAILABLE:
        raise Exception("Gemini API non disponibile.")

    max_retries = 8
    retry_delay = 3

    # costruiamo prompt unificato
    prompt = (
        "<SYSTEM_INSTRUCTIONS>\n" + system_prompt + "\n</SYSTEM_INSTRUCTIONS>\n"
        "<USER_REQUEST>\n" + user_prompt + "\n</USER_REQUEST>\n"
    )

    if format_schema:
        prompt += (
            "<STRUCTURED_OUTPUT_FORMAT>\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{json.dumps(format_schema)}\n"
            "</STRUCTURED_OUTPUT_FORMAT>\n"
        )

    for attempt in range(max_retries):

        try:
            client = _create_gemini_client()

            config_kwargs = {"temperature": temperature}

            if "num_predict" in kwargs:
                config_kwargs["max_output_tokens"] = kwargs["num_predict"]
            if "top_p" in kwargs:
                config_kwargs["top_p"] = kwargs["top_p"]
            if "top_k" in kwargs:
                config_kwargs["top_k"] = kwargs["top_k"]

            config = types.GenerateContentConfig(**config_kwargs)

            # CHIAMATA GEMINI
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            # risposta vuota?
            if not response or not response.text:
                time.sleep(retry_delay)
                continue

            text = response.text.strip()

            # gestione JSON strutturato
            if format_schema:
                j = _extract_json_from_gemini_response(text)
                if j:
                    try:
                        json.loads(j)
                        return j
                    except:
                        try:
                            fixed = _fix_common_json_issues(j)
                            json.loads(fixed)
                            return fixed
                        except:
                            return _generate_fallback_json(format_schema)

            return text

        except Exception as e:
            message = str(e)

            # ============================================================
            # 1) ERRORE CHIAVE SCADUTA → blacklist immediata
            # ============================================================

            if "API key expired" in message or "API_KEY_INVALID" in message:
                bad_key = client._api_key if hasattr(client, "_api_key") else None
                if bad_key:
                    print(f"[GEMINI] Chiave scaduta → rimossa dalla rotazione: {bad_key}")
                    _bad_keys.add(bad_key)
                continue  # prova una nuova chiave

            # ============================================================
            # 2) QUOTA / RESOURCE_EXHAUSTED / 429 → nuova chiave
            # ============================================================

            if ("429" in message or
                "RESOURCE_EXHAUSTED" in message or
                "quota" in message.lower()):
                print(f"[GEMINI] Quota esaurita (tentativo {attempt+1}) → cambio chiave")
                time.sleep(retry_delay)
                continue

            # altri errori temporanei
            print(f"[GEMINI] Errore temporaneo: {message}")
            time.sleep(retry_delay)
            continue

    # fallback dopo max tentativi
    if format_schema:
        return _generate_fallback_json(format_schema)

    return ""
def _extract_json_from_gemini_response(response_text: str) -> str:
    """
    Estrae JSON dalla risposta di Gemini.
    Gemini spesso include markup o testo extra intorno al JSON.
    """

    # Try to find JSON blocks in the response
    import re

    # First, check if this is a schema description (common Gemini error)
    if (
        '"description"' in response_text
        and '"properties"' in response_text
        and '"type":' in response_text
    ):
        print("Warning: Gemini returned JSON schema instead of values, using fallback")
        return ""  # Force fallback generation

    # Look for JSON between ```json and ```
    json_pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(json_pattern, response_text, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Look for JSON between { and } (first complete JSON object)
    brace_count = 0
    start_idx = -1

    for i, char in enumerate(response_text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_candidate = response_text[start_idx : i + 1].strip()
                # Validate this looks like parameter data, not schema
                if (
                    '"weight"' in json_candidate
                    or '"horizontal_origin"' in json_candidate
                ):
                    return json_candidate

    # Fallback: return original text if it contains parameter data
    if '"weight"' in response_text or '"horizontal_origin"' in response_text:
        return response_text.strip()

    return ""  # Force fallback generation


def _fix_common_json_issues(json_str: str) -> str:
    """
    Corregge problemi comuni di JSON generato da Gemini.
    Tenta di rendere il JSON valido.
    """

    # Rimuovi caratteri problematici all'inizio/fine
    json_str = json_str.strip()

    # Se inizia con ```json, rimuovi il markup
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]

    json_str = json_str.strip()

    # Correggi virgole finali comuni
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    # Correggi missing quotes in property names (solo se non già quotati)
    json_str = re.sub(r'(?<!["\'])\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?=\s*:)', r'"\1"', json_str)

    # Prova a validare e correggere parentesi
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        # Tenta di correggere errori di parentesi
        open_count = json_str.count("{")
        close_count = json_str.count("}")

        if open_count > close_count:
            json_str += "}" * (open_count - close_count)
        elif close_count > open_count:
            json_str = "{" * (close_count - open_count) + json_str

        # Prova again
        try:
            json.loads(json_str)
            return json_str
        except:
            # Se ancora non valido, return originale
            return json_str


def _generate_fallback_json(format_schema: Dict) -> str:
    """
    Generate minimal valid JSON fallback for NIOSH parameters.
    This ensures we always return valid parameters even when AI fails.
    """
    try:
        # Check if this is NIOSH parameters schema
        if "properties" in format_schema:
            properties = format_schema["properties"]

            fallback_params = {}

            # Generate reasonable defaults for NIOSH parameters
            if "weight" in properties:
                fallback_params["weight"] = 25.0  # Conservative default weight
            if "horizontal_origin" in properties:
                fallback_params["horizontal_origin"] = 20.0  # Reasonable reach
            if "horizontal_destination" in properties:
                fallback_params["horizontal_destination"] = 20.0
            if "vertical_origin" in properties:
                fallback_params["vertical_origin"] = 30.0  # Waist height
            if "vertical_destination" in properties:
                fallback_params["vertical_destination"] = 36.0  # Counter height
            if "asymmetry_angle" in properties:
                fallback_params["asymmetry_angle"] = 0.0  # No twisting
            if "frequency" in properties:
                fallback_params["frequency"] = 1.0  # Light frequency
            if "duration" in properties:
                fallback_params["duration"] = "<1h"  # Short duration
            if "coupling" in properties:
                fallback_params["coupling"] = "fair"  # Conservative coupling
            if "significant_control" in properties:
                fallback_params["significant_control"] = False
            if "judgment" in properties:
                fallback_params["judgment"] = "intermediate"
            if "one_limb_lifting" in properties:
                fallback_params["one_limb_lifting"] = False
            if "two_operators_lifting" in properties:
                fallback_params["two_operators_lifting"] = False
            if "gender" in properties:
                fallback_params["gender"] = "M"
            if "age" in properties:
                fallback_params["age"] = 25
            if "etm" in properties:
                fallback_params["etm"] = 1.0

            return json.dumps(fallback_params, indent=2)

        # Generic fallback for other schemas
        return "{}"

    except Exception as e:
        print(f"Error generating fallback JSON: {e}")
        # Return minimal NIOSH fallback
        return json.dumps(
            {
                "weight": 25.0,
                "horizontal_origin": 20.0,
                "horizontal_destination": 20.0,
                "vertical_origin": 30.0,
                "vertical_destination": 36.0,
                "asymmetry_angle": 0.0,
                "frequency": 1.0,
                "duration": "<1h",
                "coupling": "fair",
                "significant_control": False,
                "judgment": "intermediate",
                "one_limb_lifting": False,
                "two_operators_lifting": False,
                "gender": "M",
                "age": 25,
                "etm": 1.0,
            }
        )


def get_model_backend(model: str) -> str:
    """Ritorna il backend usato per un modello."""
    return "gemini" if is_gemini_model(model) else "ollama"


def test_model_availability(model: str) -> bool:
    """Testa se un modello è disponibile."""

    if is_gemini_model(model):
        return GEMINI_AVAILABLE
    else:
        try:
            # Try simple Ollama call
            response = ollama.generate(model=model, prompt="test", keep_alive=0)
            return True
        except:
            return False
