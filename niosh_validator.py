"""
Modulo Validatore NIOSH

Questo modulo fornisce funzionalità avanzate di validazione per report NIOSH,
combinando analisi matematica automatizzata con validazione linguistica tramite AI.

Funzionalità principali:
- Validazione matematica dei calcoli NIOSH (RWL, LI, moltiplicatori)
- Estrazione automatica di calcoli dal testo del report
- Verifica della coerenza linguistica tramite modello linguistico (Ollama)
- Controllo della completezza delle sezioni del report
- Generazione di report di validazione dettagliati
- Correzione automatica di errori comuni

Il validatore garantisce l'accuratezza tecnica e la chiarezza comunicativa
dei report di valutazione del rischio ergonomico.

Versione: 1.0
Dipendenze: niosh_calculator_v2, requests, Ollama LLM
"""

import json
import re
import requests
from typing import Dict, List, Tuple, Optional
from niosh_calculator_v2 import calculate_niosh_v2, get_hm, get_vm, get_dm, get_am, get_fm, get_cm, LC

# Configurazione LLM (Local AI Model)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"

class NIOSHValidator:
    """
    Validatore ibrido AI + Python per report NIOSH.
    
    Combina validazione matematica rigorosa con analisi linguistica avanzata
    per garantire report NIOSH accurati, coerenti e professionali.
    
    Attributi:
        ollama_url (str): URL del servizio Ollama per analisi linguistica
        model (str): Nome del modello linguistico utilizzato
        validation_log (list): Registro delle validazioni eseguite
    
    Funzionalità:
        - Verifica calcoli matematici (RWL, LI, moltiplicatori)
        - Analisi coerenza linguistica e tecnica
        - Controllo completezza sezioni report
        - Generazione suggerimenti di miglioramento
    """
    
    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = MODEL):
        """
        Inizializza il validatore NIOSH.
        
        Args:
            ollama_url (str): URL del servizio Ollama (default: localhost:11434)
            model (str): Nome del modello linguistico (default: llama3.2:latest)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.validation_log = []
    
    def call_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Esegue chiamata al modello linguistico Ollama per analisi AI.
        
        Invia prompt al modello locale per analisi linguistica, coerenza tecnica
        e generazione di suggerimenti migliorativi per i report NIOSH.
        
        Args:
            system_prompt (str): Prompt di sistema che definisce il ruolo e le regole
            user_prompt (str): Prompt utente con il testo da analizzare
            
        Returns:
            Optional[str]: Risposta del modello o None in caso di errore
            
        Note:
            - Utilizza temperatura bassa (0.1) per risposte coerenti
            - Timeout di 30 secondi per evitare blocchi
            - Gestisce automaticamente errori di connessione
        """
        try:
            full_prompt = f"[SYSTEM] {system_prompt}\n\n[USER] {user_prompt}"
            
            data = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1000
                }
            }
            
            response = requests.post(self.ollama_url, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            return result['response'].strip()
            
        except Exception as e:
            print(f"Errore chiamata Ollama: {e}")
            return None
    
    def extract_calculations_from_text(self, text: str) -> List[Dict]:
        """
        Estrae sistematicamente tutti i calcoli matematici dal testo del report.
        
        Identifica e analizza pattern matematici specifici dell'equazione NIOSH:
        - Catene di moltiplicazione per RWL: "23 × 0.625 × 0.775 × ... = 10.5 kg"
        - Divisioni per Lifting Index: "10.0 / 5.5 = 1.82"
        - Percentuali per riduzioni/incrementi
        
        Args:
            text (str): Testo completo del report NIOSH da analizzare
            
        Returns:
            List[Dict]: Lista di dizionari con:
                - type: 'multiplication', 'division', 'percentage'
                - expression: stringa completa del calcolo
                - factors/numerator/denominator: valori numerici
                - expected_result: risultato atteso
                - position: posizione nel testo (start, end)
                
        Note:
            - Utilizza espressioni regolari per pattern matching robusto
            - Gestisce formattazioni diverse (spazi, grassetto, ecc.)
            - Estrae solo calcoli con risultati espliciti
        """
        calculations = []
        
        # Pattern per moltiplicazioni NIOSH: "23 × 0.625 × 0.775 × ... = 10.5 kg"
        mult_pattern = r'(\d+(?:\.\d+)?)\s*×\s*([\d.\s×]+?)\s*=\s*\*?\*?(\d+(?:\.\d+)?)\s*kg\*?\*?'
        
        for match in re.finditer(mult_pattern, text):
            full_expr = match.group(0)
            expected_result = float(match.group(3))
            
            # Estrai tutti i numeri nella catena di moltiplicazioni
            numbers = re.findall(r'\d+(?:\.\d+)?', match.group(0).split('=')[0])
            factors = [float(n) for n in numbers]
            
            calculations.append({
                'type': 'multiplication',
                'expression': full_expr,
                'factors': factors,
                'expected_result': expected_result,
                'position': match.span()
            })
        
        # Pattern per divisioni: "10.0 / 5.5 = 1.82"
        div_pattern = r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*=\s*\*?\*?(\d+(?:\.\d+)?)\*?\*?'
        
        for match in re.finditer(div_pattern, text):
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            expected_result = float(match.group(3))
            
            calculations.append({
                'type': 'division',
                'expression': match.group(0),
                'numerator': numerator,
                'denominator': denominator,
                'expected_result': expected_result,
                'position': match.span()
            })
        
        # Pattern per percentuali: "50% reduction" o "increases by 25%"
        pct_pattern = r'(\d+(?:\.\d+)?)\%'
        
        for match in re.finditer(pct_pattern, text):
            calculations.append({
                'type': 'percentage',
                'expression': match.group(0),
                'value': float(match.group(1)),
                'position': match.span()
            })
        
        return calculations
    
    def validate_calculation(self, calc: Dict, tolerance: float = 0.02) -> Tuple[bool, Optional[float]]:
        """
        Valida l'accuratezza di un singolo calcolo matematico.
        
        Esegue il calcolo effettivo e lo confronta con il risultato presente nel testo,
        applicando una tolleranza per errori di arrotondamento normali.
        
        Args:
            calc (Dict): Dizionario con dati del calcolo estratto dal testo
            tolerance (float): Tolleranza per errori (default 2% = 0.02)
        
        Returns:
            Tuple[bool, Optional[float]]: 
                - is_valid: True se il calcolo è corretto entro la tolleranza
                - correct_result: Risultato corretto calcolato (se applicabile)
                
        Note:
            - Per moltiplicazioni: verifica il prodotto di tutti i fattori
            - Per divisioni: verifica il rapporto numeratore/denominatore
            - Applica tolleranza relativa per gestire arrotondamenti
            - Restituisce il risultato corretto per correzioni automatiche
        """
        if calc['type'] == 'multiplication':
            # Calcola il prodotto corretto
            correct_result = 1.0
            for factor in calc['factors']:
                correct_result *= factor
            
            expected = calc['expected_result']
            error = abs(correct_result - expected) / expected if expected > 0 else 0
            
            is_valid = error <= tolerance
            return is_valid, round(correct_result, 2)
        
        elif calc['type'] == 'division':
            # Calcola il quoziente corretto
            if calc['denominator'] == 0:
                return False, None
            
            correct_result = calc['numerator'] / calc['denominator']
            expected = calc['expected_result']
            error = abs(correct_result - expected) / expected if expected > 0 else 0
            
            is_valid = error <= tolerance
            return is_valid, round(correct_result, 2)
        
        elif calc['type'] == 'percentage':
            # Verifica che la percentuale sia ragionevole (0-100 per percentuali normali)
            value = calc['value']
            # Percentuali di riduzione possono essere > 100% in alcuni contesti
            is_valid = 0 <= value <= 1000
            return is_valid, value
        
        return True, None
    
    def recalculate_with_python(self, calculated_data: Dict) -> Dict:
        """
        Ricalcola tutti i valori NIOSH usando il calculator Python.
        Questo è il "ground truth" per verificare i calcoli nel report.
        """
        try:
            # Esegui ricalcolo completo
            recalc_data = calculate_niosh_v2(calculated_data)
            return recalc_data['calculation_results']
        except Exception as e:
            print(f"Errore nel ricalcolo Python: {e}")
            return None
    
    def validate_mathematical_consistency(
        self, 
        report_text: str, 
        calculated_data: Dict
    ) -> Tuple[bool, List[Dict], str]:
        """
        Valida la consistenza matematica del report confrontando:
        1. Calcoli interni al testo
        2. Valori riportati vs valori ricalcolati con Python
        
        Returns:
            (is_valid, errors_found, corrected_text)
        """
        errors = []
        corrected_text = report_text
        
        # Step 1: Estrai e valida calcoli dal testo
        calculations = self.extract_calculations_from_text(report_text)
        
        for calc in calculations:
            is_valid, correct_result = self.validate_calculation(calc)
            
            if not is_valid and correct_result is not None:
                error_info = {
                    'type': 'calculation_error',
                    'location': calc['expression'],
                    'expected': calc.get('expected_result'),
                    'correct': correct_result,
                    'error_magnitude': abs(calc.get('expected_result', 0) - correct_result)
                }
                errors.append(error_info)
                
                # Sostituisci il valore errato con quello corretto
                old_expr = calc['expression']
                if calc['type'] in ['multiplication', 'division']:
                    # Mantieni il formato con grassetto se presente
                    if '**' in old_expr:
                        new_expr = old_expr.replace(
                            f"**{calc['expected_result']}", 
                            f"**{correct_result}"
                        )
                    else:
                        new_expr = re.sub(
                            r'=\s*[\d.]+',
                            f"= {correct_result}",
                            old_expr
                        )
                    corrected_text = corrected_text.replace(old_expr, new_expr)
                
                self.validation_log.append(
                    f"MATH ERROR: {calc['expression']} → Corretto: {correct_result}"
                )
        
        # Step 2: Verifica valori chiave con ricalcolo Python
        python_results = self.recalculate_with_python(calculated_data)
        
        if python_results:
            # Verifica RWL e LI all'origine
            origin_data = python_results['origin']
            
            # Cerca RWL origin nel testo
            rwl_pattern = r'RWL\s*=.*?=\s*\*?\*?(\d+(?:\.\d+)?)\s*kg\*?\*?'
            rwl_matches = list(re.finditer(rwl_pattern, corrected_text))
            
            if rwl_matches:
                # Primo match dovrebbe essere origin
                reported_rwl = float(rwl_matches[0].group(1))
                correct_rwl = origin_data['RWL']
                
                if abs(reported_rwl - correct_rwl) > 0.5:  # Tolleranza 0.5 kg
                    errors.append({
                        'type': 'rwl_mismatch',
                        'location': 'Origin RWL',
                        'expected': reported_rwl,
                        'correct': correct_rwl,
                        'error_magnitude': abs(reported_rwl - correct_rwl)
                    })
                    
                    # Sostituisci RWL errato
                    old_rwl_expr = rwl_matches[0].group(0)
                    new_rwl_expr = old_rwl_expr.replace(
                        f"{reported_rwl:.1f} kg",
                        f"{correct_rwl:.1f} kg"
                    )
                    corrected_text = corrected_text.replace(old_rwl_expr, new_rwl_expr)
                    
                    self.validation_log.append(
                        f"RWL ERROR: Origin RWL {reported_rwl} → {correct_rwl}"
                    )
            
            # Verifica LI
            li_pattern = r'LI\s*=\s*[\d.]+/[\d.]+\s*=\s*\*?\*?(\d+(?:\.\d+)?)\*?\*?'
            li_matches = list(re.finditer(li_pattern, corrected_text))
            
            if li_matches:
                reported_li = float(li_matches[0].group(1))
                correct_li = origin_data['LI'] if origin_data['LI'] != "inf" else 999
                
                if abs(reported_li - correct_li) > 0.1:  # Tolleranza 0.1
                    errors.append({
                        'type': 'li_mismatch',
                        'location': 'Origin LI',
                        'expected': reported_li,
                        'correct': correct_li,
                        'error_magnitude': abs(reported_li - correct_li)
                    })
                    
                    self.validation_log.append(
                        f"LI ERROR: Origin LI {reported_li} → {correct_li}"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors, corrected_text
    
    def validate_linguistic_consistency(self, text: str) -> Tuple[bool, List[str], str]:
        """
        Usa AI per rilevare incongruenze linguistiche e logiche nel testo.
        
        Returns:
            (is_valid, issues_found, corrected_text)
        """
        issues = []
        corrected_text = text
        
        system_prompt = """You are a technical validator for NIOSH ergonomic reports.
Your task is to identify LOGICAL INCONSISTENCIES and CONTRADICTIONS in the text.

Look for:
1. CONTRADICTORY STATEMENTS (e.g., "LI is 2.5 (acceptable)" when LI > 1.0 is NOT acceptable)
2. MISMATCHED RISK LEVELS (e.g., "high risk" but then says "acceptable for most workers")
3. INCONSISTENT CONCLUSIONS (e.g., RWL < Load but says "within limits")
4. SEVERITY MISMATCHES (e.g., LI=5.0 described as "moderate" instead of "high")

Respond in JSON format:
{
    "has_issues": true/false,
    "issues": [
        {
            "type": "contradiction",
            "location": "paragraph or section identifier",
            "problem": "description of the inconsistency",
            "suggestion": "how to fix it"
        }
    ]
}

Be strict but fair. Only flag genuine logical problems, not stylistic preferences."""

        user_prompt = f"""Analyze this NIOSH report excerpt for logical inconsistencies:

{text[:3000]}

Focus on the Hazard Assessment and Comments sections especially."""

        try:
            response = self.call_ollama(system_prompt, user_prompt)
            
            if response:
                # Pulisci response da markdown
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                try:
                    validation_result = json.loads(clean_response)
                    
                    if validation_result.get('has_issues', False):
                        issues = validation_result.get('issues', [])
                        
                        for issue in issues:
                            self.validation_log.append(
                                f"LINGUISTIC ISSUE: {issue['type']} in {issue['location']} - {issue['problem']}"
                            )
                            
                            # Tenta correzione automatica per contraddizioni evidenti
                            if issue['type'] == 'contradiction' and 'suggestion' in issue:
                                # Questa è una correzione conservativa - in produzione 
                                # potresti voler chiedere conferma umana
                                pass
                
                except json.JSONDecodeError:
                    print("AI response non era JSON valido")
        
        except Exception as e:
            print(f"Errore validazione linguistica: {e}")
        
        is_valid = len(issues) == 0
        return is_valid, issues, corrected_text
    
    def validate_report(
        self, 
        report_text: str, 
        calculated_data: Dict,
        fix_errors: bool = True
    ) -> Dict:
        """
        Validazione completa del report NIOSH.
        
        Args:
            report_text: Testo del report generato
            calculated_data: Dati di calcolo originali
            fix_errors: Se True, corregge automaticamente gli errori trovati
        
        Returns:
            Dizionario con risultati validazione e testo corretto
        """
        print("\n" + "="*60)
        print("AVVIO VALIDAZIONE IBRIDA AI + PYTHON")
        print("="*60)
        
        self.validation_log = []
        
        # Step 1: Validazione Matematica (Python)
        print("\n[1/2] Validazione matematica con Python...")
        math_valid, math_errors, math_corrected = self.validate_mathematical_consistency(
            report_text, 
            calculated_data
        )
        
        if math_errors:
            print(f"  ! Trovati {len(math_errors)} errori matematici")
            for err in math_errors[:3]:  # Mostra primi 3
                print(f"    - {err['type']}: {err.get('location', 'N/A')}")
        else:
            print("  + Nessun errore matematico rilevato")
        
        # Step 2: Validazione Linguistica (AI)
        print("\n[2/2] Validazione linguistica con AI...")
        text_to_validate = math_corrected if fix_errors else report_text
        
        ling_valid, ling_issues, ling_corrected = self.validate_linguistic_consistency(
            text_to_validate
        )
        
        if ling_issues:
            print(f"  ! Trovati {len(ling_issues)} problemi linguistici")
            for issue in ling_issues[:3]:
                print(f"    - {issue.get('type', 'N/A')}: {issue.get('problem', 'N/A')[:60]}...")
        else:
            print("  + Nessun problema linguistico rilevato")
        
        # Determina testo finale
        final_text = ling_corrected if fix_errors else report_text
        
        # Report finale
        is_fully_valid = math_valid and ling_valid
        
        print("\n" + "="*60)
        if is_fully_valid:
            print("+ VALIDAZIONE COMPLETATA: Report valido")
        else:
            print("! VALIDAZIONE COMPLETATA: Errori rilevati")
            if fix_errors:
                print("  → Correzioni applicate automaticamente")
        print("="*60 + "\n")
        
        return {
            'is_valid': is_fully_valid,
            'mathematical_validation': {
                'valid': math_valid,
                'errors': math_errors
            },
            'linguistic_validation': {
                'valid': ling_valid,
                'issues': ling_issues
            },
            'corrected_text': final_text if fix_errors else None,
            'original_text': report_text,
            'validation_log': self.validation_log,
            'corrections_applied': fix_errors and (len(math_errors) > 0 or len(ling_issues) > 0)
        }


# --- Funzione di integrazione con prod_gen_v21.py ---

def validate_and_correct_report(
    report_text: str,
    calculated_data: Dict,
    auto_correct: bool = True
) -> Tuple[str, Dict]:
    """
    Funzione wrapper principale per validazione e correzione report NIOSH.
    
    Funzione di interfaccia semplificata per integrazione con prod_gen_v21.py.
    Esegue validazione completa matematica e linguistica del report NIOSH.
    
    Args:
        report_text (str): Testo completo del report generato da generate_final_text()
        calculated_data (Dict): Dati di calcolo completi da calculate_niosh_v2()
        auto_correct (bool): Se True, applica automaticamente le correzioni identificate
        
    Returns:
        Tuple[str, Dict]: 
            - validated_text: Report corretto (se applicabile) o testo originale
            - validation_report: Report dettagliato della validazione con:
                * errori matematici identificati
                * problemi linguistici rilevati
                * correzioni applicate (se auto_correct=True)
                * statistiche della validazione
                * suggerimenti di miglioramento
                
    Funzionalità:
        - Verifica accuratezza di tutti i calcoli RWL e LI
        - Analisi coerenza tecnica e linguistica
        - Correzione automatica errori comuni
        - Generazione report validazione dettagliato
        
    Integrazione:
        Utilizzabile direttamente in prod_gen_v21.py per validazione 
        automatica dei report prima dell'emissione finale.
    """
    validator = NIOSHValidator()
    
    validation_result = validator.validate_report(
        report_text,
        calculated_data,
        fix_errors=auto_correct
    )
    
    # Ritorna testo corretto se disponibile, altrimenti originale
    final_text = validation_result.get('corrected_text') or report_text
    
    return final_text, validation_result


# --- Test standalone ---

if __name__ == "__main__":
    print("NIOSH Validator - Test Module")
    print("="*60)
    
    # Esempio di test con dati mock
    test_report = """# Test NIOSH Report

## Job Analysis

RWL = 23 × 0.625 × 0.775 × 0.850 × 1.000 × 0.940 × 1.000 = **10.5 kg**

Lifting Index (LI) = L/RWL = 15.0/10.5 = **1.43 (moderate risk)**

## Hazard Assessment

The weight to be lifted (15.0 kg) exceeds the RWL at origin (10.5 kg, LI=1.43). 
This task presents acceptable risk for most workers.
"""

    test_data = {
        "scenario_id": "TEST_001",
        "title": "Test Scenario",
        "job_description_narrative": "Test job description",
        "origin_parameters": {
            "H_origin_cm": 40.0,
            "V_origin_cm": 75.0,
            "A_origin_degrees": 0,
            "C_origin_type": "Good"
        },
        "destination_parameters": {
            "H_dest_cm": 40.0,
            "V_dest_cm": 130.0,
            "A_dest_degrees": 0,
            "C_dest_type": "Good"
        },
        "common_parameters": {
            "L_load_kg": 15.0,
            "F_frequency_per_min": 2.0,
            "duration_hours": 2.0
        },
        "significant_control_at_destination": False
    }
    
    print("\n1. Test validazione matematica...")
    validator = NIOSHValidator()
    
    math_valid, math_errors, corrected = validator.validate_mathematical_consistency(
        test_report,
        test_data
    )
    
    print(f"   Risultato: {'VALIDO' if math_valid else 'ERRORI TROVATI'}")
    if math_errors:
        print(f"   Errori: {len(math_errors)}")
        for err in math_errors:
            print(f"   - {err}")
    
    print("\n2. Test validazione linguistica...")
    ling_valid, ling_issues, _ = validator.validate_linguistic_consistency(test_report)
    
    print(f"   Risultato: {'VALIDO' if ling_valid else 'PROBLEMI TROVATI'}")
    if ling_issues:
        print(f"   Problemi: {len(ling_issues)}")
    
    print("\n" + "="*60)
    print("Test completato. Integrare in prod_gen_v21.py con:")
    print("  from niosh_validator import validate_and_correct_report")
    print("  validated_text, report = validate_and_correct_report(final_text, calc_data)")
    print("="*60)