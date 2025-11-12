"""
JSON Schema e Validazione Dati per Generatore NIOSH
Definisce le strutture dati e gestisce la validazione
"""

import json
import warnings
import jsonschema
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from sympy import false


class NioshSchemas:
    """Schema JSON per validazione dati NIOSH"""
    
    # Schema per scenario generato da LLM
    SCENARIO_SCHEMA = {
        "type": "object",
        "required": ["job_title", "job_description", "task_variables"],
        "properties": {
            "job_title": {
                "type": "string",
                "minLength": 3,
                "maxLength": 100,
                "description": "Titolo professionale del lavoro"
            },
            "job_description": {
                "type": "string", 
                "minLength": 10,
                "maxLength": 500,
                "description": "Descrizione dettagliata del lavoro"
            },
            "task_variables": {
                "type": "object",
                "required": [
                    "weight_lbs", "object_name", "origin", 
                    "frequency", "coupling", "significant_control_dest"
                ],
                "properties": {
                    "weight_lbs": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Peso dell'oggetto in libbre"
                    },
                    "object_name": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 50,
                        "description": "Nome dell'oggetto sollevato"
                    },
                    "origin": {
                        "type": "object",
                        "required": ["V", "H", "A"],
                        "properties": {
                            "V": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 70,
                                "description": "Altezza verticale in pollici"
                            },
                            "H": {
                                "type": "number", 
                                "minimum": 4,
                                "maximum": 30,
                                "description": "Distanza orizzontale in pollici"
                            },
                            "A": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 135,
                                "description": "Angolo di asimmetria in gradi"
                            }
                        }
                    },
                    "destination": {
                        "type": "object",
                        "required": ["V", "H", "A"],
                        "properties": {
                            "V": {"type": "number", "minimum": 0, "maximum": 70},
                            "H": {"type": "number", "minimum": 4, "maximum": 30},
                            "A": {"type": "number", "minimum": 0, "maximum": 135}
                        }
                    },
                    "frequency": {
                        "type": "object",
                        "required": ["lifts_per_min", "duration_hours"],
                        "properties": {
                            "lifts_per_min": {
                                "type": "number",
                                "minimum": 0.1,
                                "maximum": 15,
                                "description": "Sollevamenti al minuto"
                            },
                            "duration_hours": {
                                "type": "number",
                                "minimum": 1,
                                "maximum": 8,
                                "description": "Durata lavoro in ore"
                            }
                        }
                    },
                    "coupling": {
                        "type": "string",
                        "enum": ["good", "fair", "poor"],
                        "description": "Qualità della presa"
                    },
                    "significant_control_dest": {
                        "type": "boolean",
                        "description": "Controllo significativo richiesto a destinazione"
                    }
                }
            }
        }
    }
    
    # Schema per risultati calcolo NIOSH
    RESULTS_SCHEMA = {
        "type": "object",
        "required": ["origin", "final_rwl", "final_li", "worst_multipliers"],
        "properties": {
            "origin": {
                "type": "object",
                "required": ["rwl", "li", "multipliers"],
                "properties": {
                    "rwl": {"type": "number", "minimum": 0},
                    "li": {"type": "number", "minimum": 0},
                    "multipliers": {
                        "type": "object",
                        "required": ["HM", "VM", "DM", "AM", "FM", "CM"],
                        "properties": {
                            "HM": {"type": "number", "minimum": 0, "maximum": 1},
                            "VM": {"type": "number", "minimum": 0, "maximum": 1},
                            "DM": {"type": "number", "minimum": 0, "maximum": 1},
                            "AM": {"type": "number", "minimum": 0, "maximum": 1},
                            "FM": {"type": "number", "minimum": 0, "maximum": 1},
                            "CM": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                }
            },
            "destination": {
                "type": "object",
                "required": ["rwl", "li", "multipliers"],
                "properties": {
                    "rwl": {"type": "number", "minimum": 0},
                    "li": {"type": "number", "minimum": 0},
                    "multipliers": {
                        "type": "object",
                        "required": ["HM", "VM", "DM", "AM", "FM", "CM"],
                        "properties": {
                            "HM": {"type": "number", "minimum": 0, "maximum": 1},
                            "VM": {"type": "number", "minimum": 0, "maximum": 1},
                            "DM": {"type": "number", "minimum": 0, "maximum": 1},
                            "AM": {"type": "number", "minimum": 0, "maximum": 1},
                            "FM": {"type": "number", "minimum": 0, "maximum": 1},
                            "CM": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                }
            },
            "final_rwl": {"type": "number", "minimum": 0},
            "final_li": {"type": "number", "minimum": 0},
            "worst_multipliers": {
                "type": "array",
                "items": {
                    "type": "array",
                    "prefixItems": [
                        {"type": "string"},
                        {"type": "number"}
                    ],
                    "minItems": 2,
                    "maxItems": 2
                }
            }
        }
    }


@dataclass
class ValidationResult:
    """Risultato validazione con errori e warning"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class NioshValidator:
    """Validatore per dati NIOSH"""
    
    def __init__(self):
        self.schemas = NioshSchemas()
    
    def _sanitize_scenario_data(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizza i dati generati dal LLM per garantire valori validi
        
        Args:
            scenario_data: Dati scenario grezzi dal LLM
            
        Returns:
            Dati scenario sanitizzati
        """
        import copy
        
        sanitized = copy.deepcopy(scenario_data)
        
        # Sanitizza task_variables
        if 'task_variables' in sanitized:
            task_vars = sanitized['task_variables']
            
            # Peso - assicura che sia positivo e nel range
            if 'weight_lbs' in task_vars:
                weight = task_vars['weight_lbs']
                if weight <= 0:
                    task_vars['weight_lbs'] = max(1, abs(weight))  # Converti negativi in positivi
                elif weight > 100:
                    task_vars['weight_lbs'] = 100  # Limite massimo
            
            # Sanitizza parametri origine
            if 'origin' in task_vars:
                origin = task_vars['origin']
                
                # V (verticale) - deve essere tra 0 e 70
                if 'V' in origin:
                    v = origin['V']
                    origin['V'] = max(0, min(70, abs(v)))
                
                # H (orizzontale) - deve essere tra 4 e 30
                if 'H' in origin:
                    h = origin['H']
                    origin['H'] = max(4, min(30, abs(h)))
                
                # A (asimmetria) - deve essere tra 0 e 135
                if 'A' in origin:
                    a = origin['A']
                    origin['A'] = max(0, min(135, abs(a)))
            
            # Sanitizza parametri destinazione
            if 'destination' in task_vars:
                dest = task_vars['destination']
                
                # V (verticale) - deve essere tra 0 e 70
                if 'V' in dest:
                    v = dest['V']
                    dest['V'] = max(0, min(70, abs(v)))
                
                # H (orizzontale) - deve essere tra 4 e 30
                if 'H' in dest:
                    h = dest['H']
                    dest['H'] = max(4, min(30, abs(h)))
                
                # A (asimmetria) - deve essere tra 0 e 135
                if 'A' in dest:
                    a = dest['A']
                    dest['A'] = max(0, min(135, abs(a)))
            
            # Sanitizza frequenza
            if 'frequency' in task_vars:
                freq = task_vars['frequency']
                
                if 'lifts_per_min' in freq:
                    lifts = freq['lifts_per_min']
                    freq['lifts_per_min'] = max(0.1, min(15, abs(lifts)))
                
                if 'duration_hours' in freq:
                    duration = freq['duration_hours']
                    freq['duration_hours'] = max(1, min(8, abs(duration)))
        
        return sanitized
    
    def validate_scenario(self, scenario_data: Dict[str, Any]) -> ValidationResult:
        """
        Valida dati scenario generati da LLM
        
        Args:
            scenario_data: Dizionario con dati scenario
            
        Returns:
            ValidationResult con errori e warning
        """
        errors = []
        warnings = []
        
        try:
            # Sanitizzazione valori prima della validazione
            sanitized_data = self._sanitize_scenario_data(scenario_data)
            
            # Validazione schema JSON
            jsonschema.validate(instance=sanitized_data, schema=self.schemas.SCENARIO_SCHEMA)
        except jsonschema.ValidationError as e:
            errors.append(f"Errore schema: {e.message}")
            # Usa dati originali per error report, ma dati sanitizzati per continuare
            sanitized_data = self._sanitize_scenario_data(scenario_data)
        
        # Validazione ergonomica aggiuntiva sui dati sanitizzati
        errors.extend(self._validate_ergonomic_consistency(sanitized_data))
        warnings.extend(self._check_ergonomic_warnings(sanitized_data))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_results(self, results_data: Dict[str, Any]) -> ValidationResult:
        """
        Valida risultati calcolo NIOSH
        
        Args:
            results_data: Dizionario con risultati calcolo
            
        Returns:
            ValidationResult con errori e warning
        """
        errors = []
        warnings = []
        
        try:
            # Validazione schema JSON
            jsonschema.validate(instance=results_data, schema=self.schemas.RESULTS_SCHEMA)
        except jsonschema.ValidationError as e:
            errors.append(f"Errore schema risultati: {e.message}")
        
        # Validazione logica dei risultati
        errors.extend(self._validate_results_logic(results_data))
        warnings.extend(self._check_results_warnings(results_data))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_ergonomic_consistency(self, scenario: Dict[str, Any]) -> List[str]:
        """Valida coerenza ergonomica dei dati scenario"""
        errors = []
        task_vars = scenario.get('task_variables', {})
        
        # Controllo coerenza peso-frequenza-durata
        weight = task_vars.get('weight_lbs', 0)
        freq = task_vars.get('frequency', {}).get('lifts_per_min', 0)
        duration = task_vars.get('frequency', {}).get('duration_hours', 0)
        
        # Pesi elevati con frequenze alte sono problematici
        if weight > 50 and freq > 2:
            errors.append("Peso elevato (>50 lbs) con frequenza alta (>2/min) ergonomicamente problematico")
        
        # Frequenze molto alte richiedono durate brevi
        if freq > 5 and duration > 4:
            errors.append("Frequenza molto alta (>5/min) con durata lunga (>4 ore) non sostenibile")
        
        # Controllo posizioni realistiche
        origin = task_vars.get('origin', {})
        dest = task_vars.get('destination', {})
        
        if origin.get('V', 0) > 60:
            errors.append("Altezza origine > 60 pollici (152 cm) ergonomicamente difficile")
        
        if origin.get('H', 0) > 25:
            errors.append("Distanza orizzontale > 25 pollici (63 cm) troppo estesa")
        
        # Controllo coerenza destinazione (se presente)
        if dest:
            # Verifica che destinazione sia diversa da origine
            if abs(dest.get('V', 0) - origin.get('V', 0)) < 2:
                warnings.append("Altezza destinazione molto simile a origine - sollevamento minimo")
        
        return errors
    
    def _check_ergonomic_warnings(self, scenario: Dict[str, Any]) -> List[str]:
        """Controlla warning ergonomici per dati scenario"""
        warnings = []
        task_vars = scenario.get('task_variables', {})
        
        weight = task_vars.get('weight_lbs', 0)
        origin = task_vars.get('origin', {})
        coupling = task_vars.get('coupling', '')
        freq = task_vars.get('frequency', {}).get('lifts_per_min', 0)
        
        # Warning peso
        if weight > 40:
            warnings.append(f"Peso {weight} lbs richiede particolare attenzione")
        
        # Warning altezze
        V = origin.get('V', 0)
        if V < 15:
            warnings.append("Sollevamento da terra molto basso - rischio elevato")
        elif V > 50:
            warnings.append("Sollevamento da altezza eccessiva - rischio spalle/collo")
        
        # Warning distanza orizzontale
        H = origin.get('H', 0)
        if H > 20:
            warnings.append("Distanza orizzontale estesa - stress sulla schiena")
        
        # Warning presa
        if coupling == 'poor':
            warnings.append("Qualità presa scarsa - aumenta rischio significativo")
        
        # Warning frequenza
        if freq > 3:
            warnings.append(f"Frequenza {freq}/min alta - rischio accumulativo")
        
        return warnings
    
    def _validate_results_logic(self, results: Dict[str, Any]) -> List[str]:
        """Valida logica dei risultati calcolo"""
        errors = []
        
        # RWL deve essere positivo
        final_rwl = results.get('final_rwl', 0)
        if final_rwl <= 0:
            errors.append("RWL finale deve essere positivo")
        
        # LI deve essere positivo
        final_li = results.get('final_li', 0)
        if final_li <= 0:
            errors.append("LI finale deve essere positivo")
        
        # Controlli moltiplicatori origine
        origin = results.get('origin', {})
        if origin:
            mults = origin.get('multipliers', {})
            for mult_name, value in mults.items():
                if value < 0 or value > 1:
                    errors.append(f"Moltiplicatore {mult_name} fuori range [0,1]: {value}")
        
        # Controlli moltiplicatori destinazione
        dest = results.get('destination', {})
        if dest:
            mults = dest.get('multipliers', {})
            for mult_name, value in mults.items():
                if value < 0 or value > 1:
                    errors.append(f"Moltiplicatore destinazione {mult_name} fuori range: {value}")
        
        return errors
    
    def _check_results_warnings(self, results: Dict[str, Any]) -> List[str]:
        """Controlla warning sui risultati"""
        warnings = []
        
        final_li = results.get('final_li', 0)
        
        # Warning LI elevato
        if final_li > 3.0:
            warnings.append("LI > 3.0 - Rischio molto elevato, interventi immediati necessari")
        elif final_li > 2.0:
            warnings.append("LI > 2.0 - Rischio elevato, interventi necessari")
        elif final_li > 1.0:
            warnings.append("LI > 1.0 - Rischio moderato, miglioramenti raccomandati")
        
        # Warning RWL molto basso
        final_rwl = results.get('final_rwl', 0)
        if final_rwl < 10:
            warnings.append("RWL molto basso (<10 lbs) - capacità di sollevamento ridotta")
        
        return warnings


def main():
    """Test del validatore"""
    # Scenario di test
    test_scenario = {
        "job_title": "Operatore Magazzino",
        "job_description": "Sollevamento scatole da scaffale a nastro trasportatore",
        "task_variables": {
            "weight_lbs": 25,
            "object_name": "Scatola di cartone",
            "origin": { "V": 15, "H": 18, "A": 45 },
            "destination": { "V": 35, "H": 12, "A": 0 },
            "frequency": { "lifts_per_min": 2, "duration_hours": 6 },
            "coupling": "fair",
            "significant_control_dest": false
                            }
                    }
    
    validator = NioshValidator()
    result = validator.validate_scenario(test_scenario)
    
    print("RISULTATI VALIDAZIONE:")
    print(f"Valido: {result.is_valid}")
    print(f"Errori: {result.errors}")
    print(f"Warning: {result.warnings}")


if __name__ == "__main__":
    main()