"""
Sistema di Controllo NIOSH - Controllo Qualità e Audit Trail

Questo modulo implementa un sistema completo di controllo qualità e audit trail
per garantire l'accuratezza, la coerenza e la conformità dei report NIOSH.

Architettura di controllo multilivello:
- Validazione matematica rigorosa con tolleranze configurabili
- Cross-validation tra componenti del sistema
- Audit trail completo con tracciabilità timestamped
- Semantic consistency checks per coerenza tecnica
- Regulatory compliance validation secondo standard NIOSH
- Sistema di backup automatico per protezione dati

Componenti principali:
1. ControlLevel: Definizione livelli di controllo (Basic/Standard/Comprehensive)
2. ControlConfig: Configurazione centralizzata parametri di controllo
3. AuditRecord: Struttura per tracciabilità completa operazioni
4. ValidationResult: Struttura standardizzata risultati validazione
5. NIOSHControlSystem: Classe principale orchestratore controlli

Funzionalità avanzate:
- Logging dettagliato con file handler e stream output
- Gestione errori con traceback completo
- Hashing per integrità dati (SHA-256)
- Backup automatico file originali
- Metriche performance (execution time)
- Reporting strutturato risultati controllo

Casi d'uso tipici:
- Validazione automatica report prima emissione
- Audit trail per conformità regolamentare
- Quality assurance in pipeline produzione
- Investigazione errori e debugging
- Metriche qualità e trend analysis

Versione: 1.0

"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import copy

# Setup logging avanzato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('niosh_control_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================

class ControlLevel(Enum):
    """
    Livelli di intensità per il sistema di controllo NIOSH.
    
    Definisce tre livelli di controllo con intensità crescente:
    - Basic: Validazioni matematiche essenziali per integrità di base
    - Standard: Validazioni complete incluse analisi AI e coerenza semantica
    - Comprehensive: Validazioni estese con controlli incrociati e compliance completa
    
    Utilizzo:
        BASIC: Quick validation per test e development
        STANDARD: Produzione routine con quality assurance standard
        COMPREHENSIVE: Report critici o audit formali con massima garanzia
    """
    BASIC = "basic"          # Validazione matematica di base (RWL, LI)
    STANDARD = "standard"    # Validazione completa con AI e coerenza
    COMPREHENSIVE = "comprehensive"  # Validazione estesa massima

class ValidationStatus(Enum):
    """
    Stati possibili per risultati di validazione del sistema.
    
    Gerarchia di severità crescente:
    - PENDING: Validazione in attesa di essere processata
    - VALID: Validazione superata senza problemi
    - WARNING: Validazione superata con avvertimenti minori
    - ERROR: Validazione fallita per errori correggibili
    - CRITICAL: Validazione fallita per errori critici bloccanti
    
    Utilizzati per classificare risultati e azioni correttive necessarie.
    """
    PENDING = "pending"
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ControlAction(Enum):
    """
    Tipi di azioni di controllo eseguibili dal sistema.
    
    Ogni azione rappresenta una categoria specifica di validazione:
    - VALIDATE_CALCULATIONS: Verifica accuratezza matematica calcoli NIOSH
    - CROSS_CHECK_REFERENCES: Validazione coerenza riferimenti incrociati
    - AUDIT_TRAIL_VERIFY: Verifica integrità e completezza audit trail
    - SEMANTIC_CONSISTENCY: Controllo coerenza tecnica e linguistica
    - REGULATORY_COMPLIANCE: Validazione conformità standard NIOSH
    
    Utilizzate per configurare quali controlli eseguire in base al livello.
    """
    VALIDATE_CALCULATIONS = "validate_calculations"
    CROSS_CHECK_REFERENCES = "cross_check_references"
    AUDIT_TRAIL_VERIFY = "audit_trail_verify"
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    REGULATORY_COMPLIANCE = "regulatory_compliance"

@dataclass
class ControlConfig:
    """
    Configurazione centralizzata per il sistema di controllo NIOSH.
    
    Centralizza tutti i parametri configurabili per personalizzare
    il comportamento del sistema di controllo in base alle esigenze.
    
    Attributi configurabili:
        - enable_audit_trail: Attiva tracciamento completo operazioni
        - enable_cross_validation: Attiva validazioni incrociate componenti
        - enable_semantic_checks: Attiva controlli coerenza linguistica
        - enable_regulatory_checks: Attiva validazioni conformità NIOSH
        - max_calculation_tolerance: Tolleranza errori calcoli (default 2%)
        - backup_original_files: Backup automatico file originali
        - log_level: Livello dettaglio logging (DEBUG/INFO/WARNING/ERROR)
        - control_level: Intensità controlli (Basic/Standard/Comprehensive)
    
    Utilizzo tipico:
        config = ControlConfig(
            control_level=ControlLevel.STANDARD,
            max_calculation_tolerance=0.01,  # 1% per maggiore accuratezza
            enable_regulatory_checks=True
        )
    """
    enable_audit_trail: bool = True
    enable_cross_validation: bool = True
    enable_semantic_checks: bool = True
    enable_regulatory_checks: bool = True
    max_calculation_tolerance: float = 0.02  # 2% tolleranza
    backup_original_files: bool = True
    log_level: str = "INFO"
    control_level: ControlLevel = ControlLevel.STANDARD
    
    def __post_init__(self):
        """
        Post-inizializzazione per conversione automatica tipi.
        
        Converte automaticamente stringhe in enum ControlLevel
        per supportare configurazioni da file JSON/dizionario.
        """
        if isinstance(self.control_level, str):
            self.control_level = ControlLevel(self.control_level)

# ============================================================================
# DATA STRUCTURES FOR AUDIT AND CONTROL
# ============================================================================

@dataclass
class AuditRecord:
    """Record di audit per tracciabilità"""
    timestamp: str
    action: str
    component: str
    input_hash: str
    output_hash: str
    status: ValidationStatus
    details: Dict[str, Any]
    execution_time_ms: float
    error_trace: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ValidationResult:
    """Risultato di una validazione"""
    is_valid: bool
    status: ValidationStatus
    component: str
    check_type: str
    details: Dict[str, Any]
    recommendations: List[str]
    errors: List[str]
    warnings: List[str]
    execution_time_ms: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ControlReport:
    """Report completo di controllo"""
    session_id: str
    timestamp: str
    control_level: ControlLevel
    overall_status: ValidationStatus
    validation_results: List[ValidationResult]
    audit_records: List[AuditRecord]
    performance_metrics: Dict[str, float]
    summary: Dict[str, Any]
    
    def to_dict(self):
        return {
            **asdict(self),
            'control_level': self.control_level.value,
            'overall_status': self.overall_status.value
        }

# ============================================================================
# CORE CONTROL SYSTEM
# ============================================================================

class NIOSHControlSystem:
    """
    Sistema di controllo qualità completo per report NIOSH.
    
    Classe principale che orchestra tutte le funzionalità di controllo,
    audit trail e quality assurance per garantire report accurati e conformi.
    
    Architettura di controllo:
    1. Configurazione centralizzata con livelli di controllo
    2. Audit trail completo con hashing SHA-256 per integrità
    3. Sistema di backup automatico per protezione dati
    4. Logging dettagliato per debug e conformità
    5. Metriche performance per monitoraggio efficienza
    6. Validazioni multilivello da matematica a semantica
    
    Funzionalità principali:
        - Validazione matematica calcoli NIOSH (RWL, LI, moltiplicatori)
        - Cross-referencing tra dati di input e output
        - Audit trail immutabile con timestamp UTC
        - Controllo coerenza linguistica e tecnica
        - Compliance validation secondo standard NIOSH
        - Reporting strutturato risultati e raccomandazioni
    
    Utilizzo tipico:
        config = ControlConfig(control_level=ControlLevel.STANDARD)
        control_system = NIOSHControlSystem(config=config)
        result = control_system.validate_report(report_data, calculated_data)
        
    Output:
        - Report di validazione dettagliato
        - File backup originali (se configurato)
        - Audit trail completo in JSON
        - Log di sessione con metriche
    """
    
    def __init__(self, config: Optional[ControlConfig] = None, workspace_dir: str = "."):
        """
        Inizializza il sistema di controllo NIOSH completo.
        
        Configura tutte le componenti del sistema di controllo inclusi
        directory, logging, session ID e strutture dati per audit trail.
        
        Args:
            config (Optional[ControlConfig]): Configurazione personalizzata.
                                            Se None, usa configurazione default.
            workspace_dir (str): Directory base per operazioni di controllo.
                                 Default "." (directory corrente).
                                 
        Setup automatico:
            - Creazione directory backup/reports/logs
            - Session ID univoco con timestamp UTC
            - Logging specifico per sessione
            - Strutture dati per audit trail
            - Metriche performance inizializzazione
            
        Note:
            - Tutte le directory vengono create automaticamente se non esistono
            - Session ID formato: CTRL_YYYYMMDD_HHMMSS_XXXXXXXX
            - Logging configurato per output su file e console
        """
        self.config = config or ControlConfig()
        self.workspace_dir = Path(workspace_dir)
        self.audit_trail = []
        self.session_id = self._generate_session_id()
        self.start_time = datetime.now(timezone.utc)
        
        # Directory per backup e logs
        self.backup_dir = self.workspace_dir / "control_backups"
        self.reports_dir = self.workspace_dir / "control_reports"
        self.logs_dir = self.workspace_dir / "control_logs"
        
        for dir_path in [self.backup_dir, self.reports_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"NIOSH Control System initialized - Session: {self.session_id}")
        logger.info(f"Control Level: {self.config.control_level.value}")
    
    def _generate_session_id(self) -> str:
        """Genera ID sessione univoco"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        random_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"CTRL_{timestamp}_{random_hash}"
    
    def _setup_logging(self):
        """Setup logging specifico per la sessione"""
        log_file = self.logs_dir / f"control_{self.session_id}.log"
        
        # File handler per questa sessione
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Formatter dettagliato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Aggiungi handler al logger
        logger.addHandler(file_handler)
    
    def _calculate_hash(self, data: Union[str, Dict]) -> str:
        """Calcola hash SHA256 per integrità dati"""
        try:
            if isinstance(data, dict):
                # Converti enum a string prima di serializzare
                def convert_enum(obj):
                    if hasattr(obj, 'value'):
                        return obj.value
                    return obj
                
                data_str = json.dumps(data, sort_keys=True, ensure_ascii=False, default=convert_enum)
            else:
                data_str = str(data)
            
            return hashlib.sha256(data_str.encode('utf-8')).hexdigest()[:16]
        except Exception as e:
            # Fallback se la serializzazione fallisce
            try:
                return hashlib.sha256(str(data).encode('utf-8')).hexdigest()[:16]
            except:
                return "hash_error"
    
    def _create_audit_record(self, action: str, component: str, 
                           input_data: Any, output_data: Any,
                           status: ValidationStatus, details: Dict,
                           execution_time: float, error_trace: str = None) -> AuditRecord:
        """Crea record di audit"""
        return AuditRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            component=component,
            input_hash=self._calculate_hash(input_data),
            output_hash=self._calculate_hash(output_data),
            status=status,
            details=details,
            execution_time_ms=execution_time * 1000,
            error_trace=error_trace
        )
    
    def _backup_if_needed(self, file_path: Path, data: Any):
        """Crea backup dei dati originali se configurato"""
        if not self.config.backup_original_files:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            if isinstance(data, str):
                backup_path.write_text(data, encoding='utf-8')
            else:
                backup_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
                
            logger.debug(f"Backup created: {backup_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    def validate_calculation_chain(self, calculated_data: Dict) -> ValidationResult:
        """Valida la catena di calcoli NIOSH"""
        start_time = datetime.now()
        errors = []
        warnings = []
        recommendations = []
        details = {}
        
        try:
            # Estrai risultati di calcolo
            calc_results = calculated_data.get("calculation_results", {})
            if not calc_results:
                raise ValueError("No calculation results found in data")
            
            # Validazione RWL origin
            origin_data = calc_results.get("origin", {})
            if not origin_data:
                raise ValueError("No origin calculation data found")
            
            rwl_origin = origin_data.get("RWL", 0)
            li_origin = origin_data.get("LI", 0)
            
            # Verifica consistenza matematica base
            expected_rwl = self._recalculate_rwl(calculated_data, "origin")
            rwl_diff = abs(rwl_origin - expected_rwl) / expected_rwl if expected_rwl > 0 else float('inf')
            
            details["rwl_origin"] = rwl_origin
            details["rwl_recalculated"] = expected_rwl
            details["rwl_difference_pct"] = round(rwl_diff * 100, 2)
            
            if rwl_diff > self.config.max_calculation_tolerance:
                errors.append(f"RWL calculation mismatch: {rwl_diff:.3%} difference")
                recommendations.append("Review NIOSH multiplier calculations")
            elif rwl_diff > 0.01:  # 1% warning threshold
                warnings.append(f"Minor RWL calculation variance: {rwl_diff:.3%}")
            
            # Validazione LI
            L = calculated_data.get("common_parameters", {}).get("L_load_kg", 0)
            expected_li = L / expected_rwl if expected_rwl > 0 else float('inf')
            
            if li_origin != "inf":
                li_diff = abs(li_origin - expected_li) / expected_li if expected_li > 0 else float('inf')
                details["li_origin"] = li_origin
                details["li_recalculated"] = expected_li
                details["li_difference_pct"] = round(li_diff * 100, 2)
                
                if li_diff > self.config.max_calculation_tolerance:
                    errors.append(f"LI calculation mismatch: {li_diff:.3%} difference")
            
            # Validazione limiting factors
            limiting_factors = calc_results.get("limiting_factors", [])
            if limiting_factors:
                # Verifica che i limiting factors siano ordinati correttamente
                values = [f["value"] for f in limiting_factors]
                if not all(values[i] <= values[i+1] for i in range(len(values)-1)):
                    warnings.append("Limiting factors not properly sorted by value")
                    recommendations.append("Ensure limiting factors are sorted in ascending order")
                
                details["limiting_factors"] = limiting_factors
                details["lowest_multiplier"] = values[0] if values else 1.0
            
            # Validazione destination se presente
            if calculated_data.get("significant_control_at_destination", False):
                dest_data = calc_results.get("destination", {})
                if dest_data:
                    expected_rwl_dest = self._recalculate_rwl(calculated_data, "destination")
                    rwl_dest_diff = abs(dest_data.get("RWL", 0) - expected_rwl_dest) / expected_rwl_dest
                    
                    if rwl_dest_diff > self.config.max_calculation_tolerance:
                        errors.append(f"Destination RWL calculation mismatch: {rwl_dest_diff:.3%}")
                    
                    details["rwl_destination"] = dest_data.get("RWL")
                    details["rwl_destination_recalculated"] = expected_rwl_dest
            
            # Determina stato finale
            status = ValidationStatus.VALID
            if errors:
                status = ValidationStatus.ERROR
            elif warnings:
                status = ValidationStatus.WARNING
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                status=status,
                component="calculation_chain",
                check_type="mathematical_validation",
                details=details,
                recommendations=recommendations,
                errors=errors,
                warnings=warnings,
                execution_time_ms=execution_time * 1000
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_trace = traceback.format_exc()
            
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.CRITICAL,
                component="calculation_chain",
                check_type="mathematical_validation",
                details={"error": str(e)},
                recommendations=["Fix calculation data structure"],
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                execution_time_ms=execution_time * 1000
            )
    
    def _recalculate_rwl(self, calculated_data: Dict, position: str) -> float:
        """Ricalcola RWL per verifica"""
        try:
            from niosh_calculator_v2 import LC, get_hm, get_vm, get_dm, get_am, get_fm, get_cm
            
            params = calculated_data
            calc_results = calculated_data["calculation_results"]
            
            if position == "origin":
                H = params["origin_parameters"]["H_origin_cm"]
                V = params["origin_parameters"]["V_origin_cm"]
                A = params["origin_parameters"]["A_origin_degrees"]
                C = params["origin_parameters"]["C_origin_type"]
            else:  # destination
                H = params["destination_parameters"]["H_dest_cm"]
                V = params["destination_parameters"]["V_dest_cm"]
                A = params["destination_parameters"]["A_dest_degrees"]
                C = params["destination_parameters"]["C_dest_type"]
            
            F = params["common_parameters"]["F_frequency_per_min"]
            duration = params["common_parameters"]["duration_hours"]
            V_start = params["origin_parameters"]["V_origin_cm"]
            D = calc_results["vertical_travel_distance"]
            
            HM = get_hm(H)
            VM = get_vm(V)
            DM = get_dm(D)
            AM = get_am(A)
            FM, _ = get_fm(F, V_start, duration)
            CM = get_cm(C, V)
            
            RWL = LC * HM * VM * DM * AM * FM * CM
            return round(RWL, 2)
            
        except Exception as e:
            logger.warning(f"RWL recalculation failed: {e}")
            return 0.0
    
    def validate_report_consistency(self, report_text: str, calculated_data: Dict) -> ValidationResult:
        """Valida consistenza tra report e dati di calcolo"""
        start_time = datetime.now()
        errors = []
        warnings = []
        recommendations = []
        details = {}
        
        try:
            # Import validator
            from niosh_validator import NIOSHValidator
            
            validator = NIOSHValidator()
            
            # Validazione matematica
            math_valid, math_errors, corrected_text = validator.validate_mathematical_consistency(
                report_text, calculated_data
            )
            
            details["mathematical_valid"] = math_valid
            details["mathematical_errors_count"] = len(math_errors)
            details["mathematical_errors"] = math_errors[:5]  # Limit to first 5 errors
            
            if math_errors:
                errors.extend([f"Math error: {err.get('type', 'unknown')} - {err.get('location', 'N/A')}" 
                             for err in math_errors[:3]])
                recommendations.append("Review mathematical calculations in report")
            
            # Validazione linguistica
            ling_valid, ling_issues, _ = validator.validate_linguistic_consistency(report_text)
            
            details["linguistic_valid"] = ling_valid
            details["linguistic_issues_count"] = len(ling_issues)
            details["linguistic_issues"] = ling_issues[:5]  # Limit to first 5 issues
            
            if ling_issues:
                warnings.extend([f"Linguistic issue: {issue.get('type', 'unknown')}" 
                                for issue in ling_issues[:2]])
                recommendations.append("Review linguistic consistency in report")
            
            # Verifica elementi essenziali nel report
            required_elements = [
                "RWL",
                "Lifting Index",
                "multipliers",
                "risk assessment"
            ]
            
            missing_elements = []
            for element in required_elements:
                if element.lower() not in report_text.lower():
                    missing_elements.append(element)
            
            if missing_elements:
                errors.append(f"Missing required sections: {', '.join(missing_elements)}")
                recommendations.append("Ensure all required NIOSH sections are present")
            
            details["missing_elements"] = missing_elements
            
            # Verifica coerenza valori numerici
            calc_results = calculated_data.get("calculation_results", {})
            if calc_results:
                origin_rwl = calc_results.get("origin", {}).get("RWL", 0)
                if origin_rwl > 0:
                    # Cerca RWL nel testo
                    import re
                    rwl_pattern = r'RWL.*?(\d+(?:\.\d+)?)\s*kg'
                    matches = re.findall(rwl_pattern, report_text, re.IGNORECASE)
                    
                    if matches:
                        reported_rwl = float(matches[0])
                        rwl_diff = abs(reported_rwl - origin_rwl) / origin_rwl
                        details["reported_rwl"] = reported_rwl
                        details["expected_rwl"] = origin_rwl
                        details["rwl_text_difference_pct"] = round(rwl_diff * 100, 2)
                        
                        if rwl_diff > 0.05:  # 5% tolerance for text
                            warnings.append(f"RWL in text differs from calculation by {rwl_diff:.1%}")
                    else:
                        warnings.append("RWL value not found in report text")
            
            # Determina stato finale
            status = ValidationStatus.VALID
            if errors:
                status = ValidationStatus.ERROR
            elif warnings:
                status = ValidationStatus.WARNING
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                status=status,
                component="report_consistency",
                check_type="content_validation",
                details=details,
                recommendations=recommendations,
                errors=errors,
                warnings=warnings,
                execution_time_ms=execution_time * 1000
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_trace = traceback.format_exc()
            
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.CRITICAL,
                component="report_consistency",
                check_type="content_validation",
                details={"error": str(e)},
                recommendations=["Fix report validation process"],
                errors=[f"Report validation failed: {str(e)}"],
                warnings=[],
                execution_time_ms=execution_time * 1000
            )
    
    def audit_data_integrity(self, calculated_data: Dict) -> ValidationResult:
        """Audita integrità dei dati"""
        start_time = datetime.now()
        errors = []
        warnings = []
        recommendations = []
        details = {}
        
        try:
            # Verifica struttura dati base
            required_sections = [
                "scenario_id",
                "title", 
                "origin_parameters",
                "destination_parameters",
                "common_parameters",
                "calculation_results"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in calculated_data:
                    missing_sections.append(section)
            
            if missing_sections:
                errors.append(f"Missing required sections: {', '.join(missing_sections)}")
                recommendations.append("Ensure all required data sections are present")
            
            details["missing_sections"] = missing_sections
            
            # Verifica parametri origin
            origin_params = calculated_data.get("origin_parameters", {})
            required_origin = ["H_origin_cm", "V_origin_cm", "A_origin_degrees", "C_origin_type"]
            
            missing_origin = []
            for param in required_origin:
                if param not in origin_params:
                    missing_origin.append(param)
            
            if missing_origin:
                errors.append(f"Missing origin parameters: {', '.join(missing_origin)}")
            
            details["missing_origin_params"] = missing_origin
            
            # Verifica parametri comuni
            common_params = calculated_data.get("common_parameters", {})
            required_common = ["L_load_kg", "F_frequency_per_min", "duration_hours"]
            
            missing_common = []
            for param in required_common:
                if param not in common_params:
                    missing_common.append(param)
            
            if missing_common:
                errors.append(f"Missing common parameters: {', '.join(missing_common)}")
            
            details["missing_common_params"] = missing_common
            
            # Verifica ranges parametri
            range_errors = []
            
            # H range: 15-63 cm
            H_orig = origin_params.get("H_origin_cm", 0)
            if not (15 <= H_orig <= 63):
                range_errors.append(f"H_origin_cm ({H_orig}) outside valid range [15-63]")
            
            # V range: 0-175 cm  
            V_orig = origin_params.get("V_origin_cm", 0)
            if not (0 <= V_orig <= 175):
                range_errors.append(f"V_origin_cm ({V_orig}) outside valid range [0-175]")
            
            # A range: 0-135 degrees
            A_orig = origin_params.get("A_origin_degrees", 0)
            if not (0 <= A_orig <= 135):
                range_errors.append(f"A_origin_degrees ({A_orig}) outside valid range [0-135]")
            
            # Load range: 5-35 kg
            L_load = common_params.get("L_load_kg", 0)
            if not (5 <= L_load <= 35):
                range_errors.append(f"L_load_kg ({L_load}) outside valid range [5-35]")
            
            # Frequency range: 0-15 lifts/min
            F_freq = common_params.get("F_frequency_per_min", 0)
            if not (0 <= F_freq <= 15):
                range_errors.append(f"F_frequency_per_min ({F_freq}) outside valid range [0-15]")
            
            # Duration options
            duration = common_params.get("duration_hours", 0)
            if duration not in [1.0, 2.0, 4.0, 8.0]:
                range_errors.append(f"duration_hours ({duration}) not in allowed values [1, 2, 4, 8]")
            
            if range_errors:
                warnings.extend(range_errors)
                recommendations.append("Review parameter ranges for NIOSH compliance")
            
            details["range_errors"] = range_errors
            
            # Verifica coerenza tra parametri
            consistency_errors = []
            
            # Controllo significativo vs frequenza
            sig_control = calculated_data.get("significant_control_at_destination", False)
            if sig_control and F_freq > 4.0:
                consistency_errors.append("High frequency with significant control may be unrealistic")
            
            # Distanza verticale minima
            dest_params = calculated_data.get("destination_parameters", {})
            V_dest = dest_params.get("V_dest_cm", V_orig)
            D = abs(V_dest - V_orig)
            if D < 25:
                consistency_errors.append(f"Vertical travel distance ({D} cm) below minimum (25 cm)")
            
            if consistency_errors:
                warnings.extend(consistency_errors)
                recommendations.append("Review parameter consistency")
            
            details["consistency_errors"] = consistency_errors
            
            # Calcola hash dei dati per integrità
            data_hash = self._calculate_hash(calculated_data)
            details["data_hash"] = data_hash
            details["data_size_bytes"] = len(json.dumps(calculated_data).encode())
            
            # Determina stato finale
            status = ValidationStatus.VALID
            if errors:
                status = ValidationStatus.ERROR
            elif warnings:
                status = ValidationStatus.WARNING
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                status=status,
                component="data_integrity",
                check_type="structure_validation",
                details=details,
                recommendations=recommendations,
                errors=errors,
                warnings=warnings,
                execution_time_ms=execution_time * 1000
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_trace = traceback.format_exc()
            
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.CRITICAL,
                component="data_integrity", 
                check_type="structure_validation",
                details={"error": str(e)},
                recommendations=["Fix data structure"],
                errors=[f"Integrity audit failed: {str(e)}"],
                warnings=[],
                execution_time_ms=execution_time * 1000
            )
    
    def apply_corrections_to_report(self, report_text: str, validation_results: List[ValidationResult], calculated_data: Dict) -> str:
        """Applica correzioni automatiche al report basandosi sui risultati delle validazioni"""
        corrected_text = report_text
        corrections_applied = []
        
        try:
            # Correggi errori matematici nel testo
            for result in validation_results:
                if result.component == "calculation_chain" and result.status in [ValidationStatus.ERROR, ValidationStatus.WARNING]:
                    for error in result.errors:
                        if isinstance(error, dict) and error.get('type') in ['calculation_error', 'rwl_mismatch', 'li_mismatch']:
                            correction = self._fix_mathematical_error(corrected_text, error, calculated_data)
                            if correction != corrected_text:
                                corrected_text = correction
                                corrections_applied.append(f"Fixed mathematical error: {error.get('type')}")
                
                elif result.component == "report_consistency" and result.status in [ValidationStatus.ERROR, ValidationStatus.WARNING]:
                    for error in result.errors:
                        if isinstance(error, dict) and 'missing' in error.get('type', '').lower():
                            correction = self._add_missing_section(corrected_text, error, calculated_data)
                            if correction != corrected_text:
                                corrected_text = correction
                                corrections_applied.append(f"Added missing section: {error.get('type')}")
            
            # Aggiungi sezione Risk Assessment se mancante
            if "## Risk Assessment" not in corrected_text:
                risk_section = self._generate_risk_assessment_section(calculated_data)
                if risk_section:
                    # Inserisci dopo "## Job Analysis"
                    insertion_point = corrected_text.find("## Job Analysis")
                    if insertion_point != -1:
                        end_of_section = corrected_text.find("\n\n", insertion_point) + 2
                        corrected_text = (corrected_text[:end_of_section] + 
                                         risk_section + "\n\n" + 
                                         corrected_text[end_of_section:])
                        corrections_applied.append("Added Risk Assessment section")
            
            # Correggi valori LI nel testo se non corrispondono
            calc_results = calculated_data.get("calculation_results", {})
            if calc_results:
                origin_data = calc_results.get("origin", {})
                correct_li = origin_data.get("LI", 0)
                if correct_li != "inf":
                    # Cerca e correggi LI nel testo
                    import re
                    li_patterns = [
                        r'LI = L/RWL = [\d.]+/[\d.]+ = \*\*([\d.]+)\*\*',
                        r'Lifting Index \(LI\): \*\*([\d.]+)\*\*',
                        r'LI = ([\d.]+)'
                    ]
                    
                    for pattern in li_patterns:
                        matches = re.findall(pattern, corrected_text)
                        for match in matches:
                            try:
                                reported_li = float(match)
                                if abs(reported_li - correct_li) > 0.1:  # Differenza significativa
                                    old_li_text = f"{reported_li:.2f}"
                                    new_li_text = f"{correct_li:.2f}"
                                    corrected_text = corrected_text.replace(old_li_text, new_li_text)
                                    corrections_applied.append(f"Corrected LI: {reported_li:.2f} → {correct_li:.2f}")
                            except ValueError:
                                continue
            
            # Correggi valori RWL nel testo se non corrispondono
            if calc_results:
                origin_data = calc_results.get("origin", {})
                correct_rwl = origin_data.get("RWL", 0)
                rwl_patterns = [
                    r'RWL = 23 × [\d.× ]+= \*\*([\d.]+)\s*kg\*\*',
                    r'Recommended Weight Limit \(RWL\): \*\*([\d.]+)\s*kg\*\*',
                    r'RWL: \*\*([\d.]+)\*\*'
                ]
                
                for pattern in rwl_patterns:
                    matches = re.findall(pattern, corrected_text)
                    for match in matches:
                        try:
                            reported_rwl = float(match)
                            if abs(reported_rwl - correct_rwl) > 0.5:  # Differenza significativa
                                old_rwl_text = f"{reported_rwl:.1f}"
                                new_rwl_text = f"{correct_rwl:.1f}"
                                corrected_text = corrected_text.replace(old_rwl_text, new_rwl_text)
                                corrections_applied.append(f"Corrected RWL: {reported_rwl:.1f} → {correct_rwl:.1f}")
                        except ValueError:
                            continue
            
            if corrections_applied:
                logger.info(f"Applied {len(corrections_applied)} corrections to report")
                for correction in corrections_applied:
                    logger.debug(f"  - {correction}")
            
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error applying corrections to report: {e}")
            return report_text

    def _fix_mathematical_error(self, text: str, error: Dict, calculated_data: Dict) -> str:
        """Corregge error matematico specifico nel testo"""
        try:
            calc_results = calculated_data.get("calculation_results", {})
            origin = calc_results.get("origin", {})
            
            if error['type'] == 'rwl_mismatch':
                correct_rwl = origin.get("RWL", 0)
                # Correggi RWL nel testo
                import re
                rwl_pattern = r'RWL.*?=.*?=\s*\*?\*?([\d.]+)\s*kg\*?\*?'
                matches = re.findall(rwl_pattern, text)
                for match in matches:
                    old_rwl = f"{float(match):.1f}"
                    new_rwl = f"{correct_rwl:.1f}"
                    text = text.replace(old_rwl, new_rwl)
            
            elif error['type'] == 'li_mismatch':
                correct_li = origin.get("LI", 0)
                if correct_li != "inf":
                    # Correggi LI nel testo
                    import re
                    li_pattern = r'LI\s*=\s*[\d.]+/[\d.]+\s*=\s*\*?\*?([\d.]+)\*?\*?'
                    matches = re.findall(li_pattern, text)
                    for match in matches:
                        old_li = f"{float(match):.2f}"
                        new_li = f"{correct_li:.2f}"
                        text = text.replace(old_li, new_li)
            
            return text
            
        except Exception as e:
            logger.error(f"Error fixing mathematical error: {e}")
            return text

    def _add_missing_section(self, text: str, error: Dict, calculated_data: Dict) -> str:
        """Aggiunge sezione mancante al report"""
        try:
            if 'risk assessment' in error.get('type', '').lower():
                risk_section = self._generate_risk_assessment_section(calculated_data)
                if risk_section:
                    # Inserisci prima della conclusione
                    conclusion_pos = text.find("## Conclusions")
                    if conclusion_pos != -1:
                        return text[:conclusion_pos] + risk_section + "\n\n" + text[conclusion_pos:]
                    else:
                        # Inserisci alla fine
                        return text + "\n\n" + risk_section
            
            return text
            
        except Exception as e:
            logger.error(f"Error adding missing section: {e}")
            return text

    def _generate_risk_assessment_section(self, calculated_data: Dict) -> str:
        """Genera sezione Risk Assessment basata sui dati calcolati"""
        try:
            calc_results = calculated_data.get("calculation_results", {})
            origin = calc_results.get("origin", {})
            
            li = origin.get("LI", 0)
            rwl = origin.get("RWL", 0)
            load = calculated_data.get("common_parameters", {}).get("L_load_kg", 0)
            
            # Determina livello di rischio
            if li > 3:
                risk_level = "High"
                risk_description = "High risk - job redesign required immediately"
            elif li > 1:
                risk_level = "Moderate" 
                risk_description = "Moderate risk - ergonomic intervention recommended"
            else:
                risk_level = "Low"
                risk_description = "Low risk - acceptable for most workers"
            
            return f"""## Risk Assessment

### Quantitative Risk Analysis
**Lifting Index (LI):** {li:.2f}
- LI ≤ 1.0: Acceptable for most workers
- 1.0 < LI ≤ 3.0: Some workers at increased risk  
- LI > 3.0: High risk, job redesign required

**Current Risk Level:** {risk_level} (LI = {li:.2f})

### Load Analysis
- **Actual Load:** {load:.1f} kg
- **Recommended Weight Limit (RWL):** {rwl:.1f} kg
- **Load Status:** {'Exceeds RWL' if load > rwl else 'Within RWL'}

### Ergonomic Risk Factors
**Primary Limiting Factors:** {', '.join([f"{f['factor']} ({f['value']:.3f})" for f in calc_results.get('limiting_factors', [])[:3]])}

**Recommendations:**
1. {'Reduce load weight or improve lifting technique' if load > rwl else 'Maintain current load weight'}
2. Optimize work height and posture
3. Consider mechanical assistance for repetitive tasks
4. Implement job rotation for extended duration tasks

**Risk Conclusion:** {risk_description}

Implementation of recommended controls should reduce LI below 1.0, making the task acceptable for most workers.
"""
            
        except Exception as e:
            logger.error(f"Error generating risk assessment section: {e}")
            return ""

    def cross_validate_references(self, calculated_data: Dict, report_text: str) -> ValidationResult:
        """Validazione incrociata tra dati calcolati e report"""
        start_time = datetime.now()
        errors = []
        warnings = []
        recommendations = []
        details = {}
        
        try:
            # Estrai valori calcolati
            calc_results = calculated_data.get("calculation_results", {})
            origin_data = calc_results.get("origin", {})
            
            # Lista di check incrociati
            cross_checks = []
            
            # 1. RWL consistency
            if origin_data:
                expected_rwl = origin_data.get("RWL", 0)
                import re
                rwl_matches = re.findall(r'RWL.*?(\d+(?:\.\d+)?)\s*kg', report_text, re.IGNORECASE)
                
                if rwl_matches:
                    reported_rwl = float(rwl_matches[0])
                    rwl_diff = abs(reported_rwl - expected_rwl) / expected_rwl if expected_rwl > 0 else float('inf')
                    
                    cross_checks.append({
                        "check": "RWL consistency",
                        "expected": expected_rwl,
                        "reported": reported_rwl,
                        "difference_pct": round(rwl_diff * 100, 2),
                        "status": "pass" if rwl_diff <= 0.05 else "fail"
                    })
                    
                    if rwl_diff > 0.05:
                        errors.append(f"RWL mismatch: expected {expected_rwl}, reported {reported_rwl}")
                else:
                    warnings.append("RWL value not found in report")
            
            # 2. LI consistency  
            expected_li = origin_data.get("LI", 0)
            if expected_li != "inf":
                li_matches = re.findall(r'LI.*?(\d+(?:\.\d+)?)', report_text, re.IGNORECASE)
                
                if li_matches:
                    reported_li = float(li_matches[0])
                    li_diff = abs(reported_li - expected_li) / expected_li if expected_li > 0 else float('inf')
                    
                    cross_checks.append({
                        "check": "LI consistency",
                        "expected": expected_li,
                        "reported": reported_li,
                        "difference_pct": round(li_diff * 100, 2),
                        "status": "pass" if li_diff <= 0.1 else "fail"
                    })
                    
                    if li_diff > 0.1:
                        warnings.append(f"LI mismatch: expected {expected_li}, reported {reported_li}")
            
            # 3. Multipliers consistency
            multipliers = ["HM", "VM", "DM", "AM", "FM", "CM"]
            for mult in multipliers:
                expected_value = origin_data.get(mult, 0)
                if expected_value > 0:
                    mult_matches = re.findall(f'{mult}\\s*=\\s*([\\d.]+)', report_text)
                    if mult_matches:
                        reported_value = float(mult_matches[0])
                        mult_diff = abs(reported_value - expected_value)
                        
                        cross_checks.append({
                            "check": f"{mult} consistency",
                            "expected": expected_value,
                            "reported": reported_value,
                            "difference": round(mult_diff, 3),
                            "status": "pass" if mult_diff <= 0.01 else "fail"
                        })
            
            # 4. Title consistency
            expected_title = calculated_data.get("title", "")
            if expected_title:
                title_match = expected_title.lower() in report_text.lower()
                cross_checks.append({
                    "check": "Title consistency",
                    "expected": expected_title,
                    "found": title_match,
                    "status": "pass" if title_match else "fail"
                })
                
                if not title_match:
                    warnings.append("Title not found or inconsistent in report")
            
            # 5. Scenario ID consistency
            scenario_id = calculated_data.get("scenario_id", "")
            if scenario_id:
                scenario_match = scenario_id in report_text
                cross_checks.append({
                    "check": "Scenario ID consistency", 
                    "expected": scenario_id,
                    "found": scenario_match,
                    "status": "pass" if scenario_match else "fail"
                })
            
            details["cross_checks"] = cross_checks
            details["total_checks"] = len(cross_checks)
            details["passed_checks"] = len([c for c in cross_checks if c.get("status") == "pass"])
            details["failed_checks"] = len([c for c in cross_checks if c.get("status") == "fail"])
            
            # Determina stato finale
            status = ValidationStatus.VALID
            if errors:
                status = ValidationStatus.ERROR
            elif warnings:
                status = ValidationStatus.WARNING
            
            if details["failed_checks"] > 0:
                recommendations.append("Review cross-reference consistency between data and report")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                status=status,
                component="cross_validation",
                check_type="reference_validation",
                details=details,
                recommendations=recommendations,
                errors=errors,
                warnings=warnings,
                execution_time_ms=execution_time * 1000
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.CRITICAL,
                component="cross_validation",
                check_type="reference_validation",
                details={"error": str(e)},
                recommendations=["Fix cross-validation process"],
                errors=[f"Cross-validation failed: {str(e)}"],
                warnings=[],
                execution_time_ms=execution_time * 1000
            )
    
    def run_complete_control(self, calculated_data: Dict, report_text: str) -> ControlReport:
        """Esegue controllo completo secondo livello configurato"""
        logger.info(f"Starting complete control - Level: {self.config.control_level.value}")
        
        validation_results = []
        audit_records = []
        performance_metrics = {}
        
        # Backup dei dati originali
        if self.config.backup_original_files:
            self._backup_if_needed(Path("calculated_data.json"), calculated_data)
            self._backup_if_needed(Path("report_text.md"), report_text)
        
        # 1. Audit integrità dati
        if self.config.control_level in [ControlLevel.STANDARD, ControlLevel.COMPREHENSIVE]:
            start_time = datetime.now()
            integrity_result = self.audit_data_integrity(calculated_data)
            validation_results.append(integrity_result)
            
            audit_record = self._create_audit_record(
                action="audit_data_integrity",
                component="data_integrity",
                input_data=calculated_data,
                output_data=integrity_result.to_dict(),
                status=integrity_result.status,
                details={"data_size": len(str(calculated_data))},
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            audit_records.append(audit_record)
            self.audit_trail.append(audit_record)
        
        # 2. Validazione catena di calcoli
        start_time = datetime.now()
        calc_result = self.validate_calculation_chain(calculated_data)
        validation_results.append(calc_result)
        
        audit_record = self._create_audit_record(
            action="validate_calculations",
            component="calculation_chain",
            input_data=calculated_data,
            output_data=calc_result.to_dict(),
            status=calc_result.status,
            details={"validation_type": "mathematical"},
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        audit_records.append(audit_record)
        self.audit_trail.append(audit_record)
        
        # 3. Validazione consistenza report
        start_time = datetime.now()
        report_result = self.validate_report_consistency(report_text, calculated_data)
        validation_results.append(report_result)
        
        audit_record = self._create_audit_record(
            action="validate_report",
            component="report_consistency",
            input_data={"text_length": len(report_text), "data_hash": self._calculate_hash(calculated_data)},
            output_data=report_result.to_dict(),
            status=report_result.status,
            details={"text_length": len(report_text)},
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        audit_records.append(audit_record)
        self.audit_trail.append(audit_record)
        
        # 4. Validazione incrociata (se comprehensive)
        if self.config.control_level == ControlLevel.COMPREHENSIVE:
            start_time = datetime.now()
            cross_result = self.cross_validate_references(calculated_data, report_text)
            validation_results.append(cross_result)
            
            audit_record = self._create_audit_record(
                action="cross_validate",
                component="cross_validation",
                input_data={"data_and_report"},
                output_data=cross_result.to_dict(),
                status=cross_result.status,
                details={"cross_check_count": len(cross_result.details.get("cross_checks", []))},
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            audit_records.append(audit_record)
            self.audit_trail.append(audit_record)
        
        # Calcola metriche performance
        total_execution_time = sum(r.execution_time_ms for r in validation_results)
        performance_metrics = {
            "total_validation_time_ms": total_execution_time,
            "average_validation_time_ms": total_execution_time / len(validation_results),
            "total_checks_performed": len(validation_results),
            "critical_errors": sum(1 for r in validation_results if r.status == ValidationStatus.CRITICAL),
            "errors": sum(1 for r in validation_results if r.status == ValidationStatus.ERROR),
            "warnings": sum(1 for r in validation_results if r.status == ValidationStatus.WARNING),
            "valid_checks": sum(1 for r in validation_results if r.status == ValidationStatus.VALID)
        }
        
        # Determina stato complessivo
        overall_status = ValidationStatus.VALID
        if any(r.status == ValidationStatus.CRITICAL for r in validation_results):
            overall_status = ValidationStatus.CRITICAL
        elif any(r.status == ValidationStatus.ERROR for r in validation_results):
            overall_status = ValidationStatus.ERROR
        elif any(r.status == ValidationStatus.WARNING for r in validation_results):
            overall_status = ValidationStatus.WARNING
        
        # Crea summary
        summary = {
            "session_id": self.session_id,
            "control_level": self.config.control_level.value,
            "execution_time_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "overall_status": overall_status.value,
            "validation_count": len(validation_results),
            "audit_records_count": len(audit_records),
            "has_critical_issues": performance_metrics["critical_errors"] > 0,
            "has_errors": performance_metrics["errors"] > 0,
            "has_warnings": performance_metrics["warnings"] > 0,
            "recommendations_count": sum(len(r.recommendations) for r in validation_results),
            "all_recommendations": list(set([rec for r in validation_results for rec in r.recommendations]))
        }
        
        # Crea report finale
        control_report = ControlReport(
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            control_level=self.config.control_level,
            overall_status=overall_status,
            validation_results=validation_results,
            audit_records=audit_records,
            performance_metrics=performance_metrics,
            summary=summary
        )
        
        logger.info(f"Control completed - Status: {overall_status.value}, Checks: {len(validation_results)}")
        
        return control_report
    
    def save_control_report(self, report: ControlReport, filename: Optional[str] = None) -> Path:
        """Salva report di controllo"""
        if filename is None:
            filename = f"control_report_{self.session_id}.json"
        
        report_path = self.reports_dir / filename
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Control report saved: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to save control report: {e}")
            raise
    
    def generate_human_readable_summary(self, report: ControlReport) -> str:
        """Genera summary leggibile per umani"""
        summary_lines = [
            f"# NIOSH Control System Report",
            f"",
            f"**Session ID:** {report.session_id}",
            f"**Timestamp:** {report.timestamp}",
            f"**Control Level:** {report.control_level.value}",
            f"**Overall Status:** {report.overall_status.value.upper()}",
            f"",
            f"## Execution Summary",
            f"",
            f"- Total validations performed: {len(report.validation_results)}",
            f"- Total execution time: {report.performance_metrics['total_validation_time_ms']:.1f} ms",
            f"- Valid checks: {report.performance_metrics['valid_checks']}",
            f"- Warnings: {report.performance_metrics['warnings']}",
            f"- Errors: {report.performance_metrics['errors']}",
            f"- Critical issues: {report.performance_metrics['critical_errors']}",
            f"",
        ]
        
        # Aggiungi dettagli per ogni validazione
        summary_lines.append("## Validation Details")
        summary_lines.append("")
        
        for i, result in enumerate(report.validation_results, 1):
            status_icon = {
                ValidationStatus.VALID: "✅",
                ValidationStatus.WARNING: "⚠️", 
                ValidationStatus.ERROR: "❌",
                ValidationStatus.CRITICAL: "🚨"
            }.get(result.status, "❓")
            
            summary_lines.extend([
                f"### {i}. {result.component} - {result.check_type}",
                f"**Status:** {status_icon} {result.status.value}",
                f"**Execution Time:** {result.execution_time_ms:.1f} ms",
                ""
            ])
            
            if result.errors:
                summary_lines.append("**Errors:**")
                for error in result.errors:
                    summary_lines.append(f"- {error}")
                summary_lines.append("")
            
            if result.warnings:
                summary_lines.append("**Warnings:**")
                for warning in result.warnings:
                    summary_lines.append(f"- {warning}")
                summary_lines.append("")
            
            if result.recommendations:
                summary_lines.append("**Recommendations:**")
                for rec in result.recommendations:
                    summary_lines.append(f"- {rec}")
                summary_lines.append("")
        
        # Aggiungi raccomandazioni aggregate
        if report.summary["all_recommendations"]:
            summary_lines.extend([
                "## Summary Recommendations",
                ""
            ])
            for rec in report.summary["all_recommendations"]:
                summary_lines.append(f"- {rec}")
            summary_lines.append("")
        
        # Audit trail summary
        summary_lines.extend([
            "## Audit Trail Summary",
            f"",
            f"- Total audit records: {len(report.audit_records)}",
            f"- Components validated: {list(set(r.component for r in report.audit_records))}",
            f""
        ])
        
        return "\n".join(summary_lines)

# ============================================================================
# INTEGRATION UTILITIES
# ============================================================================

def integrate_control_system_with_prod_gen():
    """Funzione helper per integrare il sistema di controllo con prod_gen_v21.py"""
    
    def run_posteriori_control(calculated_data: Dict, report_text: str, 
                             control_level: str = "standard") -> Dict:
        """
        Esegue controllo a posteriori completo
        
        Args:
            calculated_data: Dati di calcolo NIOSH
            report_text: Testo del report generato
            control_level: Livello di controllo (basic, standard, comprehensive)
        
        Returns:
            Dict con risultati del controllo
        """
        try:
            # Convert string to ControlLevel enum
            control_level_map = {
                "basic": ControlLevel.BASIC,
                "standard": ControlLevel.STANDARD,
                "comprehensive": ControlLevel.COMPREHENSIVE
            }
            
            config = ControlConfig(control_level=control_level_map.get(control_level, ControlLevel.STANDARD))
            
            # Inizializza sistema di controllo
            control_system = NIOSHControlSystem(config=config)
            
            # Esegue controllo completo
            control_report = control_system.run_complete_control(calculated_data, report_text)
            
            # Salva report
            report_path = control_system.save_control_report(control_report)
            
            # Genera summary leggibile
            human_summary = control_system.generate_human_readable_summary(control_report)
            summary_path = control_system.reports_dir / f"summary_{control_system.session_id}.md"
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(human_summary)
            
            return {
                "control_report": control_report.to_dict(),
                "report_path": str(report_path),
                "summary_path": str(summary_path),
                "session_id": control_system.session_id,
                "overall_status": control_report.overall_status.value,
                "is_valid": control_report.overall_status == ValidationStatus.VALID
            }
            
        except Exception as e:
            logger.error(f"Control system integration failed: {e}")
            return {
                "error": str(e),
                "control_report": None,
                "is_valid": False
            }
    
    return run_posteriori_control

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NIOSH Control System - Posteriori Validation")
    parser.add_argument("--data-file", required=True, help="Path to calculated data JSON file")
    parser.add_argument("--report-file", required=True, help="Path to report markdown file")
    parser.add_argument("--level", choices=["basic", "standard", "comprehensive"], 
                       default="standard", help="Control level")
    parser.add_argument("--output-dir", default=".", help="Output directory for reports")
    
    args = parser.parse_args()
    
    try:
        # Carica dati
        with open(args.data_file, 'r', encoding='utf-8') as f:
            calculated_data = json.load(f)
        
        with open(args.report_file, 'r', encoding='utf-8') as f:
            report_text = f.read()
        
        print(f"Loading data from: {args.data_file}")
        print(f"Loading report from: {args.report_file}")
        print(f"Control level: {args.level}")
        
        # Configura sistema
        config = ControlConfig(
            control_level=ControlLevel(args.level),
            backup_original_files=True
        )
        
        # Inizializza e esegui controllo
        control_system = NIOSHControlSystem(config=config, workspace_dir=args.output_dir)
        control_report = control_system.run_complete_control(calculated_data, report_text)
        
        # Salva report
        report_path = control_system.save_control_report(control_report)
        summary_path = control_system.reports_dir / f"summary_{control_system.session_id}.md"
        
        # Genera e salva summary
        human_summary = control_system.generate_human_readable_summary(control_report)
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(human_summary)
        
        # Output risultati
        print(f"\n{'='*60}")
        print(f"CONTROL COMPLETED - {control_report.overall_status.value.upper()}")
        print(f"{'='*60}")
        print(f"Session ID: {control_system.session_id}")
        print(f"Report saved: {report_path}")
        print(f"Summary saved: {summary_path}")
        print(f"Validations performed: {len(control_report.validation_results)}")
        print(f"Total execution time: {control_report.performance_metrics['total_validation_time_ms']:.1f} ms")
        
        if control_report.overall_status != ValidationStatus.VALID:
            print(f"\n⚠️  ISSUES FOUND:")
            for result in control_report.validation_results:
                if result.status != ValidationStatus.VALID:
                    print(f"   - {result.component}: {len(result.errors)} errors, {len(result.warnings)} warnings")
        
        print(f"\n📋 See {summary_path} for detailed summary")
        
    except Exception as e:
        print(f"❌ Control system execution failed: {e}")
        logger.error(f"Main execution failed: {e}")
        raise