"""
NIOSH Control System Logger - Logging avanzato e gestione errori
"""

import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import functools
from contextlib import contextmanager

class NIOSHControlLogger:
    """Logger specializzato per il sistema di controllo NIOSH"""
    
    def __init__(self, log_dir: str = "control_logs", session_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.session_id = session_id or f"LOG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup logger principale
        self.logger = logging.getLogger(f"niosh_control_{self.session_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler per log dettagliato
        self.detail_log_file = self.log_dir / f"control_detail_{self.session_id}.log"
        detail_handler = logging.FileHandler(self.detail_log_file, encoding='utf-8')
        detail_handler.setLevel(logging.DEBUG)
        detail_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(detail_handler)
        
        # File handler per log semplice
        self.simple_log_file = self.log_dir / f"control_simple_{self.session_id}.log"
        simple_handler = logging.FileHandler(self.simple_log_file, encoding='utf-8')
        simple_handler.setLevel(logging.INFO)
        simple_handler.setFormatter(simple_formatter)
        self.logger.addHandler(simple_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # Metriche
        self.metrics = {
            "errors": 0,
            "warnings": 0,
            "info": 0,
            "debug": 0,
            "session_start": datetime.now(timezone.utc).isoformat(),
            "last_activity": None
        }
        
        # Error registry
        self.error_registry = []
        
        self.logger.info(f"NIOSH Control Logger initialized - Session: {self.session_id}")
    
    def _update_metrics(self, level: str):
        """Aggiorna metriche di logging"""
        self.metrics[level] += 1
        self.metrics["last_activity"] = datetime.now(timezone.utc).isoformat()
    
    def debug(self, message: str, **kwargs):
        """Log debug"""
        self.logger.debug(message, extra=kwargs)
        self._update_metrics("debug")
    
    def info(self, message: str, **kwargs):
        """Log info"""
        self.logger.info(message, extra=kwargs)
        self._update_metrics("info")
    
    def warning(self, message: str, **kwargs):
        """Log warning"""
        self.logger.warning(message, extra=kwargs)
        self._update_metrics("warnings")
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error con tracing completo"""
        error_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_message": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None,
            **kwargs
        }
        
        # Registra errore
        self.error_registry.append(error_data)
        
        # Log standard
        self.logger.error(f"{message} - {str(exception) if exception else 'No exception'}", 
                         extra=kwargs)
        self._update_metrics("errors")
        
        # Salva errore su file separato
        self._save_error_to_file(error_data)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical con tracing completo"""
        error_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_message": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None,
            "severity": "critical",
            **kwargs
        }
        
        # Registra errore critico
        self.error_registry.append(error_data)
        
        # Log standard
        self.logger.critical(f"CRITICAL: {message} - {str(exception) if exception else 'No exception'}", 
                           extra=kwargs)
        self._update_metrics("errors")
        
        # Salva errore su file separato
        self._save_error_to_file(error_data)
    
    def _save_error_to_file(self, error_data: Dict):
        """Salva dati errore su file dedicato"""
        try:
            error_file = self.log_dir / f"errors_{self.session_id}.jsonl"
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_data, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.warning(f"Failed to save error to file: {e}")
    
    def log_validation_start(self, component: str, check_type: str, input_size: int = 0):
        """Log inizio validazione"""
        self.info(f"Starting validation - Component: {component}, Check: {check_type}, Input size: {input_size}")
    
    def log_validation_end(self, component: str, check_type: str, status: str, 
                         execution_time_ms: float, errors_count: int = 0, warnings_count: int = 0):
        """Log fine validazione"""
        self.info(f"Validation completed - Component: {component}, Check: {check_type}, "
                 f"Status: {status}, Time: {execution_time_ms:.1f}ms, "
                 f"Errors: {errors_count}, Warnings: {warnings_count}")
    
    def log_control_session_start(self, session_id: str, control_level: str):
        """Log inizio sessione di controllo"""
        self.info(f"Control session started - ID: {session_id}, Level: {control_level}")
    
    def log_control_session_end(self, session_id: str, overall_status: str, 
                              total_validations: int, total_time_ms: float):
        """Log fine sessione di controllo"""
        self.info(f"Control session completed - ID: {session_id}, Status: {overall_status}, "
                 f"Validations: {total_validations}, Total time: {total_time_ms:.1f}ms")
    
    def get_metrics_summary(self) -> Dict:
        """Restituisce summary delle metriche"""
        return {
            "session_id": self.session_id,
            "metrics": self.metrics.copy(),
            "error_count": len(self.error_registry),
            "last_errors": self.error_registry[-5:] if self.error_registry else []
        }
    
    def save_metrics_report(self):
        """Salva report metriche su file"""
        try:
            metrics_file = self.log_dir / f"metrics_{self.session_id}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.get_metrics_summary(), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.warning(f"Failed to save metrics report: {e}")
    
    def export_session_log(self, format: str = "json") -> str:
        """Esporta log completo sessione"""
        try:
            session_data = {
                "session_id": self.session_id,
                "metrics": self.metrics,
                "error_registry": self.error_registry,
                "log_files": {
                    "detail": str(self.detail_log_file),
                    "simple": str(self.simple_log_file)
                }
            }
            
            if format == "json":
                return json.dumps(session_data, indent=2, ensure_ascii=False, default=str)
            else:
                # Formato testo
                lines = [
                    f"NIOSH Control Session Log - {self.session_id}",
                    "=" * 60,
                    f"Session started: {self.metrics['session_start']}",
                    f"Last activity: {self.metrics['last_activity']}",
                    f"Total entries: {sum(self.metrics[k] for k in ['errors', 'warnings', 'info', 'debug'])}",
                    f"Errors: {self.metrics['errors']}",
                    f"Warnings: {self.metrics['warnings']}",
                    f"Info: {self.metrics['info']}",
                    f"Debug: {self.metrics['debug']}",
                    "",
                    "Recent errors:" if self.error_registry else "No errors registered."
                ]
                
                for error in self.error_registry[-5:]:
                    lines.extend([
                        f"- {error['timestamp']}: {error['message']}",
                        f"  Type: {error.get('exception_type', 'N/A')}",
                        f"  Message: {error.get('exception_message', 'N/A')}",
                        ""
                    ])
                
                return "\n".join(lines)
                
        except Exception as e:
            return f"Error exporting session log: {str(e)}"

# Context manager per operazioni di controllo
@contextmanager
def control_operation(logger: NIOSHControlLogger, operation_name: str, component: str):
    """Context manager per log automatico operazioni di controllo"""
    start_time = datetime.now()
    logger.info(f"Starting {operation_name} - Component: {component}")
    
    try:
        yield
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Completed {operation_name} - Component: {component}, Time: {execution_time:.1f}ms")
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Failed {operation_name} - Component: {component}, Time: {execution_time:.1f}ms", 
                    exception=e)
        raise

# Decorator per logging automatico funzioni
def log_control_function(logger: NIOSHControlLogger):
    """Decorator per logging automatico di funzioni di controllo"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            func_name = func.__name__
            class_name = args[0].__class__.__name__ if args and hasattr(args[0], '__class__') else "Unknown"
            
            logger.info(f"Executing function: {class_name}.{func_name}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(f"Function completed: {class_name}.{func_name}, Time: {execution_time:.1f}ms")
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"Function failed: {class_name}.{func_name}, Time: {execution_time:.1f}ms", 
                           exception=e)
                raise
                
        return wrapper
    return decorator

# Logger singleton globale
_global_logger = None

def get_control_logger(session_id: Optional[str] = None) -> NIOSHControlLogger:
    """Ottiene logger singleton globale"""
    global _global_logger
    if _global_logger is None or (_global_logger.session_id != session_id and session_id is not None):
        _global_logger = NIOSHControlLogger(session_id=session_id)
    return _global_logger

def setup_control_logging(log_dir: str = "control_logs", level: str = "INFO") -> NIOSHControlLogger:
    """Setup iniziale logging controllo"""
    logger = get_control_logger()
    
    # Imposta livello logging
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Aggiorna handler console
    for handler in logger.logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(log_level)
    
    return logger

# Classe per gestione errori specifici del sistema controllo
class NIOSHControlError(Exception):
    """Classe base per errori del sistema controllo"""
    def __init__(self, message: str, component: str = "unknown", error_code: str = None):
        super().__init__(message)
        self.component = component
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)

class ValidationError(NIOSHControlError):
    """Errore di validazione"""
    pass

class CalculationError(NIOSHControlError):
    """Errore di calcolo"""
    pass

class DataIntegrityError(NIOSHControlError):
    """Errore di integrità dati"""
    pass

class ConfigurationError(NIOSHControlError):
    """Errore di configurazione"""
    pass

# Funzione utility per gestione errori
def handle_control_error(logger: NIOSHControlLogger, error: Exception, 
                       context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Gestione centralizzata errori"""
    
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": context or {},
        "traceback": traceback.format_exc()
    }
    
    # Log appropriato basato sul tipo di errore
    if isinstance(error, (ValidationError, CalculationError, DataIntegrityError)):
        logger.error(f"Control error in {getattr(error, 'component', 'unknown')}: {str(error)}", 
                    exception=error, **(context or {}))
    elif isinstance(error, ConfigurationError):
        logger.critical(f"Configuration error: {str(error)}", exception=error, **(context or {}))
    else:
        logger.error(f"Unexpected error: {str(error)}", exception=error, **(context or {}))
    
    return error_info

if __name__ == "__main__":
    # Test del logger
    logger = setup_control_logging()
    
    logger.info("Testing NIOSH Control Logger")
    
    # Test logging levels
    logger.debug("Debug message test")
    logger.info("Info message test")
    logger.warning("Warning message test")
    
    # Test error handling
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        logger.error("Test error logging", exception=e, test_context="logger_test")
    
    # Test context manager
    with control_operation(logger, "test operation", "test_component"):
        logger.info("Inside controlled operation")
    
    # Test decorator
    @log_control_function(logger)
    def test_function(x, y):
        return x + y
    
    result = test_function(5, 3)
    logger.info(f"Function result: {result}")
    
    # Export session
    print("\nSession log export:")
    print(logger.export_session_log())
    
    # Save metrics
    logger.save_metrics_report()
    print(f"\nMetrics saved to: {logger.log_dir}/metrics_{logger.session_id}.json")