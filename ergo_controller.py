#!/usr/bin/env python3
"""
ErgoController 3.0 - Validatore e Correttore Avanzato di Report NIOSH

Questo modulo fornisce analisi completa, validazione e correzione automatica
di report ergonomici in formato Markdown secondo le linee guida NIOSH.

Requisiti:
    pip install spacy ollama
    python -m spacy download en_core_web_sm

Utilizzo:
    python ergo_controller.py <report.md>
    python ergo_controller.py <report.md> --dry-run
    python ergo_controller.py <report.md> --verbose

Esempio:
    python ergo_controller.py warehouse_report.md
"""

import re
import json
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    import spacy
except ImportError:
    print("❌ Errore: spacy non installato. Eseguire: pip install spacy")
    sys.exit(1)

try:
    import ollama
except ImportError:
    print("⚠️  Attenzione: ollama non installato. Le correzioni LLM saranno disabilitate.")
    print("   Installare con: pip install ollama")
    ollama = None


# ========== STRUTTURE DATI ==========

@dataclass
class Section:
    """Rappresenta una sezione del report ergonomico"""
    name: str
    content: str
    numeric_data: Dict[str, float] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"Section(name='{self.name}', metrics={len(self.numeric_data)})"


@dataclass
class Issue:
    """Rappresenta un'inconsistenza rilevata"""
    severity: str  # 'critical', 'warning', 'info'
    section: str
    description: str
    expected: Any
    found: Any
    correction_type: str  # 'numeric', 'textual', 'structural'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Issue({self.severity}, {self.section})"


# ========== MODULO PARSING ==========

class MarkdownParser:
    """Gestisce il parsing di report Markdown strutturati"""
    
    SECTION_PATTERN = r'^##\s+(.+?)$'
    BULLET_PATTERN = r'^\s*[-*]\s+(.+?)$'
    
    def parse_sections(self, text: str) -> Dict[str, Section]:
        """Divide il report in sezioni basandosi sugli header ##"""
        sections = {}
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            header_match = re.match(self.SECTION_PATTERN, line, re.MULTILINE)
            
            if header_match:
                # Save previous section
                if current_section:
                    content = '\n'.join(current_content).strip()
                    sections[current_section] = Section(
                        name=current_section,
                        content=content
                    )
                
                # Start new section
                current_section = header_match.group(1).strip()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            content = '\n'.join(current_content).strip()
            sections[current_section] = Section(
                name=current_section,
                content=content
            )
        
        return sections
    
    def extract_numeric(self, text: str, key: str) -> Optional[float]:
        """Extracts specific numeric values from text"""
        # Try pattern: "Key: value unit"
        pattern = rf'{re.escape(key)}[:\s]+([0-9.]+)\s*(?:kg|cm|°|degrees?|lifts?/min)?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Try pattern in parentheses: "Key (V): value"
        pattern = rf'{re.escape(key)}[^:]*:\s*([0-9.]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        return None
    
    def extract_all_metrics(self, section: Section) -> Dict[str, float]:
        """Extracts all NIOSH parameters from a section"""
        metrics = {}
        
        # Standard NIOSH parameters
        keys = [
            'Load Weight', 'Vertical Location', 'Horizontal Distance',
            'Vertical Travel Distance', 'Asymmetry Angle', 'Frequency',
            'Duration', 'RWL', 'LI', 'Lifting Index',
            'HM', 'VM', 'DM', 'AM', 'FM', 'CM'
        ]
        
        for key in keys:
            value = self.extract_numeric(section.content, key)
            if value is not None:
                metrics[key] = value
        
        return metrics


# ========== MODULO NLP ==========

class SemanticInterpreter:
    """Interpretazione semantica utilizzando spaCy"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ Warning: spaCy model not found. Using simplified analysis without NLP.")
            print("   To install: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.height_mapping = {
            "floor": (0, 25),
            "ground": (0, 25),
            "ankle": (10, 30),
            "shin": (20, 40),
            "knee": (40, 60),
            "thigh": (60, 80),
            "waist": (70, 100),
            "hip": (75, 95),
            "belly button": (85, 105),
            "elbow": (90, 110),
            "chest": (110, 130),
            "shoulder": (120, 140),
            "eye level": (140, 165),
            "head": (150, 175),
            "top shelf": (140, 170),
            "overhead": (170, 200),
            "above head": (180, 210)
        }
        
        self.frequency_mapping = {
            "rarely": 0.2,
            "occasionally": 0.5,
            "sometimes": 1.0,
            "regularly": 2.0,
            "often": 3.0,
            "frequently": 4.5,
            "very frequently": 6.0,
            "constantly": 8.0,
            "continuous": 12.0
        }
        
        self.posture_indicators = {
            "bending": {"height_max": 50, "asymmetry_risk": True},
            "stooping": {"height_max": 60, "asymmetry_risk": True},
            "reaching up": {"height_min": 140, "horizontal_risk": True},
            "reaching out": {"horizontal_min": 50, "asymmetry_risk": False},
            "twisting": {"asymmetry_min": 30, "asymmetry_risk": True},
            "turning": {"asymmetry_min": 20, "asymmetry_risk": True},
            "squatting": {"height_max": 40, "asymmetry_risk": False},
            "kneeling": {"height_max": 30, "asymmetry_risk": False}
        }
    
    def interpret_height(self, text: str) -> Tuple[float, float]:
        """Converte espressioni linguistiche in intervalli di altezza"""
        text_lower = text.lower()
        
        # Check direct mappings
        for keyword, (low, high) in self.height_mapping.items():
            if keyword in text_lower:
                return (low, high)
        
        # Extract numeric ranges
        range_match = re.search(r'between\s+([0-9.]+)\s+(?:and|to)\s+([0-9.]+)', text_lower)
        if range_match:
            return (float(range_match.group(1)), float(range_match.group(2)))
        
        # Single numeric value
        num_match = re.search(r'([0-9.]+)\s*(?:cm|centimeters?)', text_lower)
        if num_match:
            val = float(num_match.group(1))
            return (val - 10, val + 10)  # Add tolerance
        
        # Default fallback
        return (70, 100)  # Waist height as default
    
    def interpret_frequency(self, text: str) -> float:
        """Interpreta descrizioni di frequenza"""
        text_lower = text.lower()
        
        # Check mappings
        for keyword, freq in self.frequency_mapping.items():
            if keyword in text_lower:
                return freq
        
        # Extract numeric frequency
        freq_match = re.search(r'([0-9.]+)\s*(?:lifts?|times?)/(?:min|minute)', text_lower)
        if freq_match:
            return float(freq_match.group(1))
        
        # Per hour conversions
        hour_match = re.search(r'([0-9.]+)\s*(?:lifts?|times?)/(?:hour|hr)', text_lower)
        if hour_match:
            return float(hour_match.group(1)) / 60
        
        return 1.0  # Default
    
    def extract_posture_indicators(self, text: str) -> Dict[str, bool]:
        """Identifies ergonomic postures mentioned"""
        text_lower = text.lower()
        indicators = {}
        
        for posture, _ in self.posture_indicators.items():
            indicators[posture] = posture in text_lower
        
        return indicators
    
    def extract_all_metrics(self, section: Section) -> Dict[str, float]:
        """Complete semantic extraction for a section"""
        metrics = {}
        content_lower = section.content.lower()
        
        # Extract keywords for later reference
        if self.nlp is not None:
            doc = self.nlp(content_lower)
            section.keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        else:
            # Fallback: simple keyword extraction without NLP
            import re
            # Extract common ergonomic terms
            ergonomic_terms = [
                'lift', 'lifting', 'load', 'weight', 'reach', 'bend', 'twist',
                'height', 'distance', 'frequency', 'duration', 'control',
                'horizontal', 'vertical', 'asymmetry', 'coupling'
            ]
            section.keywords = [term for term in ergonomic_terms if term in content_lower]
        
        # Try to infer height from context
        if any(word in content_lower for word in ['height', 'level', 'location']):
            height_range = self.interpret_height(section.content)
            metrics['inferred_height_min'] = height_range[0]
            metrics['inferred_height_max'] = height_range[1]
        
        # Try to infer frequency
        if 'frequency' in content_lower or 'lifts' in content_lower:
            metrics['inferred_frequency'] = self.interpret_frequency(section.content)
        
        return metrics


# ========== MODULO VALIDAZIONE ==========

class NIOSHValidator:
    """Applica le regole di coerenza NIOSH"""
    
    # NIOSH Constants
    LC = 23  # Load Constant (kg)
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues: List[Issue] = []
    
    def check_logical_consistency(self, sections: Dict[str, Section]) -> List[Issue]:
        """Complete logical consistency check"""
        self.issues = []
        
        if self.verbose:
            print("   Running validation checks...")
        
        # Run all validation checks
        self.issues.extend(self._check_weight_consistency(sections))
        self.issues.extend(self._check_height_coherence(sections))
        self.issues.extend(self._check_horizontal_distance(sections))
        self.issues.extend(self._check_asymmetry_coherence(sections))
        self.issues.extend(self._check_frequency_duration(sections))
        self.issues.extend(self._check_hazard_redesign_mapping(sections))
        self.issues.extend(self._validate_niosh_equation(sections))
        
        return self.issues
    
    def _check_weight_consistency(self, sections: Dict[str, Section]) -> List[Issue]:
        """Verifies weight consistency across sections"""
        issues = []
        
        # Extract weights from different sections
        job_desc_weight = None
        job_analysis_weight = None
        
        if 'Job Description' in sections:
            job_desc_weight = sections['Job Description'].numeric_data.get('Load Weight')
        
        if 'Job Analysis' in sections:
            job_analysis_weight = sections['Job Analysis'].numeric_data.get('Load Weight')
        
        # Check consistency
        if job_desc_weight and job_analysis_weight:
            tolerance = 0.05 * job_desc_weight  # 5% tolerance
            if abs(job_desc_weight - job_analysis_weight) > tolerance:
                issues.append(Issue(
                    severity='critical',
                    section='Job Analysis',
                    description='Load weight mismatch with Job Description',
                    expected=f"{job_desc_weight} kg",
                    found=f"{job_analysis_weight} kg",
                    correction_type='numeric',
                    metadata={'correct_value': job_desc_weight, 'key': 'Load Weight'}
                ))
        
        return issues
    
    def _check_height_coherence(self, sections: Dict[str, Section]) -> List[Issue]:
        """Verifies height coherence with descriptions"""
        issues = []
        
        if 'Job Description' not in sections or 'Job Analysis' not in sections:
            return issues
        
        context = sections['Job Description']
        analysis = sections['Job Analysis']
        
        context_lower = context.content.lower()
        vertical_loc = analysis.numeric_data.get('Vertical Location')
        
        if vertical_loc is None:
            return issues
        
        # Check floor level
        if any(word in context_lower for word in ['floor', 'ground', 'ground level']):
            if vertical_loc > 50:
                issues.append(Issue(
                    severity='warning',
                    section='Job Analysis',
                    description='Vertical location inconsistent with "floor level" mention',
                    expected='0-50 cm',
                    found=f'{vertical_loc} cm',
                    correction_type='numeric',
                    metadata={'correct_value': 25, 'key': 'Vertical Location'}
                ))
        
        # Check waist height
        if any(word in context_lower for word in ['waist', 'hip', 'belt']):
            if vertical_loc < 60 or vertical_loc > 110:
                issues.append(Issue(
                    severity='info',
                    section='Job Analysis',
                    description='Vertical location inconsistent with "waist height" mention',
                    expected='70-100 cm',
                    found=f'{vertical_loc} cm',
                    correction_type='numeric',
                    metadata={'correct_value': 85, 'key': 'Vertical Location'}
                ))
        
        # Check overhead
        if any(word in context_lower for word in ['overhead', 'above', 'top shelf']):
            if vertical_loc < 140:
                issues.append(Issue(
                    severity='warning',
                    section='Job Analysis',
                    description='Vertical location inconsistent with "overhead" mention',
                    expected='>140 cm',
                    found=f'{vertical_loc} cm',
                    correction_type='numeric',
                    metadata={'correct_value': 160, 'key': 'Vertical Location'}
                ))
        
        return issues
    
    def _check_horizontal_distance(self, sections: Dict[str, Section]) -> List[Issue]:
        """Validates horizontal distance logic"""
        issues = []
        
        if 'Job Analysis' not in sections:
            return issues
        
        analysis = sections['Job Analysis']
        horiz_dist = analysis.numeric_data.get('Horizontal Distance')
        
        if horiz_dist is None:
            return issues
        
        # Check excessive reach
        if horiz_dist > 63:
            # Should be mentioned in Hazard Assessment
            if 'Hazard Assessment' in sections:
                hazard_content = sections['Hazard Assessment'].content.lower()
                if 'reach' not in hazard_content and 'horizontal' not in hazard_content:
                    issues.append(Issue(
                        severity='warning',
                        section='Hazard Assessment',
                        description='Excessive horizontal distance not mentioned in hazards',
                        expected='Mention of reach/horizontal distance issue',
                        found='No mention found',
                        correction_type='textual',
                        metadata={'horiz_dist': horiz_dist}
                    ))
        
        # Check "close to body" claims
        if 'Job Description' in sections:
            context_lower = sections['Job Description'].content.lower()
            if 'close to body' in context_lower or 'near body' in context_lower:
                if horiz_dist > 40:
                    issues.append(Issue(
                        severity='info',
                        section='Job Analysis',
                        description='Horizontal distance inconsistent with "close to body"',
                        expected='<40 cm',
                        found=f'{horiz_dist} cm',
                        correction_type='numeric',
                        metadata={'correct_value': 35, 'key': 'Horizontal Distance'}
                    ))
        
        return issues
    
    def _check_asymmetry_coherence(self, sections: Dict[str, Section]) -> List[Issue]:
        """Checks asymmetry angle coherence"""
        issues = []
        
        if 'Job Analysis' not in sections:
            return issues
        
        analysis = sections['Job Analysis']
        asymmetry = analysis.numeric_data.get('Asymmetry Angle')
        
        if asymmetry is None:
            return issues
        
        # High asymmetry should be flagged
        if asymmetry > 45:
            hazard_mentioned = False
            redesign_mentioned = False
            
            if 'Hazard Assessment' in sections:
                hazard_content = sections['Hazard Assessment'].content.lower()
                if 'asym' in hazard_content or 'twist' in hazard_content or 'turn' in hazard_content:
                    hazard_mentioned = True
            
            if 'Redesign Suggestions' in sections:
                redesign_content = sections['Redesign Suggestions'].content.lower()
                if 'asym' in redesign_content or 'twist' in redesign_content or 'straight' in redesign_content:
                    redesign_mentioned = True
            
            if not hazard_mentioned:
                issues.append(Issue(
                    severity='warning',
                    section='Hazard Assessment',
                    description=f'High asymmetry angle ({asymmetry}°) not mentioned',
                    expected='Mention of twisting/asymmetry hazard',
                    found='No mention found',
                    correction_type='textual',
                    metadata={'asymmetry': asymmetry}
                ))
            
            if not redesign_mentioned:
                issues.append(Issue(
                    severity='info',
                    section='Redesign Suggestions',
                    description=f'No solution for high asymmetry ({asymmetry}°)',
                    expected='Suggestion to reduce twisting',
                    found='No mention found',
                    correction_type='textual',
                    metadata={'asymmetry': asymmetry}
                ))
        
        # Check twisting mentions
        if 'Job Description' in sections:
            context_lower = sections['Job Description'].content.lower()
            if 'twist' in context_lower or 'turn' in context_lower:
                if asymmetry == 0:
                    issues.append(Issue(
                        severity='warning',
                        section='Job Analysis',
                        description='Twisting mentioned but asymmetry angle is 0',
                        expected='>0°',
                        found='0°',
                        correction_type='numeric',
                        metadata={'correct_value': 30, 'key': 'Asymmetry Angle'}
                    ))
        
        return issues
    
    def _check_frequency_duration(self, sections: Dict[str, Section]) -> List[Issue]:
        """Verifies frequency-duration coherence"""
        issues = []
        
        if 'Job Analysis' not in sections:
            return issues
        
        analysis = sections['Job Analysis']
        frequency = analysis.numeric_data.get('Frequency')
        duration = analysis.numeric_data.get('Duration')
        
        if frequency is None or duration is None:
            return issues
        
        # High frequency should have short duration
        if frequency > 4 and duration > 2:
            issues.append(Issue(
                severity='info',
                section='Job Analysis',
                description='High frequency with long duration (ergonomic concern)',
                expected='Duration <2 hours for frequency >4/min',
                found=f'Frequency: {frequency}/min, Duration: {duration}h',
                correction_type='textual',
                metadata={'frequency': frequency, 'duration': duration}
            ))
        
        # Continuous work should have low frequency
        if 'Job Description' in sections:
            context_lower = sections['Job Description'].content.lower()
            if 'continuous' in context_lower or 'constantly' in context_lower:
                if frequency > 1:
                    issues.append(Issue(
                        severity='info',
                        section='Job Analysis',
                        description='"Continuous" work implies lower frequency',
                        expected='<1 lift/min',
                        found=f'{frequency} lifts/min',
                        correction_type='numeric',
                        metadata={'correct_value': 0.5, 'key': 'Frequency'}
                    ))
        
        return issues
    
    def _check_hazard_redesign_mapping(self, sections: Dict[str, Section]) -> List[Issue]:
        """Ensures every hazard has a corresponding solution"""
        issues = []
        
        if 'Hazard Assessment' not in sections or 'Redesign Suggestions' not in sections:
            return issues
        
        hazard_content = sections['Hazard Assessment'].content.lower()
        redesign_content = sections['Redesign Suggestions'].content.lower()
        
        # Keyword mappings
        hazard_solution_map = {
            'reach': ['closer', 'distance', 'proximity', 'workstation'],
            'twist': ['straight', 'align', 'orient', 'rotation'],
            'bend': ['height', 'raise', 'lift table', 'adjust'],
            'heavy': ['reduce', 'lighter', 'split', 'assist'],
            'awkward': ['posture', 'adjust', 'ergonomic', 'neutral'],
            'repetitive': ['frequency', 'rotation', 'break', 'variety']
        }
        
        for hazard_key, solution_keywords in hazard_solution_map.items():
            if hazard_key in hazard_content:
                solution_found = any(keyword in redesign_content for keyword in solution_keywords)
                if not solution_found:
                    issues.append(Issue(
                        severity='warning',
                        section='Redesign Suggestions',
                        description=f'No solution for "{hazard_key}" hazard',
                        expected=f'Suggestion addressing {hazard_key} (e.g., {", ".join(solution_keywords[:2])})',
                        found='No corresponding solution',
                        correction_type='textual',
                        metadata={'hazard': hazard_key, 'solutions': solution_keywords}
                    ))
        
        return issues
    
    def _validate_niosh_equation(self, sections: Dict[str, Section]) -> List[Issue]:
        """Validates RWL and LI calculations"""
        issues = []
        
        if 'Job Analysis' not in sections:
            return issues
        
        analysis = sections['Job Analysis']
        metrics = analysis.numeric_data
        
        # Extract required values
        load_weight = metrics.get('Load Weight')
        rwl = metrics.get('RWL')
        li = metrics.get('LI') or metrics.get('Lifting Index')
        
        # Extract multipliers
        hm = metrics.get('HM')
        vm = metrics.get('VM')
        dm = metrics.get('DM')
        am = metrics.get('AM')
        fm = metrics.get('FM')
        cm = metrics.get('CM')
        
        # Check RWL calculation if all multipliers available
        if all(m is not None for m in [hm, vm, dm, am, fm, cm]):
            calculated_rwl = self.LC * hm * vm * dm * am * fm * cm
            if rwl is not None:
                tolerance = 0.1 * calculated_rwl
                if abs(calculated_rwl - rwl) > tolerance:
                    issues.append(Issue(
                        severity='critical',
                        section='Job Analysis',
                        description='RWL calculation error',
                        expected=f'{calculated_rwl:.2f} kg',
                        found=f'{rwl} kg',
                        correction_type='numeric',
                        metadata={'correct_value': round(calculated_rwl, 2), 'key': 'RWL'}
                    ))
        
        # Check LI calculation
        if load_weight and rwl:
            calculated_li = load_weight / rwl
            if li is not None:
                tolerance = 0.1
                if abs(calculated_li - li) > tolerance:
                    issues.append(Issue(
                        severity='critical',
                        section='Job Analysis',
                        description='Lifting Index calculation error',
                        expected=f'{calculated_li:.2f}',
                        found=f'{li}',
                        correction_type='numeric',
                        metadata={'correct_value': round(calculated_li, 2), 'key': 'Lifting Index'}
                    ))
            
            # Check hazard flag
            if calculated_li > 1.0:
                if 'Hazard Assessment' in sections:
                    hazard_content = sections['Hazard Assessment'].content.lower()
                    if 'lifting index' not in hazard_content and 'li' not in hazard_content:
                        issues.append(Issue(
                            severity='warning',
                            section='Hazard Assessment',
                            description=f'LI > 1.0 ({calculated_li:.2f}) but not flagged',
                            expected='Warning about excessive Lifting Index',
                            found='No LI warning found',
                            correction_type='textual',
                            metadata={'li': calculated_li}
                        ))
        
        return issues


# ========== MODULO CORREZIONE ==========

class IntelligentCorrector:
    """Gestisce le correzioni automatiche"""
    
    def __init__(self, use_llm: bool = True, verbose: bool = False):
        self.use_llm = use_llm and ollama is not None
        self.verbose = verbose
        self.ollama_model = "llama3.2"
        
        if self.use_llm:
            try:
                # Test Ollama connection
                ollama.list()
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Ollama not available: {e}")
                self.use_llm = False
    
    def correct_section(
        self,
        section: Section,
        issue: Issue,
        context: Dict[str, Section]
    ) -> str:
        """Corrects a section based on issue type"""
        if issue.correction_type == 'numeric':
            return self._apply_numeric_correction(section, issue)
        elif issue.correction_type == 'textual' and self.use_llm:
            return self._apply_llm_correction(section, issue, context)
        elif issue.correction_type == 'textual':
            return self._apply_rule_based_text_correction(section, issue)
        else:
            return section.content
    
    def _apply_numeric_correction(self, section: Section, issue: Issue) -> str:
        """Rule-based numeric corrections"""
        content = section.content
        correct_value = issue.metadata.get('correct_value')
        key = issue.metadata.get('key')
        
        if correct_value is None or key is None:
            return content
        
        # Find and replace the value
        # Pattern: "Key: old_value unit"
        pattern = rf'({re.escape(key)}[:\s]+)[0-9.]+(\s*(?:kg|cm|°|degrees?|lifts?/min)?)'
        replacement = rf'\g<1>{correct_value}\g<2>'
        
        corrected = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        if corrected == content:
            # Try alternative pattern: "Key (abbr): value"
            pattern = rf'({re.escape(key)}[^:]*:\s*)[0-9.]+'
            replacement = rf'\g<1>{correct_value}'
            corrected = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return corrected
    
    def _apply_llm_correction(
        self,
        section: Section,
        issue: Issue,
        context: Dict[str, Section]
    ) -> str:
        """LLM-based text regeneration"""
        prompt = self._build_correction_prompt(section, issue, context)
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            corrected = response['message']['content'].strip()
            
            # Remove markdown code blocks if present
            corrected = re.sub(r'^```(?:markdown)?\n', '', corrected)
            corrected = re.sub(r'\n```$', '', corrected)
            
            return corrected
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  LLM correction failed: {e}")
            return self._apply_rule_based_text_correction(section, issue)
    
    def _build_correction_prompt(
        self,
        section: Section,
        issue: Issue,
        context: Dict[str, Section]
    ) -> str:
        """Builds correction prompt for LLM"""
        # Get related sections for context
        related_sections = []
        if issue.section == 'Hazard Assessment' and 'Job Analysis' in context:
            related_sections.append(('Job Analysis', context['Job Analysis'].content))
        if issue.section == 'Redesign Suggestions' and 'Hazard Assessment' in context:
            related_sections.append(('Hazard Assessment', context['Hazard Assessment'].content))
        
        related_text = ""
        if related_sections:
            related_text = "\n\nRelated sections for context:\n"
            for name, content in related_sections:
                related_text += f"\n### {name}\n{content[:300]}...\n"
        
        prompt = f"""You are an ergonomics expert following NIOSH guidelines.

Section to correct: {section.name}
Current text:
```
{section.content}
```

Issue detected: {issue.description}
Expected: {issue.expected}
Found: {issue.found}
{related_text}

Task: Rewrite this section maintaining:
- Technical NIOSH terminology
- Formal professional tone
- Numerical accuracy
- Logical consistency
- Bullet point format where appropriate

Corrected section (provide ONLY the corrected section content, no explanations):"""
        
        return prompt
    
    def _apply_rule_based_text_correction(self, section: Section, issue: Issue) -> str:
        """Fallback rule-based text corrections"""
        content = section.content
        
        # Add missing hazard mentions
        if 'not mentioned' in issue.description.lower():
            if 'reach' in issue.description.lower():
                addition = f"\n- Excessive horizontal reach distance ({issue.metadata.get('horiz_dist', 'N/A')} cm) increases biomechanical stress"
                content += addition
            
            elif 'asymmetry' in issue.description.lower():
                asymmetry = issue.metadata.get('asymmetry', 'N/A')
                addition = f"\n- High asymmetry angle ({asymmetry}°) creates twisting stress on the spine"
                content += addition
            
            elif 'lifting index' in issue.description.lower():
                li = issue.metadata.get('li', 'N/A')
                addition = f"\n- Lifting Index exceeds 1.0 (LI = {li:.2f}), indicating increased risk of injury"
                content += addition
        
        # Add missing redesign suggestions
        if issue.section == 'Redesign Suggestions' and 'no solution' in issue.description.lower():
            hazard = issue.metadata.get('hazard', '')
            solutions = issue.metadata.get('solutions', [])
            
            if hazard == 'reach' and solutions:
                addition = f"\n- Relocate materials closer to the worker to reduce horizontal reach distance"
                content += addition
            elif hazard == 'twist' and solutions:
                addition = f"\n- Reorient workstation to eliminate twisting and allow straight-ahead lifting"
                content += addition
            elif hazard == 'bend' and solutions:
                addition = f"\n- Raise load origin height using lift tables or elevated platforms"
                content += addition
            elif hazard == 'heavy' and solutions:
                addition = f"\n- Reduce load weight through splitting loads or mechanical assist devices"
                content += addition
        
        return content


# ========== CONTROLLORE PRINCIPALE ==========

class ErgoController:
    """Orchestratore principale"""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.parser = MarkdownParser()
        self.interpreter = SemanticInterpreter()
        self.validator = NIOSHValidator(verbose=verbose)
        self.corrector = IntelligentCorrector(use_llm=True, verbose=verbose)
    
    def run(self, file_path: str) -> None:
        """Pipeline completa di controllo e correzione"""
        
        # 0. Validate file exists
        path = Path(file_path)
        if not path.exists():
            print(f"❌ Error: File not found: {file_path}")
            sys.exit(1)
        
        # 1. Load and parse
        print("📖 Loading report...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        sections = self.parser.parse_sections(text)
        print(f"✅ Parsed {len(sections)} sections")
        
        if self.verbose:
            for name in sections.keys():
                print(f"   • {name}")
        
        # 2. Semantic analysis
        print("\n🧠 Performing semantic analysis...")
        for section in sections.values():
            # Extract numeric data
            section.numeric_data = self.parser.extract_all_metrics(section)
            
            # Add semantic interpretation
            inferred_metrics = self.interpreter.extract_all_metrics(section)
            section.numeric_data.update(inferred_metrics)
            
            if self.verbose and section.numeric_data:
                print(f"   • Extracted {len(section.numeric_data)} metrics from {section.name}")
        
        # 3. Validation
        print("\n🔍 Checking logical consistency...")
        issues = self.validator.check_logical_consistency(sections)
        
        if not issues:
            print("\n✅ Report fully consistent! No issues found.")
            return
        
        # 4. Report issues
        print(f"\n⚠️  Found {len(issues)} issue(s):\n")
        
        critical_count = sum(1 for i in issues if i.severity == 'critical')
        warning_count = sum(1 for i in issues if i.severity == 'warning')
        info_count = sum(1 for i in issues if i.severity == 'info')
        
        for i, issue in enumerate(issues, 1):
            severity_icon = {
                'critical': '🔴',
                'warning': '⚠️ ',
                'info': 'ℹ️ '
            }.get(issue.severity, '•')
            
            print(f"{i}. {severity_icon} [{issue.severity.upper()}] {issue.section}")
            print(f"   {issue.description}")
            print(f"   Expected: {issue.expected}")
            print(f"   Found: {issue.found}")
            print()
        
        print(f"Summary: {critical_count} critical, {warning_count} warnings, {info_count} info\n")
        
        # 5. Dry-run check
        if self.dry_run:
            print("🔍 Dry-run mode: No corrections applied")
            return
        
        # 6. Backup original
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        print(f"💾 Backup saved: {backup_path}\n")
        
        # 7. Apply corrections
        print("🔧 Applying corrections...")
        corrections_applied = 0
        
        for issue in issues:
            section_name = issue.section
            if section_name not in sections:
                continue
            
            try:
                corrected = self.corrector.correct_section(
                    sections[section_name],
                    issue,
                    sections
                )
                
                if corrected != sections[section_name].content:
                    sections[section_name].content = corrected
                    corrections_applied += 1
                    
                    correction_icon = {
                        'numeric': '🔢',
                        'textual': '📝',
                        'structural': '🏗️'
                    }.get(issue.correction_type, '✓')
                    
                    print(f"   {correction_icon} Corrected {section_name} ({issue.correction_type})")
            
            except Exception as e:
                print(f"   ❌ Failed to correct {section_name}: {e}")
        
        # 8. Rebuild and save
        corrected_report = self._rebuild_report(sections, text)
        output_path = path.stem + "_corrected" + path.suffix
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(corrected_report)
        
        print(f"\n✅ Applied {corrections_applied} correction(s)")
        print(f"📄 Corrected report saved as: {output_path}")
    
    def _rebuild_report(self, sections: Dict[str, Section], original_text: str) -> str:
        """Rebuilds the Markdown report"""
        # Extract header (everything before first ##)
        header_match = re.search(r'^(.*?)(?=^##\s)', original_text, re.MULTILINE | re.DOTALL)
        header = header_match.group(1).strip() if header_match else ""
        
        # Rebuild sections
        rebuilt = header + "\n\n" if header else ""
        
        for section in sections.values():
            rebuilt += f"## {section.name}\n\n"
            rebuilt += section.content.strip() + "\n\n"
        
        return rebuilt.strip() + "\n"


# ========== PUNTO DI ACCESSO CLI ==========

def main():
    """
    Punto di accesso principale per ErgoController.
    
    Esempi di utilizzo:
        # Validazione e correzione di base
        python ergo_controller.py warehouse_report.md
        
        # Dry-run (mostra problemi senza correggere)
        python ergo_controller.py warehouse_report.md --dry-run
        
        # Output dettagliato
        python ergo_controller.py warehouse_report.md --verbose
        
        # Combina flag
        python ergo_controller.py warehouse_report.md --dry-run --verbose
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ErgoController 3.0 - NIOSH Report Validator and Corrector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s report.md                    Validate and correct report
  %(prog)s report.md --dry-run          Show issues without correcting
  %(prog)s report.md --verbose          Show detailed processing info
  %(prog)s report.md --dry-run -v       Dry-run with verbose output

Report Structure Required:
  The report must be in Markdown format with sections marked by ## headers.
  Expected sections: Job Description, Job Analysis, Hazard Assessment, 
  Redesign Suggestions, Comments.
        """
    )
    
    parser.add_argument(
        'file',
        help='Path to the Markdown report file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show issues without applying corrections'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 3.0'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print("   ErgoController 3.0 - NIOSH Report Analysis Tool")
    print("=" * 60)
    print()
    
    # Run controller
    try:
        controller = ErgoController(dry_run=args.dry_run, verbose=args.verbose)
        controller.run(args.file)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("   Analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()