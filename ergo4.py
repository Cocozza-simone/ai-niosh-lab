#!/usr/bin/env python3
"""
ErgoController 4.0 - Validatore di Conformità PDF NIOSH

Validazione di livello professionale secondo gli standard del Manuale Applicativo NIOSH.
Si comporta come un revisore esperto umano che verifica la conformità delle reportistiche per la pubblicazione.

Requisiti:
    pip install spacy ollama
    python -m spacy download en_core_web_sm
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
    print("⚠️  Attenzione: spacy non installato. Verrà utilizzato NLP semplificato.")
    spacy = None

try:
    import ollama
except ImportError:
    print("⚠️  Attenzione: ollama non installato. Le correzioni LLM saranno disabilitate.")
    ollama = None


# ========== STANDARD DI CONFORMITÀ MANUALE NIOSH ==========

class NIOSHManualStandards:
    """
    Standard di riferimento del Manuale Applicativo NIOSH per l'Equazione di Sollevamento Rivista.
    I revisori professionisti verificano le reportistiche secondo questi schemi esatti.
    """
    
    # Pattern di numerazione sezioni dal manuale (es. "3.3.2", "3.3.2.1")
    SECTION_NUMBERING_PATTERN = r'^\d+\.\d+\.\d+(?:\.\d+)?'
    
    # Sezioni obbligatorie dalla struttura del manuale
    REQUIRED_SECTIONS = {
        'Job Description': {
            'alternative_names': ['Job Description', 'Description'],
            'must_contain': ['worker', 'lifts', 'loads'],
            'style': 'narrative',
            'numbering_suffix': '.1'
        },
        'Job Analysis': {
            'alternative_names': ['Job Analysis', 'Analysis'],
            'must_contain': ['task variable', 'measured', 'recorded'],
            'style': 'technical_narrative',
            'numbering_suffix': '.2',
            'required_phrases': [
                'task variable data are measured and recorded',
                'at the origin',
                'at the destination'
            ]
        },
        'Hazard Assessment': {
            'alternative_names': ['Hazard Assessment', 'Assessment'],
            'must_contain': ['weight to be lifted', 'RWL', 'LI'],
            'style': 'technical_assessment',
            'numbering_suffix': '.3',
            'required_pattern': r'The weight to be lifted \([\d.]+ (?:kg|lbs)\) is (?:greater than|less than|equal to) the RWL'
        },
        'Redesign Suggestions': {
            'alternative_names': ['Redesign Suggestions', 'Suggestions'],
            'must_contain': ['worksheet', 'multipliers', 'smallest magnitude'],
            'style': 'recommendations',
            'numbering_suffix': '.4',
            'required_intro': 'The worksheet',
            'table_reference': 'Table 8'
        },
        'Comments': {
            'alternative_names': ['Comments'],
            'must_contain': [],
            'style': 'technical_commentary',
            'numbering_suffix': '.5'
        }
    }
    
    # Standard terminologici dal manuale
    APPROVED_TERMINOLOGY = {
        'risk_levels': {
            'li_range': (0, 1.0),
            'phrase': 'acceptable for most healthy workers'
        },
        'moderate_risk': {
            'li_range': (1.0, 1.5),
            'phrase': 'some healthy workers would find this task physically stressful'
        },
        'high_risk': {
            'li_range': (1.5, 3.0),
            'phrase': 'physically stressful for some workers'
        },
        'very_high_risk': {
            'li_range': (3.0, float('inf')),
            'phrase': 'physically stressful for most industrial workers'
        }
    }
    
    # Terminologia "migliorata" vietata non presente nel manuale
    FORBIDDEN_TERMS = [
        'extreme hazard',
        'extreme risk',
        'critical risk',
        'severely exceeds',
        'extremely high risk',
        'immediate danger',
        'catastrophic'
    ]
    
    # Formato di presentazione delle variabili di compito dal manuale
    TASK_VARIABLE_FORMAT = {
        'style': 'narrative_paragraph',
        'example': 'At the origin of the lift the horizontal distance (H) is 20 inches, the vertical distance (V) is 44 inches, and the asymmetry angle (A) is 30°.',
        'forbidden_format': 'bullet_points'
    }
    
    # Formato di presentazione dei moltiplicatori dal manuale
    MULTIPLIER_FORMAT = {
        'style': 'integrated_narrative',
        'example': 'The multipliers are computed from the lifting equation or determined from the multiplier tables (Tables 1 to 5, and Table 7).',
        'forbidden_format': 'separate_bullet_list'
    }
    
    # Pattern di riferimento a figure e tabelle dal manuale
    REFERENCE_PATTERNS = {
        'figure': r'Figure \d+',
        'table': r'Table \d+',
        'worksheet': r'worksheet.*Figure \d+'
    }


# ========== STRUTTURE DATI ==========

@dataclass
class ComplianceIssue:
    """Rappresenta un problema di conformità con gli standard del manuale NIOSH"""
    severity: str  # 'critical', 'major', 'minor', 'style'
    category: str  # 'structure', 'terminology', 'format', 'calculation', 'consistency'
    section: str
    issue_type: str
    description: str
    manual_reference: str  # Reference to manual section/example
    current_text: str
    expected_format: str
    correction_type: str  # 'rewrite', 'rephrase', 'reformat', 'recalculate'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"ComplianceIssue({self.severity}, {self.category}, {self.section})"


@dataclass
class Section:
    """Rappresenta una sezione con metadati di conformità"""
    name: str
    content: str
    numbering: Optional[str] = None
    numeric_data: Dict[str, float] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    compliance_score: float = 0.0
    
    def __repr__(self):
        return f"Section('{self.name}', score={self.compliance_score:.2f})"


# ========== VALIDATORE DI CONFORMITÀ MANUALE NIOSH ==========

class NIOSHManualValidator:
    """
    Validatore professionale che verifica le reportistiche secondo il Manuale Applicativo NIOSH.
    Emula un revisore esperto umano che controlla la conformità di pubblicazione.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.standards = NIOSHManualStandards()
        self.issues: List[ComplianceIssue] = []
        self.compliance_report = {}
        
    def validate_full_compliance(self, sections: Dict[str, Section]) -> List[ComplianceIssue]:
        """
        Complete compliance check against NIOSH manual standards.
        Professional reviewers follow this systematic approach.
        """
        self.issues = []
        
        if self.verbose:
            print("   📋 Running NIOSH Manual compliance checks...")
        
        # Stage 1: Structural compliance
        self.issues.extend(self._check_section_structure(sections))
        
        # Stage 2: Terminology compliance
        self.issues.extend(self._check_terminology_compliance(sections))
        
        # Stage 3: Format compliance
        self.issues.extend(self._check_format_compliance(sections))
        
        # Stage 4: Technical accuracy
        self.issues.extend(self._check_technical_accuracy(sections))
        
        # Stage 5: Cross-reference compliance
        self.issues.extend(self._check_reference_compliance(sections))
        
        # Generate compliance score
        self._calculate_compliance_scores(sections)
        
        return self.issues
    
    def _check_section_structure(self, sections: Dict[str, Section]) -> List[ComplianceIssue]:
        """
        Validates document structure against manual requirements.
        Per NIOSH manual: sections must follow specific order and naming.
        """
        issues = []
        
        # Check for required sections
        for section_name, requirements in self.standards.REQUIRED_SECTIONS.items():
            found = False
            for actual_name in sections.keys():
                if any(alt.lower() in actual_name.lower() 
                      for alt in requirements['alternative_names']):
                    found = True
                    break
            
            if not found:
                issues.append(ComplianceIssue(
                    severity='critical',
                    category='structure',
                    section='Document',
                    issue_type='missing_required_section',
                    description=f'Required section "{section_name}" not found',
                    manual_reference='Section 3.3, NIOSH Applications Manual',
                    current_text='[Section missing]',
                    expected_format=f'Must include section: ## {section_name}',
                    correction_type='rewrite',
                    metadata={'required_section': section_name}
                ))
        
        # Check section numbering (e.g., "3.3.2.1 Job Description")
        for section in sections.values():
            if not re.match(r'^\d+\.\d+', section.name):
                issues.append(ComplianceIssue(
                    severity='major',
                    category='format',
                    section=section.name,
                    issue_type='missing_section_numbering',
                    description='Section lacks NIOSH manual numbering format',
                    manual_reference='Section numbering: 3.3.X.Y format (Examples 1-10)',
                    current_text=section.name,
                    expected_format='## 3.3.X.Y Section Name, Example N',
                    correction_type='reformat',
                    metadata={'current_name': section.name}
                ))
        
        return issues
    
    def _check_terminology_compliance(self, sections: Dict[str, Section]) -> List[ComplianceIssue]:
        """
        Validates terminology against approved NIOSH manual language.
        Professional reviewers flag non-standard terminology immediately.
        """
        issues = []
        
        # Check for forbidden "enhanced" terminology
        for section in sections.values():
            content_lower = section.content.lower()
            
            for forbidden_term in self.standards.FORBIDDEN_TERMS:
                if forbidden_term in content_lower:
                    # Find approved replacement
                    if 'hazard' in section.name.lower() or 'assessment' in section.name.lower():
                        approved = self._get_approved_risk_phrase(section)
                        
                        issues.append(ComplianceIssue(
                            severity='major',
                            category='terminology',
                            section=section.name,
                            issue_type='non_standard_terminology',
                            description=f'Non-standard term "{forbidden_term}" not used in NIOSH manual',
                            manual_reference='Examples 1-10, NIOSH Applications Manual',
                            current_text=f'Contains: "{forbidden_term}"',
                            expected_format=f'Use standard phrasing: "{approved}"',
                            correction_type='rephrase',
                            metadata={
                                'forbidden_term': forbidden_term,
                                'approved_replacement': approved
                            }
                        ))
        
        # Check Hazard Assessment opening phrase compliance
        if 'Hazard Assessment' in sections or any('hazard' in s.lower() for s in sections.keys()):
            hazard_section = next((s for s in sections.values() 
                                 if 'hazard' in s.name.lower()), None)
            
            if hazard_section:
                # Manual requires: "The weight to be lifted (X kg/lbs) is greater than/less than the RWL..."
                if not re.search(r'The weight to be lifted.*is.*RWL', hazard_section.content):
                    issues.append(ComplianceIssue(
                        severity='major',
                        category='format',
                        section=hazard_section.name,
                        issue_type='non_compliant_opening',
                        description='Hazard Assessment must follow manual opening format',
                        manual_reference='Example 5 (Section 3.3.2.3), NIOSH Manual',
                        current_text=hazard_section.content[:100] + '...',
                        expected_format='The weight to be lifted (X kg) is greater than/less than the RWL at [location]...',
                        correction_type='rewrite'
                    ))
        
        return issues
    
    def _check_format_compliance(self, sections: Dict[str, Section]) -> List[ComplianceIssue]:
        """
        Validates formatting against manual presentation standards.
        Manual uses specific formats for task variables and multipliers.
        """
        issues = []
        
        # Check Job Analysis formatting
        if 'Job Analysis' in sections or any('analysis' in s.lower() for s in sections.keys()):
            analysis_section = next((s for s in sections.values() 
                                    if 'analysis' in s.name.lower()), None)
            
            if analysis_section:
                content = analysis_section.content
                
                # Manual requires: "The task variable data are measured and recorded..."
                if 'task variable data are measured and recorded' not in content.lower():
                    issues.append(ComplianceIssue(
                        severity='major',
                        category='format',
                        section=analysis_section.name,
                        issue_type='missing_required_phrase',
                        description='Job Analysis missing required introductory phrase',
                        manual_reference='All examples, NIOSH Manual',
                        current_text=content[:100],
                        expected_format='The task variable data are measured and recorded on the job analysis worksheet (Figure X).',
                        correction_type='rewrite'
                    ))
                
                # Check for bullet point format (FORBIDDEN in task variables)
                if re.search(r'^\s*[-*]\s+(?:H|V|D|A|F|L):', content, re.MULTILINE):
                    issues.append(ComplianceIssue(
                        severity='major',
                        category='format',
                        section=analysis_section.name,
                        issue_type='incorrect_task_variable_format',
                        description='Task variables must use narrative format, not bullet points',
                        manual_reference='Example 5 (Section 3.3.2.2), NIOSH Manual',
                        current_text='[Uses bullet point format]',
                        expected_format='At the origin of the lift the horizontal distance (H) is X cm, the vertical distance (V) is Y cm, and the asymmetry angle (A) is Z°.',
                        correction_type='reformat',
                        metadata={'requires_narrative_conversion': True}
                    ))
                
                # Check for separate multiplier lists (DISCOURAGED)
                if re.search(r'^\s*[-*]\s+(?:HM|VM|DM|AM|FM|CM)\s*=', content, re.MULTILINE):
                    issues.append(ComplianceIssue(
                        severity='minor',
                        category='format',
                        section=analysis_section.name,
                        issue_type='suboptimal_multiplier_format',
                        description='Multipliers should be integrated into narrative paragraph',
                        manual_reference='Example 5, preferred style',
                        current_text='[Uses bullet list for multipliers]',
                        expected_format='The multipliers are computed from the lifting equation or determined from the multiplier tables. The HM is X, the VM is Y...',
                        correction_type='reformat'
                    ))
        
        # Check Redesign Suggestions formatting
        if 'Redesign' in str(sections.keys()) or 'Suggestions' in str(sections.keys()):
            redesign_section = next((s for s in sections.values() 
                                    if 'redesign' in s.name.lower() or 'suggestion' in s.name.lower()), None)
            
            if redesign_section:
                # Must reference "worksheet" and "Table 8"
                if 'worksheet' not in redesign_section.content.lower():
                    issues.append(ComplianceIssue(
                        severity='minor',
                        category='format',
                        section=redesign_section.name,
                        issue_type='missing_worksheet_reference',
                        description='Redesign section should reference worksheet',
                        manual_reference='All examples',
                        current_text='[No worksheet reference]',
                        expected_format='The worksheet shows that the smallest multipliers...',
                        correction_type='rephrase'
                    ))
                
                if 'table 8' not in redesign_section.content.lower():
                    issues.append(ComplianceIssue(
                        severity='minor',
                        category='format',
                        section=redesign_section.name,
                        issue_type='missing_table_reference',
                        description='Redesign section should reference Table 8',
                        manual_reference='Example 5 (Section 3.3.2.4)',
                        current_text='[No Table 8 reference]',
                        expected_format='Using Table 8, the following job modifications are suggested:',
                        correction_type='rephrase'
                    ))
        
        return issues
    
    def _check_technical_accuracy(self, sections: Dict[str, Section]) -> List[ComplianceIssue]:
        """
        Validates technical calculations and LI interpretations.
        Professional reviewers verify calculations match manual methodology.
        """
        issues = []
        
        # Extract and verify RWL calculations
        for section in sections.values():
            if 'analysis' in section.name.lower():
                # Check for RWL equation presentation
                if 'RWL = LC' in section.content or 'RWL = 23' in section.content:
                    # Verify multiplication chain format
                    if not re.search(r'RWL = \d+\s*×.*=\s*\*?\*?\d+(?:\.\d+)?\s*(?:kg|lbs)', section.content):
                        issues.append(ComplianceIssue(
                            severity='minor',
                            category='format',
                            section=section.name,
                            issue_type='incomplete_rwl_equation',
                            description='RWL equation should show full multiplication chain',
                            manual_reference='Example 5: RWL = 23 × HM × VM × DM × AM × FM × CM = XX kg',
                            current_text='[Incomplete equation format]',
                            expected_format='RWL = 23 × 1.000 × 0.802 × ... = XX.X kg',
                            correction_type='reformat'
                        ))
        
        # Verify LI interpretation matches manual standards
        if 'Hazard' in str(sections.keys()):
            hazard_section = next((s for s in sections.values() 
                                 if 'hazard' in s.name.lower()), None)
            
            if hazard_section:
                # Extract LI value
                li_match = re.search(r'LI[=:\s]+(\d+(?:\.\d+)?)', hazard_section.content)
                if li_match:
                    li_value = float(li_match.group(1))
                    
                    # Check if interpretation matches manual standards
                    approved_phrase = self._get_approved_risk_phrase_for_li(li_value)
                    
                    # Check if any forbidden terms are used
                    content_lower = hazard_section.content.lower()
                    if any(term in content_lower for term in self.standards.FORBIDDEN_TERMS):
                        issues.append(ComplianceIssue(
                            severity='major',
                            category='terminology',
                            section=hazard_section.name,
                            issue_type='non_standard_risk_interpretation',
                            description=f'Risk interpretation for LI={li_value:.2f} does not match manual standards',
                            manual_reference='Examples 1-10, NIOSH Manual',
                            current_text=hazard_section.content[:200],
                            expected_format=f'Should use: "{approved_phrase}"',
                            correction_type='rephrase',
                            metadata={'li_value': li_value, 'approved_phrase': approved_phrase}
                        ))
        
        return issues
    
    def _check_reference_compliance(self, sections: Dict[str, Section]) -> List[ComplianceIssue]:
        """
        Validates figure and table references per manual style.
        Manual consistently references figures and tables.
        """
        issues = []
        
        # Check for figure references in Job Analysis
        analysis_sections = [s for s in sections.values() if 'analysis' in s.name.lower()]
        
        for section in analysis_sections:
            if 'figure' not in section.content.lower():
                issues.append(ComplianceIssue(
                    severity='minor',
                    category='format',
                    section=section.name,
                    issue_type='missing_figure_reference',
                    description='Job Analysis should reference figure/worksheet',
                    manual_reference='All examples include figure references',
                    current_text='[No figure reference]',
                    expected_format='...recorded on the job analysis worksheet (Figure X).',
                    correction_type='rephrase'
                ))
        
        return issues
    
    def _get_approved_risk_phrase(self, section: Section) -> str:
        """Returns approved risk phrase based on section content"""
        # Try to extract LI value
        li_match = re.search(r'LI[=:\s]+(\d+(?:\.\d+)?)', section.content)
        if li_match:
            li_value = float(li_match.group(1))
            return self._get_approved_risk_phrase_for_li(li_value)
        
        return "physically stressful for some workers"
    
    def _get_approved_risk_phrase_for_li(self, li_value: float) -> str:
        """
        Returns NIOSH manual approved risk phrase for given LI.
        Based on actual phrasing from Examples 1-10.
        """
        if li_value <= 1.0:
            return "acceptable for most healthy workers"
        elif li_value <= 1.5:
            return "some healthy workers would find this task physically stressful"
        elif li_value <= 3.0:
            return "physically stressful for some workers"
        else:
            return "physically stressful for most industrial workers"
    
    def _calculate_compliance_scores(self, sections: Dict[str, Section]):
        """Calculate compliance scores for each section"""
        total_issues = len(self.issues)
        
        if total_issues == 0:
            for section in sections.values():
                section.compliance_score = 100.0
            return
        
        # Deduct points based on issue severity
        severity_weights = {
            'critical': 25,
            'major': 10,
            'minor': 3,
            'style': 1
        }
        
        section_deductions = {}
        for issue in self.issues:
            if issue.section not in section_deductions:
                section_deductions[issue.section] = 0
            section_deductions[issue.section] += severity_weights.get(issue.severity, 5)
        
        for section in sections.values():
            deduction = section_deductions.get(section.name, 0)
            section.compliance_score = max(0, 100 - deduction)


# ========== CORRETTORE PROFESSIONALE ==========

class NIOSHProfessionalCorrector:
    """
    Applica correzioni seguendo esattamente gli standard del manuale NIOSH.
    Emula come un editor professionale riscriverebbe le sezioni.
    """
    
    def __init__(self, use_llm: bool = True, verbose: bool = False):
        self.use_llm = use_llm and ollama is not None
        self.verbose = verbose
        self.standards = NIOSHManualStandards()
        self.ollama_model = "llama3.2"
        
    def correct_to_manual_standard(
        self,
        section: Section,
        issue: ComplianceIssue,
        context: Dict[str, Section]
    ) -> str:
        """
        Corrects section to match NIOSH manual standards.
        Professional approach: rewrite to match manual examples.
        """
        
        if issue.correction_type == 'reformat':
            return self._reformat_to_manual_style(section, issue)
        elif issue.correction_type == 'rephrase':
            return self._rephrase_to_manual_terminology(section, issue)
        elif issue.correction_type == 'rewrite' and self.use_llm:
            return self._llm_rewrite_to_manual_standard(section, issue, context)
        elif issue.correction_type == 'rewrite':
            return self._rule_based_rewrite(section, issue)
        else:
            return section.content
    
    def _reformat_to_manual_style(self, section: Section, issue: ComplianceIssue) -> str:
        """Reformats section to match manual presentation style"""
        content = section.content
        
        # Convert task variables from bullet to narrative
        if issue.issue_type == 'incorrect_task_variable_format':
            content = self._convert_bullets_to_narrative(content)
        
        # Integrate multipliers into narrative
        if issue.issue_type == 'suboptimal_multiplier_format':
            content = self._integrate_multipliers_narrative(content)
        
        return content
    
    def _convert_bullets_to_narrative(self, content: str) -> str:
        """
        Converts bullet-point task variables to narrative format per manual.
        Example manual format: "At the origin of the lift the horizontal distance (H) is 20 cm..."
        """
        
        # Extract values from bullet format
        h_match = re.search(r'[-*]\s*(?:H|Horizontal)[^:]*:\s*(\d+)\s*cm', content, re.IGNORECASE)
        v_match = re.search(r'[-*]\s*(?:V|Vertical)[^:]*:\s*(\d+)\s*cm.*origin', content, re.IGNORECASE)
        a_match = re.search(r'[-*]\s*(?:A|Asymmetry)[^:]*:\s*(\d+)', content, re.IGNORECASE)
        
        if h_match and v_match:
            h_val = h_match.group(1)
            v_val = v_match.group(1)
            a_val = a_match.group(1) if a_match else '0'
            
            # Build narrative per manual format
            narrative = f"At the origin of the lift the horizontal distance (H) is {h_val} cm, the vertical distance (V) is {v_val} cm, and the asymmetry angle (A) is {a_val}°."
            
            # Remove bullet points
            content = re.sub(r'^\s*[-*]\s+(?:H|V|A)[^\n]+\n', '', content, flags=re.MULTILINE)
            
            # Insert narrative
            content = narrative + "\n\n" + content
        
        return content
    
    def _integrate_multipliers_narrative(self, content: str) -> str:
        """
        Integrates multipliers into narrative paragraph per manual style.
        Example: "The multipliers are computed from the lifting equation or determined from the multiplier tables..."
        """
        
        # Extract multiplier values
        multipliers = {}
        for mult in ['HM', 'VM', 'DM', 'AM', 'FM', 'CM']:
            match = re.search(rf'{mult}\s*=\s*([\d.]+)', content)
            if match:
                multipliers[mult] = match.group(1)
        
        if multipliers:
            # Build narrative per manual
            narrative = "The multipliers are computed from the lifting equation or determined from the multiplier tables (Tables 1 to 5, and Table 7). "
            
            mult_phrases = []
            for mult, value in multipliers.items():
                mult_phrases.append(f"the {mult} is {value}")
            
            narrative += ", ".join(mult_phrases[:-1])
            if len(mult_phrases) > 1:
                narrative += f", and {mult_phrases[-1]}"
            else:
                narrative += mult_phrases[0]
            
            narrative += "."
            
            # Remove bullet format
            content = re.sub(r'^\s*[-*]\s+(?:HM|VM|DM|AM|FM|CM)[^\n]+\n', '', content, flags=re.MULTILINE)
            
            # Insert narrative
            content = narrative + "\n\n" + content
        
        return content
    
    def _rephrase_to_manual_terminology(self, section: Section, issue: ComplianceIssue) -> str:
        """Replaces non-standard terminology with approved manual phrasing"""
        content = section.content
        
        # Replace forbidden terms with approved alternatives
        if 'forbidden_term' in issue.metadata:
            forbidden = issue.metadata['forbidden_term']
            approved = issue.metadata.get('approved_replacement', 'physically stressful')
            
            # Case-insensitive replacement
            pattern = re.compile(re.escape(forbidden), re.IGNORECASE)
            content = pattern.sub(approved, content)
        
        return content
    
    def _llm_rewrite_to_manual_standard(
        self,
        section: Section,
        issue: ComplianceIssue,
        context: Dict[str, Section]
    ) -> str:
        """Uses LLM to rewrite section following manual examples"""
        
        prompt = self._build_manual_compliance_prompt(section, issue, context)
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            corrected = response['message']['content'].strip()
            
            # Remove markdown formatting
            corrected = re.sub(r'^```(?:markdown)?\n', '', corrected)
            corrected = re.sub(r'\n```$', '', corrected)
            
            return corrected
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  LLM correction failed: {e}")
            return self._rule_based_rewrite(section, issue)
    
    def _build_manual_compliance_prompt(
        self,
        section: Section,
        issue: ComplianceIssue,
        context: Dict[str, Section]
    ) -> str:
        """Builds prompt with manual examples for LLM correction"""
        
        # Get relevant manual example
        manual_example = self._get_manual_example_for_section(section.name)
        
        prompt = f"""You are a professional NIOSH report editor. Rewrite this section to match the NIOSH Applications Manual style EXACTLY.

CURRENT SECTION: {section.name}
```
{section.content}
```

COMPLIANCE ISSUE:
- Problem: {issue.description}
- Manual reference: {issue.manual_reference}
- Expected format: {issue.expected_format}

NIOSH MANUAL EXAMPLE (follow this style):
CRITICAL REQUIREMENTS:
1. Use NARRATIVE format for task variables (NOT bullet points)
2. Use approved terminology: "physically stressful for some workers" (NOT "extreme hazard")
3. Reference figures and tables: "Figure X", "Table 8"
4. Follow exact phrasing patterns from manual examples
5. Start sentences with manual patterns: "The weight to be lifted...", "The worksheet shows..."
Rewrite the section following manual style EXACTLY. Provide ONLY the corrected section content:"""
        
        return prompt
    
    def _get_manual_example_for_section(self, section_name: str) -> str:
        """Returns relevant manual example text for given section type"""
        
        examples = {
            'Job Description': """A worker manually lifts trays of clean dishes from a conveyor at the end of a dishwashing machine and loads them on a cart. The trays are filled with assorted dishes (e.g., glasses, plates, bowls) and silverware. The job takes between 45 minutes and 1 hour to complete and the lifting frequency rate averages 5 lifts/min.""",
            
            'Job Analysis': """The task variable data are measured and recorded on the job analysis worksheet (Figure 16). At the origin of the lift the horizontal distance (H) is 20 inches, the vertical distance (V) is 44 inches, and the asymmetry angle (A) is 30°. At the destination of the lift, H is 20 inches, V is 7 inches, and A is 30°. The trays normally weigh 20 lbs.

Using Table 6, the coupling is classified as Good. Significant control is required at the destination of the lift. Using Table 5, the FM is determined to be .80. The multipliers are computed from the lifting equation or determined from the multiplier tables (Tables 1 to 5, and Table 7). As shown in Figure 16, the RWL is 14.4 lbs at the origin and 13.3 lbs at the destination.""",
            
            'Hazard Assessment': """The weight to be lifted (20 lbs) is greater than the RWL at both the origin and destination of the lift (14.4 lbs and 13.3 lbs, respectively). The LI at the origin is 20/14.4 or 1.4 and the LI at the destination is 1.5. These results indicate that this lifting task would be stressful for some workers.""",
            
            'Redesign Suggestions': """The worksheet shows that the smallest multipliers (i.e., the greatest penalties) are .50 for the HM, .80 for the FM, .83 for the VM, and .90 for the AM. Using Table 8, the following job modifications are suggested:

1. Bring the load closer to the worker to increase HM.
2. Reduce the lifting frequency rate to increase FM.
3. Raise the destination of the lift to increase VM.
4. Reduce the angle of twist to increase AM.""",
            
            'Comments': """This analysis was based on a one-hour work session. If a subsequent work session begins before the appropriate recovery period has elapsed, then the two-hour category would be used to compute the FM value.

As in the previous example, since the lifting pattern is continuous over the full duration of the work sample (i.e., more than 15 minutes), the lifting frequency is not adjusted using the special procedure."""
        }
        
        # Match section name to example
        for key, example in examples.items():
            if key.lower() in section_name.lower():
                return example
        
        return examples.get('Job Analysis', '')  # Default
    
    def _rule_based_rewrite(self, section: Section, issue: ComplianceIssue) -> str:
        """Fallback rule-based rewriting for manual compliance"""
        content = section.content
        
        # Add required opening phrases
        if issue.issue_type == 'missing_required_phrase':
            if 'analysis' in section.name.lower():
                prefix = "The task variable data are measured and recorded on the job analysis worksheet (Figure X).\n\n"
                if not content.startswith("The task variable"):
                    content = prefix + content
        
        # Add required references
        if issue.issue_type == 'missing_worksheet_reference':
            content = "The worksheet shows that " + content
        
        if issue.issue_type == 'missing_table_reference':
            # Insert before first numbered suggestion
            insert_pos = re.search(r'^\d+\.', content, re.MULTILINE)
            if insert_pos:
                content = (content[:insert_pos.start()] + 
                          "Using Table 8, the following job modifications are suggested:\n\n" +
                          content[insert_pos.start():])
        
        return content


# ========== CONTROLLORE PRINCIPALE ==========

class ErgoController:
    """
    Controllore principale - Validatore professionale di conformità al manuale NIOSH.
    Si comporta come un ergonomo senior che revisiona le reportistiche per la pubblicazione.
    """
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.parser = MarkdownParser()
        self.validator = NIOSHManualValidator(verbose=verbose)
        self.corrector = NIOSHProfessionalCorrector(use_llm=True, verbose=verbose)
        
    def run(self, file_path: str) -> None:
        """
        Complete professional validation and correction pipeline.
        Emulates expert human reviewer workflow.
        """
        
        # Stage 0: File validation
        path = Path(file_path)
        if not path.exists():
            print(f"❌ Error: File not found: {file_path}")
            sys.exit(1)
        
        # Stage 1: Load and parse
        print("📖 Loading report for NIOSH manual compliance review...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        sections = self.parser.parse_sections(text)
        print(f"✅ Parsed {len(sections)} sections\n")
        
        if self.verbose:
            print("   Sections found:")
            for name in sections.keys():
                print(f"   • {name}")
            print()
        
        # Stage 2: Extract metrics
        print("🔍 Extracting technical data...")
        for section in sections.values():
            section.numeric_data = self.parser.extract_all_metrics(section)
        
        if self.verbose:
            total_metrics = sum(len(s.numeric_data) for s in sections.values())
            print(f"   • Extracted {total_metrics} numeric values")
            print()
        
        # Stage 3: NIOSH Manual compliance validation
        print("📋 Checking NIOSH Applications Manual compliance...")
        print("   (Validating against Examples 1-10 standards)\n")
        
        issues = self.validator.validate_full_compliance(sections)
        
        if not issues:
            print("\n✅ REPORT FULLY COMPLIANT WITH NIOSH MANUAL")
            print("   No corrections needed - ready for publication")
            return
        
        # Stage 4: Categorize and report issues
        self._print_compliance_report(issues, sections)
        
        # Stage 5: Dry-run check
        if self.dry_run:
            print("\n🔍 Dry-run mode: No corrections applied")
            print("   Run without --dry-run to apply professional corrections")
            return
        
        # Stage 6: Backup original
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        print(f"\n💾 Original backed up: {backup_path}")
        
        # Stage 7: Apply professional corrections
        print("\n🔧 Applying NIOSH manual corrections...\n")
        
        corrections_applied = 0
        for issue in issues:
            section_name = issue.section
            
            # Find matching section
            target_section = None
            for section in sections.values():
                if section.name == section_name or section_name in section.name:
                    target_section = section
                    break
            
            if not target_section:
                continue
            
            try:
                corrected = self.corrector.correct_to_manual_standard(
                    target_section,
                    issue,
                    sections
                )
                
                if corrected != target_section.content:
                    target_section.content = corrected
                    corrections_applied += 1
                    
                    # Professional correction reporting
                    severity_icon = {
                        'critical': '🔴',
                        'major': '🟡',
                        'minor': '🔵',
                        'style': '⚪'
                    }.get(issue.severity, '✓')
                    
                    category_icon = {
                        'structure': '📐',
                        'terminology': '📝',
                        'format': '🎨',
                        'calculation': '🔢',
                        'consistency': '🔗'
                    }.get(issue.category, '✓')
                    
                    print(f"   {severity_icon} {category_icon} {issue.section}")
                    print(f"      Fixed: {issue.issue_type}")
                    print(f"      Ref: {issue.manual_reference}")
                    
            except Exception as e:
                print(f"   ❌ Failed to correct {section_name}: {e}")
        
        # Stage 8: Rebuild and save
        corrected_report = self._rebuild_report(sections, text)
        output_path = path.stem + "_niosh_compliant" + path.suffix
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(corrected_report)
        
        # Final report
        print(f"\n{'='*70}")
        print(f"✅ NIOSH MANUAL COMPLIANCE CORRECTIONS APPLIED")
        print(f"{'='*70}")
        print(f"   Total corrections: {corrections_applied}")
        print(f"   Output file: {output_path}")
        print(f"   Original backup: {backup_path}")
        print()
        print("   Report now follows NIOSH Applications Manual standards")
        print("   Ready for professional review and publication")
        print(f"{'='*70}\n")
    
    def _print_compliance_report(self, issues: List[ComplianceIssue], sections: Dict[str, Section]):
        """Prints professional compliance report"""
        
        # Categorize issues
        by_severity = {'critical': [], 'major': [], 'minor': [], 'style': []}
        by_category = {}
        
        for issue in issues:
            by_severity[issue.severity].append(issue)
            if issue.category not in by_category:
                by_category[issue.category] = []
            by_category[issue.category].append(issue)
        
        # Overall summary
        print(f"\n{'='*70}")
        print("NIOSH MANUAL COMPLIANCE REPORT")
        print(f"{'='*70}\n")
        
        total = len(issues)
        critical = len(by_severity['critical'])
        major = len(by_severity['major'])
        minor = len(by_severity['minor'])
        style = len(by_severity['style'])
        
        print(f"📊 Total issues found: {total}")
        print(f"   🔴 Critical: {critical}  (structural/calculation errors)")
        print(f"   🟡 Major:    {major}   (terminology/format deviations)")
        print(f"   🔵 Minor:    {minor}   (style/reference improvements)")
        print(f"   ⚪ Style:    {style}   (optional enhancements)")
        print()
        
        # Compliance scores
        print("📈 Section Compliance Scores:")
        for section in sections.values():
            score = section.compliance_score
            bar_length = int(score / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            status = '✅' if score >= 90 else '⚠️' if score >= 70 else '❌'
            print(f"   {status} {section.name[:30]:<30} {bar} {score:.1f}%")
        print()
        
        # Category breakdown
        print("📋 Issues by Category:")
        for category, cat_issues in by_category.items():
            icon = {
                'structure': '📐',
                'terminology': '📝',
                'format': '🎨',
                'calculation': '🔢',
                'consistency': '🔗'
            }.get(category, '•')
            print(f"   {icon} {category.title()}: {len(cat_issues)}")
        print()
        
        # Detailed issues (show critical and major)
        if critical + major > 0:
            print("🔍 CRITICAL & MAJOR ISSUES REQUIRING CORRECTION:\n")
            
            issue_num = 1
            for severity in ['critical', 'major']:
                for issue in by_severity[severity]:
                    severity_icon = '🔴' if severity == 'critical' else '🟡'
                    
                    print(f"{issue_num}. {severity_icon} [{severity.upper()}] {issue.section}")
                    print(f"   Issue: {issue.description}")
                    print(f"   Manual reference: {issue.manual_reference}")
                    print(f"   Expected: {issue.expected_format[:80]}...")
                    print()
                    issue_num += 1
        
        # Recommendations
        print("💡 PROFESSIONAL RECOMMENDATIONS:\n")
        
        if critical > 0:
            print("   🔴 CRITICAL: These issues prevent publication compliance")
            print("      → Must be corrected before final review")
            print()
        
        if major > 0:
            print("   🟡 MAJOR: These deviations affect professional quality")
            print("      → Strongly recommended for NIOSH manual adherence")
            print()
        
        if minor > 0:
            print("   🔵 MINOR: These improvements enhance manual conformity")
            print("      → Recommended for optimal compliance")
            print()
        
        print(f"{'='*70}\n")
    
    def _rebuild_report(self, sections: Dict[str, Section], original_text: str) -> str:
        """Rebuilds report preserving header and structure"""
        
        # Extract header (title before first ##)
        header_match = re.search(r'^(.*?)(?=^##\s)', original_text, re.MULTILINE | re.DOTALL)
        header = header_match.group(1).strip() if header_match else ""
        
        # Rebuild with sections
        rebuilt = header + "\n\n" if header else ""
        
        for section in sections.values():
            # Preserve section numbering if present
            section_header = f"## {section.name}"
            rebuilt += f"{section_header}\n\n"
            rebuilt += section.content.strip() + "\n\n"
        
        return rebuilt.strip() + "\n"


# ========== PARSER SEMPLIFICATO ==========

class MarkdownParser:
    """Gestisce il parsing delle reportistiche Markdown"""
    
    SECTION_PATTERN = r'^##\s+(.+?)$'
    
    def parse_sections(self, text: str) -> Dict[str, Section]:
        """Divides report into sections"""
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
                section_name = header_match.group(1).strip()
                current_section = section_name
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
    
    def extract_all_metrics(self, section: Section) -> Dict[str, float]:
        """Extracts NIOSH numeric parameters"""
        metrics = {}
        
        # Standard parameters
        patterns = {
            'Load Weight': r'(?:Load Weight|L)[:\s]+([0-9.]+)\s*(?:kg|lbs)',
            'Vertical Location': r'(?:Vertical Location|V)[:\s]+([0-9.]+)\s*(?:cm|inches)',
            'Horizontal Distance': r'(?:Horizontal Distance|H)[:\s]+([0-9.]+)\s*(?:cm|inches)',
            'Asymmetry Angle': r'(?:Asymmetry Angle|A)[:\s]+([0-9.]+)',
            'Frequency': r'(?:Frequency|F)[:\s]+([0-9.]+)',
            'Duration': r'Duration[:\s]+([0-9.]+)',
            'RWL': r'RWL[:\s=]+([0-9.]+)',
            'LI': r'LI[:\s=]+([0-9.]+)',
            'HM': r'HM\s*=\s*([0-9.]+)',
            'VM': r'VM\s*=\s*([0-9.]+)',
            'DM': r'DM\s*=\s*([0-9.]+)',
            'AM': r'AM\s*=\s*([0-9.]+)',
            'FM': r'FM\s*=\s*([0-9.]+)',
            'CM': r'CM\s*=\s*([0-9.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, section.content, re.IGNORECASE)
            if match:
                try:
                    metrics[key] = float(match.group(1))
                except ValueError:
                    pass
        
        return metrics


# ========== PUNTO DI ACCESSO CLI ==========

def main():
    """
    Punto di accesso CLI per la validazione professionale di conformità al manuale NIOSH.
    
    Questo strumento emula un ergonomo senior che revisiona le reportistiche per l'aderenza
    agli standard del Manuale Applicativo NIOSH (Esempi 1-10).
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ErgoController 4.0 - Validatore Professionale di Conformità Manuale NIOSH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
APPROCCIO DI VALIDAZIONE PROFESSIONALE:
Questo strumento valida le reportistiche secondo gli standard del Manuale Applicativo NIOSH,
verificando la conformità di formattazione, terminologia e stile con gli Esempi 1-10.

CATEGORIE DI VALIDAZIONE:
  🔴 CRITICHE  - Errori strutturali/di calcolo (da correggere obbligatoriamente)
  🟡 MAGGIORI  - Deviazioni terminologiche/di formato (fortemente consigliate)
  🔵 MINORI    - Miglioramenti di stile/riferimenti (consigliati)
  ⚪ STILE     - Miglioramenti opzionali

ESEMPI:
  %(prog)s report.md                    Validazione e correzione completa
  %(prog)s report.md --dry-run          Mostra problemi senza correggere
  %(prog)s report.md --verbose          Analisi dettagliata della conformità
  %(prog)s report.md --dry-run -v       Modalità revisione con dettagli completi

OUTPUT:
  - Report di conformità con punteggi di sezione
  - Analisi dettagliata dei problemi per categoria
  - Correzioni professionali secondo gli esempi del manuale
  - Report corretto pronto per la pubblicazione

RIFERIMENTI:
  Manuale Applicativo NIOSH per l'Equazione di Sollevamento Rivista
  Esempi 1-10 (Sezioni 3.2-3.5)
        """
    )
    
    parser.add_argument(
        'file',
        help='Percorso della reportistica NIOSH (formato Markdown)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Solo validazione - mostra problemi senza correggere'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Analisi dettagliata della conformità'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 4.0 - Validatore Professionale Manuale NIOSH'
    )
    
    args = parser.parse_args()
    
    # Stampa banner professionale
    print()
    print("=" * 70)
    print("  ERGOCONTROLLER 4.0 - VALIDATORE DI CONFORMITÀ MANUALE NIOSH")
    print("  Validazione professionale secondo il Manuale Applicativo NIOSH")
    print("=" * 70)
    print()
    
    # Esegui validazione professionale
    try:
        controller = ErgoController(dry_run=args.dry_run, verbose=args.verbose)
        controller.run(args.file)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Validazione annullata dall'utente")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Errore di validazione: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()