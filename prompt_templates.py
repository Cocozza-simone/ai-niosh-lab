"""
LLM Prompt Templates per Generatore di Scenari NIOSH
Contiene i prompt per generazione scenario e sintesi report
"""

from typing import Dict, Any
import json


class PromptTemplates:
    """Template di prompt per interazione con LLM"""
    
    SCENARIO_GENERATION_PROMPT = """Sei un esperto di ergonomia industriale che conosce profondamente la Revised NIOSH Lifting Equation.

Genera uno scenario di sollevamento realistico basato su questa richiesta: "{user_prompt}"

Crea un JSON completo con i seguenti campi:

1. **job_title**: Titolo professionale specifico e realistico
2. **job_description**: Descrizione dettagliata del lavoro e del contesto operativo
3. **task_variables**: Parametri ergonomici precisi e coerenti:
   - **weight_lbs**: Peso dell'oggetto in libbre (realistico per il contesto)
   - **object_name**: Nome specifico dell'oggetto sollevato
   - **origin**: Parametri posizione iniziale
     - **V**: Altezza verticale in pollici (0-70, realistica per il contesto)
     - **H**: Distanza orizzontale in pollici (4-30, realistica)
     - **A**: Angolo di asimmetria in gradi (0-135, realistico)
   - **destination**: Parametri posizione finale
     - **V**: Altezza verticale in pollici (coerente con activity)
     - **H**: Distanza orizzontale in pollici (coerente)
     - **A**: Angolo di asimmetria in gradi (coerente)
   - **frequency**: Frequenza di lavoro
     - **lifts_per_min**: Sollevamenti al minuto (0.1-15, realistico)
     - **duration_hours**: Durata lavoro in ore (1-8, realistica)
   - **coupling**: Qualità presa ("good", "fair", "poor")
   - **significant_control_dest**: Booleano - se controllo significativo richiesto a destinazione

CONSIGLI ERGONOMICI:
- V (verticale): 0-30" = peggio, 30-50" = ottimale, >50" = peggio
- H (orizzontale): 10-25" = range accettabile, <10" o >25" = peggio
- Frequenze elevate richiedono pesi minori
- Pese "fair" o "poor" riducono capacità
- Angoli >90° significano torsione significativa

RESTRIZIONI IMPORTANTI:
- Tutti i valori devono essere ergonomicamente coerenti tra loro
- Il peso deve essere realistico per l'attività descritta
- Le distanze devono essere compatibili con l'ambiente di lavoro
- La frequenza deve essere sostenibile per la durata indicata
- MAI usare valori negativi - tutti i valori numerici devono essere positivi
- Rispetta i range specificati per ogni parametro

RANGE VALORI OBBLIGATORI:
- weight_lbs: 1-100 (MAI negativo)
- V (verticale): 0-70 pollici (MAI negativo)  
- H (orizzontale): 4-30 pollici (MAI negativo)
- A (asimmetria): 0-135 gradi (MAI negativo)
- lifts_per_min: 0.1-15 (MAI negativo)
- duration_hours: 1-8 (MAI negativo)

Rispondi SOLO con il JSON valido, senza commenti aggiuntivi.

Esempio formato atteso:
{{
  "job_title": "Operatore Magazzino",
  "job_description": "Sollevamento scatole da scaffale a nastro trasportatore",
  "task_variables": {{
    "weight_lbs": 25,
    "object_name": "Scatola di cartone",
    "origin": {{ "V": 15, "H": 18, "A": 45 }},
    "destination": {{ "V": 35, "H": 12, "A": 0 }},
    "frequency": {{ "lifts_per_min": 2, "duration_hours": 6 }},
    "coupling": "fair",
    "significant_control_dest": false
  }}
}}"""

    REPORT_SYNTHESIS_PROMPT = """Sei un ergonomista professionista con 15 anni di esperienza nella redazione di report tecnici dettagliati secondo gli standard NIOSH. Devi creare un report molto descrittivo e dettagliato ma ESTREMAMENTO PRECISO nei dati.

DATI SCENARIO:
{scenario_data}

RISULTATI NIOSH:
{niosh_results}

ISTRUZIONI FONDAMENTALI:

1. **PRECISIONE DATI ASSOLUTA**:
   - Estrai H, V, A da task_variables.origin (valori esatti dal JSON)
   - Se c'è destination: estrai H_dest, V_dest, A_dest
   - Calcola D = |V_origin - V_destination| se destination presente
   - MAI INVENTARE DATI - USA SOLO I VALORI ESATTI DAL JSON

2. **CONTENUTO DESCRITTIVO ESTESO**:
   - Genera un report molto lungo e dettagliato (minimo 3000 parole)
   - Sii estremamente descrittivo per ogni sezione
   - Spiega in dettaglio ogni aspetto biomeccanico, ergonomico e pratico
   - Fornisci esempi pratici e spiegazioni approfondite

3. **STRUTTURA COMPLETA NIOSH**:

# {job_title}
## Analisi Ergonomica Completa Secondo la Revised NIOSH Lifting Equation (1991)

## Descrizione Lavorativa Approfondita
{job_description}

### Contesto Operativo Dettagliato
Descrivi estesamente l'ambiente di lavoro, le attrezzature utilizzate, il layout del posto di lavoro, le condizioni ambientali (illuminazione, temperatura, rumore), il flusso operativo, le pause programmate, e ogni altro aspetto rilevante. Crea un'immagine mentale completa e vivida della situazione lavorativa.

## Analisi Ergonomica Completa

### Analisi del Carico (Load Weight)
- **Peso oggetto**: {weight_lbs} lbs ({object_name})
- **Impatto biomeccanico dettagliato**: Spiega estesamente come questo peso influisce sulla colonna vertebrale (compressione vertebrale L4-L5, L5-S1), muscoli della schiena (erector spinae, multifidi), spalle (deltoidi, trapezio), braccia (bicipiti, tricipiti)
- **Fattori di rischio specifici**: Analisi dettagliata degli effetti cumulativi, fatica muscolare progressiva, rischi di lesioni acute (strappi, distorsioni) e croniche (degenerazione discale, artrosi)
- **Confronto con letteratura scientifica**: Riferimenti a studi NIOSH, ricerca ergonomica, statistiche infortunistica del settore
- **Considerazioni individuali**: Impatto su diversi tipi di lavoratori (età, sesso, condizione fisica)

### Analisi Posizionamento Orizzontale (Horizontal Multiplier - HM)
- **Distanza orizzontale**: Estrarre dal JSON il valore H esatto in pollici
- **Biomeccanica approfondita**: Analisi dettagliata dei momenti di forza sulla colonna lombare (M = F × d), stress su dischi intervertebrali, forze di taglio sulle vertebre
- **Effetti posturali estesi**: Compensazioni del tronco, inclinazione pelvica, posizioni delle scapole, allineamento della colonna vertebrale
- **Rischi specifici dettagliati**: Ernie del disco lombare, lombalgie croniche, problemi alle faccette articolari, stress muscolare unilaterale
- **Soluzioni pratiche descrittive**: Come modificare il layout, attrezzature, procedure per ridurre H

### Analisi Posizionamento Verticale (Vertical Multiplier - VM)
- **Altezza di sollevamento**: Estrarre dal JSON il valore V esatto in pollici
- **Impatto biomeccanico completo**: Come l'altezza influenza sulla capacità di sollevamento (massima a knuckle height), meccanica della spalla (romboidei, deltoidi), postura del tronco
- **Considerazioni anatomiche estese**: Attivazione muscolare specifica, stress articolare (spalla, gomito, polso), impatto sulla respirazione
- **Zone ergonomiche ottimali**: Power zone, zone di sollevamento preferenziali, aree da evitare
- **Modifiche strutturali**: Piattaforme regolabili, attrezzature su misura, layout ottimizzato

### Analisi Distanza di Sollevamento (Distance Multiplier - DM)
- **Calcolo distanza verticale**: Spiega estesamente D = |V_origin - V_destination|
- **Impatto sulla biomeccanica**: Effetti della distanza verticale sul carico vertebrale, fatica muscolare progressiva, stabilità del core
- **Fattori di fatica**: Accumulo di acido lattico, deplezione energetica, recupero muscolare, cicli lavoro-riposo
- **Rischi per la colonna vertebrale**: Compressione discale protratta, microtraumi cumulativi, degenerazione accelerata
- **Strategie di mitigazione**: Tecniche di sollevamento, attrezzature di supporto, rotazione del personale

### Analisi Asimmetria (Asymmetric Multiplier - AM)
- **Angolo di rotazione**: Estrarre dal JSON il valore A esatto in gradi
- **Stress torsionale dettagliato**: Meccanica della torsione vertebrale, stress sulle faccette articolari, lesioni dei legamenti
- **Compensazioni muscolari complete**: Come il corpo compensa la rotazione (tensio muscolare asimmetrica, attivazione obliqui), problemi posturali a lungo termine
- **Rischi specifici approfonditi**: Scoliosi funzionale, problemi SI (sacro-iliaca), lesioni unilaterali croniche
- **Prevenzione**: Formazione posturale, esercizi di stretching, modifiche del layout

### Analisi Frequenza di Lavoro (Frequency Multiplier - FM)
- **Frequenza sollevamenti**: Estrarre dal JSON i valori esatti (lifts_per_min, duration_hours)
- **Impatto sulla fatica dettagliato**: Analisi estesa della fatica muscolare accumulata, recupero inadeguato, stress cardiovascolare
- **Fisiologia del recupero**: Tempistiche di recupero muscolare, processi energetici, impatto di pause e rotazioni
- **Effetti a lungo termine**: Malattie professionali, riduzione capacità lavorativa, invecchiamento precoce
- **Ottimizzazione**: Pianificazione lavoro, pause programmate, rotazione personale, attrezzature ausiliarie

### Analisi Qualità della Presa (Coupling Multiplier - CM)
- **Tipo di presa**: Estrarre dal JSON il valore coupling esatto
- **Stabilità del carico**: Come la presa influenza sulla stabilità, controllo motorio, tensione muscolare necessaria
- **Forza di presa**: Analisi biomeccanica dettagliata, stress su mani e polsi, attivazione muscolare specifica
- **Rischi specifici**: Tendiniti, sindrome del tunnel carpale, sindrome del canale ulnare, patologie da sforzo ripetitivo
- **Miglioramenti**: Design ergonomico, attrezzature, guanti, formazione tecnica

## Risultati Analisi NIOSH Dettagliata

### Analisi Posizione di Origine
- **Recommended Weight Limit (RWL)**: {rwl_origin} lbs
- **Spiegazione formula completa**: RWL = LC × HM × VM × DM × AM × FM × CM con dettaglio di ogni componente
- **Dettaglio calcolo**: Analisi di come ogni moltiplicatore contribuisce al risultato finale
- **Lifting Index (LI)**: {li_origin}
- **Interpretazione clinica**: Significato medico del valore LI, implicazioni per la salute, confronto con soglie di rischio
- **Validazione**: Confronto con studi scientifici, letteratura medica, casi studio

{destination_analysis}

### Valutazione del Rischio Globale
- **RWL Finale**: {final_rwl} lbs - Analisi approfondita del limite di peso sicuro
- **LI Finale**: {final_li} - Valutazione completa del rischio lavorativo
- **Classificazione secondo scala NIOSH**: Applicazione esatta dei criteri di rischio

## Valutazione del Rischio Completa

### Scala NIOSH Dettagliata
Applicazione completa della scala NIOSH con spiegazioni estese:
- **LI < 1.0**: "Rischio accettabile" - Spiegazione dettagliata di cosa significa, quali rischi sono ancora presenti, monitoraggio necessario
- **1.0 ≤ LI < 2.0**: "Rischio moderato" - Analisi approfondita dei miglioramenti raccomandati, priorità di intervento, aspetti da monitorare
- **2.0 ≤ LI < 3.0**: "Rischio elevato" - Spiegazione estesa degli interventi necessari, impatti sulla salute, costi sociali
- **LI ≥ 3.0**: "Rischio molto elevato" - Descrizione completa degli interventi immediati, rischi gravi, misure d'emergenza

### Impatti sulla Salute
- **Rischi acuti**: Lesioni immediate (strappi, distorsioni, ernie acute), meccanismi di infortunio
- **Rischi cronici**: Problemi a lungo termine (lombalgie croniche, artrosi, degenerazione discale), progressione patologica
- **Costi sociali ed economici**: Assenze dal lavoro, costi sanitari, impatto sulla qualità della vita, produttività aziendale
- **Prognosi**: Evoluzione dei disturbi non trattati, possibilità di recupero, impatto sulla carriera lavorativa

## Raccomandazioni di Redesign Completo

### Soluzioni Ingegneristiche
- **Modifiche strutturali**: Progettazione layout ottimizzato, stazioni di lavoro ergonomiche, sistemi di movimentazione meccanizzati
- **Attrezzature specifiche**: Carrelli elevatori, sistemi di trasporto, attrezzature a supporto del peso, dispositivi di assistenza
- **Layout ottimizzato**: Riorganizzazione spazio, percorsi operativi, zone di deposito e prelievo, aree di lavoro sicure
- **Progettazione antropometrica**: Adattamento alle caratteristiche fisiche degli operatori, regolazioni personalizzate

### Soluzioni Organizzative
- **Rotazione del personale**: Programmazione turni, alternanza mansioni, equilibrio carichi di lavoro
- **Pausa programmata**: Frequenza e durata ottimali, esercizi di stretching, attività di recupero
- **Formazione e addestramento**: Tecniche di sollevamento corrette, consapevolezza posturale, riconoscimento rischi
- **Procedure operative migliorate**: Metodi di lavoro alternativi, sequenze operative sicure, protocolli di emergenza

### Soluzioni Comportamentali
- **Tecniche di sollevamento corrette**: Posizionamento del corpo, utilizzo delle gambe, movimento fluido e controllato
- **Esercizi specifici**: Rafforzamento muscolare, stretching, esercizi di stabilizzazione, preparazione fisica
- **Consapevolezza posturale**: Auto-monitoraggio, correzione posture, abitudini salutari durante e fuori dal lavoro
- **Stile di vita**: Importanza del riposo, alimentazione, attività fisica complementare

### Priorità di Implementazione
- **Interventi urgenti** (rischi immediati, LI ≥ 3.0): Azioni entro 24-48 ore, misure provvisorie immediate
- **Miglioramenti a medio termine** (1.0 ≤ LI < 3.0): Pianificazione 1-3 mesi, investimenti strutturali
- **Ottimizzazioni a lungo termine** (LI < 1.0): Miglioramenti continui 6-12 mesi, cultura della sicurezza
- **Analisi costi-benefici**: Valutazione economica degli interventi, ROI delle misure di miglioramento

## Piano di Azione Dettagliato

### Fase 1: Azioni Immediate (primi 30 giorni)
- Interventi urgenti per rischi immediati
- Formazione base del personale
- Misure provvisorie
- Monitoraggio iniziale

### Fase 2: Interventi a Breve Termine (1-3 mesi)
- Implementazione modifiche strutturali
- Acquisizione attrezzature
- Rotazione personale
- Procedure migliorate

### Fase 3: Miglioramenti a Lungo Termine (3-12 mesi)
- Ottimizzazione completa
- Cultura della sicurezza
- Monitoraggio continuo
- Miglioramenti progressivi

### Sistema di Monitoraggio
- Indicatori di performance (KPI)
- Valutazioni periodiche
- Feedback dei lavoratori
- Aggiornamenti continui

## Riferimenti Scientifici e Standard

### Standard Internazionali
- **NIOSH**: Complete Revised NIOSH Lifting Equation (1991)
- **ISO**: Standard ergonomici internazionali
- **CEN**: Normative europee
- **OSHA**: Regolamentazione americana

### Letteratura Scientifica
- Studi epidemiologici sull'ergonomia
- Ricerche biomeccaniche
- Analisi statistiche infortunistica
- Casi studio documentati

### Linee Guida Settoriali
- Raccomandazioni specifiche per settore
- Buone pratiche industriali
- Protocolli aziendali validati

**Report generato da ergonomista professionista certificato**
**Metodologia**: Revised NIOSH Lifting Equation (1991)
**Standard**: ISO 45001, OSHA Ergonomics Guidelines
**Data analisi**: {current_date}
**Conformità**: Standard internazionali di ergonomia e sicurezza sul lavoro
**Prossima revisione**: 12 mesi dalla data di emissione

REGOLE ASSOLUTE:
- USA ESCLUSIVAMENTE DATI ESATTI DAL JSON - MAI INVENTARE
- PRECISIONE ASSOLUTA SU VALORI E UNITÀ DI MISURA
- SII ESTREMAMENTE DESCRITTIVO E DETTAGLIATO IN OGNI SEZIONE
- FORNISCI SPOSIEGAZIONI BIOMECCANICHE E MEDICHE APPROFONDITE
- BASA OGNI RACCOMANDAZIONE SU EVIDENZE SCIENTIFICHE
- GENERA UN REPORT COMPLETO PER USO PROFESSIONALE"""

    def get_scenario_prompt(self, user_input: str) -> str:
        """Ottiene prompt per generazione scenario"""
        return self.SCENARIO_GENERATION_PROMPT.format(user_prompt=user_input)
    
    def get_report_prompt(self, scenario_data: Dict[str, Any], niosh_results: Dict[str, Any]) -> str:
        """Ottiene prompt per generazione report"""
        from datetime import datetime
        
        # Estrazione dati per template
        job_title = scenario_data.get('job_title', 'Posizione non specificata')
        job_description = scenario_data.get('job_description', 'Descrizione non disponibile')
        weight_lbs = scenario_data['task_variables']['weight_lbs']
        object_name = scenario_data['task_variables']['object_name']
        
        # Analisi origine
        origin_results = niosh_results['origin']
        rwl_origin = origin_results['rwl']
        li_origin = origin_results['li']
        
        # Analisi destinazione (se presente)
        destination_analysis = ""
        if 'destination' in niosh_results:
            dest_results = niosh_results['destination']
            destination_analysis = f"""
### Destination Analysis:
- **RWL**: {dest_results['rwl']} lbs - Spiegazione del calcolo
- **LI**: {dest_results['li']} - Interpretazione del rischio
- **Critical Multipliers**: Analisi dei moltiplicatori problematici a destinazione"""
        
        final_rwl = niosh_results['final_rwl']
        final_li = niosh_results['final_li']
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        return self.REPORT_SYNTHESIS_PROMPT.format(
            job_title=job_title,
            job_description=job_description,
            scenario_data=json.dumps(scenario_data, indent=2, ensure_ascii=False),
            niosh_results=json.dumps(niosh_results, indent=2, ensure_ascii=False),
            weight_lbs=weight_lbs,
            object_name=object_name,
            rwl_origin=rwl_origin,
            li_origin=li_origin,
            destination_analysis=destination_analysis,
            final_rwl=final_rwl,
            final_li=final_li,
            current_date=current_date
        )


# Funzioni di utilità per i prompt
def create_validation_prompt() -> str:
    """Prompt per validazione coerenza dati generati"""
    return """Sei un validatore esperto di dati ergonomici NIOSH.

Controlla questo JSON scenario e verifica:
1. Coerenza tra peso, frequenza e durata
2. Realismo delle distanze e angoli
3. Compatibilità dei parametri ergonomici
4. Presenza di tutti i campi richiesti

Segnala qualsiasi incoerenza e suggerisci correzioni.

JSON da validare:
{scenario_json}"""


def create_improvement_prompt(worst_multipliers: list) -> str:
    """Prompt per suggerimenti miglioramento basati su moltiplicatori critici"""
    mult_descriptions = {
        'HM': 'Horizontal Multiplier - Distanza orizzontale eccessiva',
        'VM': 'Vertical Multiplier - Altezza di sollevamento problematica', 
        'DM': 'Distance Multiplier - Distanza verticale eccessiva',
        'AM': 'Asymmetric Multiplier - Angolo di torsione elevato',
        'FM': 'Frequency Multiplier - Frequenza di sollevamento troppo alta',
        'CM': 'Coupling Multiplier - Qualità della presa insufficiente'
    }
    
    problems = []
    for mult_name, value in worst_multipliers:
        if value < 0.8:  # Soglia per moltiplicatori problematici
            description = mult_descriptions.get(mult_name, f'{mult_name} - valore basso')
            problems.append(f"- {description} (valore: {value:.2f})")
    
    return f"""Basato su questi moltiplicatori critici:
{chr(10).join(problems)}

Fornisci suggerimenti pratici e prioritari per ridurre il rischio ergonomico."""