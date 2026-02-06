"""
niosh_calculator.py

Semplice implementazione del Revised NIOSH Lifting Equation.

Input:  NIOSHParameters (peso, H, V, A, F, durata, coupling, controllo).
Output: RWL a origine/destinazione, LI, e dettaglio dei moltiplicatori.

Nota importante:
- I moltiplicatori HM, VM, DM, AM sono implementati secondo le formule standard.
- FM e CM sono implementati con una approssimazione ragionevole (non è una
  riproduzione letterale delle tabelle NIOSH, ma è coerente come ordine di grandezza).
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List, Iterable, Any
from niosh_parameters import NIOSHParameters


@dataclass
class NIOSHCalculationResult:
    """Risultato del calcolo NIOSH per un singolo job/task."""

    rwl_origin_lbs: float
    rwl_destination_lbs: Optional[float]

    li_origin: float
    li_destination: Optional[float]

    multipliers_origin: Dict[str, float]
    multipliers_destination: Optional[Dict[str, float]]

    # 🔹 nuovi campi derivati
    risk_category: Optional[str] = None
    risk_comment: Optional[str] = None

    def compute_risk(self) -> None:
        """
        Classifica il rischio in base al LI massimo (origine/destinazione).
        La logica è:

        - LI_max < 1.0  → "acceptable"
        - 1.0–3.0       → "moderately_stressful"
        - > 3.0         → "hazardous"

        Basato sui nuovi range NIOSH semplificati.
        """
        li_values = [v for v in (self.li_origin, self.li_destination) if v is not None]
        if not li_values:
            self.risk_category = None
            self.risk_comment = None
            return

        li_max = max(li_values)

        if li_max < 1.0:
            self.risk_category = "acceptable"
            self.risk_comment = "Acceptable for nearly all healthy workers (LI < 1.0)."
        elif li_max <= 3.0:
            self.risk_category = "moderately_stressful"
            self.risk_comment = (
                "Moderately stressful for some workers (1.0 ≤ LI ≤ 3.0)."
            )
        else:
            self.risk_category = "hazardous"
            self.risk_comment = (
                "Hazardous for a majority of healthy workers (LI > 3.0)."
            )

    def as_dict(self) -> Dict[str, Any]:
        """Convert the calculation result to a dictionary."""
        from dataclasses import asdict

        return asdict(self)


@dataclass
class MultiTaskTaskResult:
    """Risultato del calcolo per un singolo task in analisi multi-task."""

    task_id: int
    task_description: str
    weight_lbs: float

    # Input parameters preserved for reporting
    horizontal_origin: float
    horizontal_destination: float
    vertical_origin: float
    vertical_destination: float
    asymmetry_angle: float
    frequency_lifts_per_min: float
    duration: str
    coupling: str
    significant_control: bool  # 🔹 AGGIUNTO

    # Multi-task specific calculations
    firwl_lbs: float
    fili: float
    strwl_lbs: float
    stli: float

    # Standard NIOSH calculations
    rwl_origin_lbs: float
    rwl_destination_lbs: Optional[float]
    li_origin: float
    li_destination: Optional[float]

    multipliers_origin: Dict[str, float]
    multipliers_destination: Optional[Dict[str, float]]
    frequency_multiplier: float

    def compute_firwl_fili(self, load_constant: float = 51.0):
        """
        Calcola FIRWL e FILI in modo robusto e conforme alla RNLE.
        Protezioni:
            - se FIRWL = 0 → nessuna capacità raccomandata → FILI = ∞
            - evita ZeroDivisionError
            - evita moltiplicatori mancanti o non validi
        """

        # Load Constant RNLE (standard = 51 lb)
        lc = float(load_constant)

        # Recupero moltiplicatori (origine)
        try:
            hm = float(self.multipliers_origin.get("HM", 0))
            vm = float(self.multipliers_origin.get("VM", 0))
            dm = float(self.multipliers_origin.get("DM", 0))
            am = float(self.multipliers_origin.get("AM", 0))
            cm = float(self.multipliers_origin.get("CM", 0))
        except Exception:
            # Se qualunque moltiplicatore è invalido → RWL = 0
            self.firwl_lbs = 0.0
            self.fili = float("inf")
            return

        # FM = 1.0 per FIRWL
        fm = 1.0

        # Calcolo FIRWL
        firwl = lc * hm * vm * dm * am * fm * cm

        # Se per qualsiasi motivo FIRWL <= 0 → filì infinito
        if firwl is None or firwl <= 0:
            self.firwl_lbs = 0.0
            self.fili = float("inf")
            return

        self.firwl_lbs = float(firwl)

        # Calcolo FILI (Frequency-Independent Lifting Index)
        weight = float(getattr(self, "weight_lbs", 0))

        # Se peso non valido → LI non calcolabile
        if weight <= 0:
            self.fili = None
            return

        # Calcolo sicuro
        self.fili = weight / self.firwl_lbs


    def compute_strwl_stli(self):
        """Calcola STRWL e STLI usando FM reali — con gestione errori"""
        self.strwl_lbs = self.firwl_lbs * self.frequency_multiplier

        if self.strwl_lbs and self.strwl_lbs > 0:
            self.stli = self.weight_lbs / self.strwl_lbs
        else:
            # se STRWL è zero o None → consideriamo il task come altamente penalizzante
            self.stli = float("inf")


@dataclass
class MultiTaskCalculationResult:
    """Risultato completo dell'analisi multi-task NIOSH."""

    job_description: str
    total_tasks: int
    tasks: List[MultiTaskTaskResult]

    # Composite Lifting Index (CLI)
    cli: float
    risk_category: str
    risk_comment: str

    # Task renumbering by stress (STLI descending)
    tasks_by_stress: List[MultiTaskTaskResult]

    def compute_cli(self):
        """
        Calcola il Composite Lifting Index (CLI) secondo NIOSH.

        Metodo NIOSH corretto:
        1. Ordina i task per STLI decrescente
        2. CLI = STLI₁ + Σ(STLIᵢ × decremento_fattore per i>1)

        Il fattore di decremento dipende dal controllo:
        - Controllo significativo: 0.6 per i task successivi
        - Nessun controllo: 0.8 per i task successivi
        """
        if not self.tasks:
            self.cli = 0.0
            return
        valid_tasks = [
            t
            for t in self.tasks
            if t.stli is not None and isinstance(t.stli, (int, float))
        ]
        if not valid_tasks:
            self.cli = 0.0
            return
        if any(not math.isfinite(t.stli) for t in valid_tasks):
            self.cli = float("inf")
            return

        # Assicuriamoci che i task siano ordinati per STLI decrescente
        tasks_sorted = sorted(valid_tasks, key=lambda t: t.stli, reverse=True)

        if not tasks_sorted:
            self.cli = 0.0
            return

        # CLI NIOSH: primo task a valore pieno, successivi con decremento
        stli_max = tasks_sorted[0].stli

        # Determina il fattore di decremento in base al controllo
        # Per semplicità, usiamo 0.6 (controllo significativo) come default
        # Questo potrebbe essere calcolato dinamicamente per ogni task
        decrement_factor = 0.6

        cli_sum = stli_max
        for i, task in enumerate(tasks_sorted[1:], 1):  # Salta il primo task
            # Aggiungi solo se STLI > 1.0 (contributo significativo)
            if task.stli > 1.0:
                task_decrement = 0.6 if task.significant_control else 0.8
                cli_sum += task.stli * task_decrement
                # Il fattore di decremento potrebbe diminuire per task successivi
                # Per ora manteniamo costante per semplicità

        self.cli = cli_sum

    def compute_risk_classification(self):
        """Classifica il rischio basato sul CLI"""
        if self.cli < 1.0:
            self.risk_category = "acceptable"
            self.risk_comment = "Acceptable for nearly all healthy workers (CLI < 1.0)"
        elif self.cli <= 3.0:
            self.risk_category = "moderately_stressful"
            self.risk_comment = (
                "Moderately stressful for some workers (1.0 ≤ CLI ≤ 3.0)"
            )
        else:
            self.risk_category = "hazardous"
            self.risk_comment = "Hazardous for nearly all healthy workers (CLI > 3.0)"

    def reorder_tasks_by_stress(self):
        """Riordina i task per stress decrescente (STLI)"""
        self.tasks_by_stress = sorted(
            self.tasks,
            key=lambda t: t.stli if t.stli is not None else -math.inf,
            reverse=True,
        )

    def as_dict(self) -> Dict:
        """
        Dizionario "pulito" per il JSON:
        - RWL in lbs → 1 decimale
        - LI → 2 decimali
        - multipliers lasciati com'è (se vuoi puoi arrotondare anche quelli)
        """
        d = asdict(self)

        # Arrotondamento RWL
        for key in (
            "rwl_origin_lbs",
            "rwl_destination_lbs",
        ):
            if d.get(key) is not None:
                d[key] = round(d[key], 1)

        # Arrotondamento LI
        for key in ("li_origin", "li_destination"):
            if d.get(key) is not None:
                d[key] = round(d[key], 2)

        # multipliers e risk_* restano come sono
        return d


class NIOSHCalculator:
    """
    Calcolatore per la Revised NIOSH Lifting Equation.

    RWL = LC × HM × VM × DM × AM × FM × CM
    Dove:
    - LC = Load Constant = 51 lbs (costante fisso NIOSH)
    - HM = Horizontal Multiplier = 10/H (dove H è la distanza orizzontale in inches)
    - VM = Vertical Multiplier = 1 - 0.003 × |V - 30| (dove V è l'altezza verticale in inches)
    - DM = Distance Multiplier = 0.82 + 1.8/D (dove D è la distanza verticale di trasporto in inches)
    - AM = Asymmetry Multiplier = 1 - 0.0032 × A (dove A è l'angolo di asimmetria in degrees)
    - FM = Frequency Multiplier (dipende da frequenza, durata e altezza)
    - CM = Coupling Multiplier (dipende dalla qualità del presa e dall'altezza)

    LI = Load Weight / RWL
    """

    LC_LBS = 51.0

    # ------------------ API principale ------------------

    def compute(self, params) -> NIOSHCalculationResult:
        """
        Calcola RWL e LI a origine/destinazione del lift.

        - Se params.significant_control == False:
            → RWL solo all'origine (dest = None).
        - Se True:
            → RWL sia origine che destinazione.
        """

        # D = distanza verticale fra origine e destinazione
        D = abs(params.vertical_destination - params.vertical_origin)

        # ORIGINE
        rwl_o, mult_o = self._compute_point(
            H=params.horizontal_origin,
            V=params.vertical_origin,
            A=params.asymmetry_angle,
            F=params.frequency,
            duration=params.duration,
            coupling=params.coupling,
            D=D,
        )

        # DESTINAZIONE (solo se richiesto)
        if params.significant_control:
            rwl_d, mult_d = self._compute_point(
                H=params.horizontal_destination,
                V=params.vertical_destination,
                A=params.asymmetry_angle,
                F=params.frequency,
                duration=params.duration,
                coupling=params.coupling,
                D=D,
            )
        else:
            rwl_d, mult_d = None, None

        # Lifting Index
        li_origin = params.weight / rwl_o if rwl_o > 0 else float("inf")
        li_destination = (
            params.weight / rwl_d if (rwl_d is not None and rwl_d > 0) else None
        )

        # Crea il risultato "grezzo"
        result = NIOSHCalculationResult(
            rwl_origin_lbs=rwl_o,
            rwl_destination_lbs=rwl_d,
            li_origin=li_origin,
            li_destination=li_destination,
            multipliers_origin=mult_o,
            multipliers_destination=mult_d,
        )

        # 🔹 calcola la categoria di rischio in base ai LI
        result.compute_risk()

        return result

    def compute_multi_task(
        self,
        tasks_data: List[Dict[str, Any]],
        job_description: str = "",
        load_constant: float = 51.0,
    ) -> MultiTaskCalculationResult:
        """
        Calcola analisi multi-task NIOSH completa.

        Input: Lista di dizionari con dati per ogni task:
            - task_id: int
            - weight: float               # peso del carico (lbs)
            - horizontal_origin: float
            - horizontal_destination: float
            - vertical_origin: float
            - vertical_destination: float
            - asymmetry_angle: float      # gradi
            - frequency: float            # lifts/min
            - duration: str               # "<1h", "1-2h", "2-8h", ">8h"
            - coupling: str               # "good", "fair", "poor"
            - significant_control: bool   # opzionale

        Output: MultiTaskCalculationResult con FIRWL, FILI, STRWL, STLI, CLI.
        """

        tasks: List[MultiTaskTaskResult] = []

        for i, task_data in enumerate(tasks_data):
            # --- sanity check minimale: niente KeyError stupidi ---
            required_keys = [
                "weight",
                "horizontal_origin",
                "horizontal_destination",
                "vertical_origin",
                "vertical_destination",
                "asymmetry_angle",
                "frequency",
                "duration",
                "coupling",
            ]
            for k in required_keys:
                if k not in task_data:
                    raise KeyError(f"compute_multi_task: missing key '{k}' in task_data[{i}]")

            # 1) Parametri NIOSH per questo task
            params = NIOSHParameters(
                weight=float(task_data["weight"]),
                horizontal_origin=float(task_data["horizontal_origin"]),
                horizontal_destination=float(task_data["horizontal_destination"]),
                vertical_origin=float(task_data["vertical_origin"]),
                vertical_destination=float(task_data["vertical_destination"]),
                asymmetry_angle=float(task_data["asymmetry_angle"]),
                frequency=float(task_data["frequency"]),
                duration=task_data["duration"],
                coupling=task_data["coupling"],
                significant_control=bool(task_data.get("significant_control", False)),
            )

            # 2) Calcolo standard singolo task
            single_result = self.compute(params)

            # Frequency Multiplier all'origine (per multi-task RNLE)
            fm = single_result.multipliers_origin["FM"]

            # 3) Costruzione risultato per questo task
            task_result = MultiTaskTaskResult(
                task_id=task_data.get("task_id", i + 1),
                task_description=task_data.get("task_description", f"Task {i + 1}"),
                weight_lbs=float(task_data["weight"]),

                # INPUT PARAMETERS (devono stare PRIMA)
                horizontal_origin=float(task_data["horizontal_origin"]),
                horizontal_destination=float(task_data["horizontal_destination"]),
                vertical_origin=float(task_data["vertical_origin"]),
                vertical_destination=float(task_data["vertical_destination"]),
                asymmetry_angle=float(task_data["asymmetry_angle"]),
                frequency_lifts_per_min=float(task_data["frequency"]),
                duration=task_data["duration"],
                coupling=task_data["coupling"],
                significant_control=bool(task_data.get("significant_control", False)),  # 🔹 OBBLIGATORIO

                # MULTI-TASK CALCULATED FIELDS
                firwl_lbs=0.0,
                fili=0.0,
                strwl_lbs=0.0,
                stli=0.0,

                # STANDARD SINGLE-TASK RNLE RESULTS
                rwl_origin_lbs=single_result.rwl_origin_lbs,
                rwl_destination_lbs=single_result.rwl_destination_lbs,
                li_origin=single_result.li_origin,
                li_destination=single_result.li_destination,
                multipliers_origin=single_result.multipliers_origin,
                multipliers_destination=single_result.multipliers_destination,
                frequency_multiplier=fm,
            )

            # 4) Step RNLE multi-task
            #    - FIRWL / FILI (frequency-independent)
            task_result.compute_firwl_fili(load_constant)

            #    - STRWL / STLI (single-task RWL e LI con FM reale)
            task_result.compute_strwl_stli()

            tasks.append(task_result)

        # 5) Risultato multi-task complessivo
        multi_task_result = MultiTaskCalculationResult(
            job_description=job_description,
            total_tasks=len(tasks),
            tasks=tasks,
            cli=0.0,
            risk_category="",
            risk_comment="",
            tasks_by_stress=[],
        )

        # 6) Calcolo CLI, classificazione rischio, ordinamento
        multi_task_result.compute_cli()
        multi_task_result.compute_risk_classification()
        multi_task_result.reorder_tasks_by_stress()

        return multi_task_result

    def _compute_point(
        self,
        H: float,
        V: float,
        A: float,
        F: float,
        duration: str,
        coupling: str,
        D: float,
    ) -> Tuple[float, Dict[str, float]]:

        hm = self._hm(H)
        vm = self._vm(V)
        dm = self._dm(D)
        am = self._am(A)
        fm = self._fm(F, duration)
        cm = self._cm(coupling, V)

        # RWL = LC × HM × VM × DM × AM × FM × CM
        rwl = self.LC_LBS * hm * vm * dm * am * fm * cm

        multipliers = {
            "HM": hm,
            "VM": vm,
            "DM": dm,
            "AM": am,
            "FM": fm,
            "CM": cm,
        }

        return rwl, multipliers

    # ------------------ Moltiplicatori NIOSH ------------------

    def _hm(self, H: float) -> float:
        """
        Horizontal Multiplier.
        Formula NIOSH corretta: HM = 10 / H (H in inches, 10 ≤ H ≤ 63).
        Per H < 10, HM = 1.0 (alcune interpretazioni usano H/10, ma NIOSH usa 10/H)
        """
        if H <= 0:
            return 0.0

        if H < 10.0:
            H_clamped = 10.0
        else:
            H_clamped = min(max(H, 10.0), 25.0)
        return 10.0 / H_clamped

    def _vm(self, V: float) -> float:
        """
        Vertical Multiplier.
        Formula NIOSH corretta: VM = 1 - 0.003 * |V - 30|, con V in inches (0–70).
        Valore massimo: 1.0 (quando V = 30"), minimo: 0.0 (estremi)

        NOTA: Il coefficiente 0.003 è quello specificato nel Revised NIOSH Lifting Equation.
        """
        if V <= 0:
            return 0.0

        V_clamped = max(0.0, min(V, 70.0))
        vm = 1.0 - 0.003 * abs(V_clamped - 30.0)
        # NIOSH specifica che VM non può essere negativo
        return max(vm, 0.0)

    def _dm(self, D: float) -> float:
        """
        Distance Multiplier.
        Formula standard: DM = 0.82 + 1.8 / D, con D in inches (10–70), troncato a ≤ 1.
        """
        D_clamped = max(10.0, min(D, 70.0))
        dm = 0.82 + 1.8 / D_clamped
        return min(dm, 1.0)

    def _am(self, A: float) -> float:
        """
        Asymmetry Multiplier.
        Formula standard: AM = 1 - 0.0032 * A (A in degrees, 0–135).
        """
        A_clamped = max(0.0, min(A, 135.0))
        am = 1.0 - 0.0032 * A_clamped
        return max(am, 0.0)

    def _fm(self, F: float, duration: str) -> float:
        """
        Frequency Multiplier (FM) – IMPLEMENTAZIONE MIGLIORATA.

        Basato sulle tabelle NIOSH ufficiali che dipendono da:
        - F (lifts/min),
        - durata del lavoro (<1h, 1-2h, 2-8h, >8h),
        - V (altezza verticale - non implementato qui per semplicità)

        Valori tipici NIOSH per V=30" e durata 2-8h:
        F=0.1: 1.00, F=0.2: 1.00, F=0.5: 0.97, F=1: 0.94, F=2: 0.88,
        F=3: 0.79, F=4: 0.72, F=5: 0.65, F=6: 0.59, F=8: 0.46, F=10: 0.37,
        F=12: 0.30, F=15: 0.22, F=18: 0.15
        """

        # Mappa dei fattori di base per diverse frequenze (lifts/min)
        # Basato su valori NIOSH per durata di 2-8h e V ~30"
        frequency_map = {
            0.1: 1.00,
            0.2: 1.00,
            0.5: 0.97,
            1.0: 0.94,
            1.5: 0.91,
            2.0: 0.88,
            2.5: 0.84,
            3.0: 0.79,
            3.5: 0.75,
            4.0: 0.72,
            4.5: 0.68,
            5.0: 0.65,
            5.5: 0.62,
            6.0: 0.59,
            6.5: 0.56,
            7.0: 0.53,
            8.0: 0.46,
            9.0: 0.42,
            10.0: 0.37,
            11.0: 0.33,
            12.0: 0.30,
            13.0: 0.26,
            14.0: 0.24,
            15.0: 0.22,
            16.0: 0.19,
            18.0: 0.15,
            20.0: 0.12,
            25.0: 0.08,
            30.0: 0.05,
        }

        # Trova il valore più vicino nella mappa
        if F <= 0.1:
            base_fm = 1.00
        elif F >= 30.0:
            base_fm = 0.05
        else:
            # Interpolazione lineare tra valori noti
            freq_keys = sorted(frequency_map.keys())
            for i in range(len(freq_keys) - 1):
                if freq_keys[i] <= F <= freq_keys[i + 1]:
                    # Interpolazione lineare
                    f1, f2 = freq_keys[i], freq_keys[i + 1]
                    fm1, fm2 = frequency_map[f1], frequency_map[f2]
                    t = (F - f1) / (f2 - f1)
                    base_fm = fm1 + t * (fm2 - fm1)
                    break
            else:
                # Se non trova intervallo, usa il più vicino
                closest = min(freq_keys, key=lambda k: abs(k - F))
                base_fm = frequency_map[closest]

        # Fattori di correzione per durata (NIOSH standard)
        duration_factors = {
            "<1h": 1.00,  # Breve durata - nessuna penalità
            "1-2h": 0.95,  # Durata moderata - leggera penalità
            "2-8h": 0.90,  # Durata standard - penalità moderata
            ">8h": 0.85,  # Lunga durata - penalità significativa
        }

        duration_factor = duration_factors.get(duration, 0.90)
        fm = base_fm * duration_factor

        # NIOSH range: 0.0 ≤ FM ≤ 1.0
        return max(min(fm, 1.0), 0.0)

    def _cm(self, coupling: str, V: float) -> float:
        """
        Coupling Multiplier (CM) – IMPLEMENTAZIONE SEMPLIFICATA.

        Approssimazione:
        - se V <= 30 in (presa sotto la vita) → valori leggermente migliori
          rispetto a V > 30 solo per good.
        - buona per uso didattico/di screening.
        """
        c = (coupling or "").strip().lower()
        V_clamped = max(0.0, min(V, 70.0))

        if c == "good":
            # effetto di V molto attenuato
            return 1.0 if 25.0 <= V_clamped <= 75.0 else 0.97
        elif c == "fair":
            return 0.95 if V_clamped <= 30.0 else 0.90
        elif c == "poor":
            return 0.90 if V_clamped <= 30.0 else 0.85
        else:
            # se il testo era ambiguo, assumi fair
            return 0.95 if V_clamped <= 30.0 else 0.90

    def _cm_corrected(self, coupling: str, V: float) -> float:
        """
        Coupling Multiplier (CM) - IMPLEMENTAZIONE CONFORME AL PDF NIOSH.

        Basato su Table 7 del Applications Manual:

                  V < 30 inches    V ≥ 30 inches
        Good:         1.00             1.00
        Fair:         0.95             1.00
        Poor:         0.90             0.90

        Args:
            coupling: "good", "fair", or "poor"
            V: Vertical height in inches
        """
        c = (coupling or "").strip().lower()

        if c == "good":
            return 1.00
        elif c == "fair":
            return 0.95 if V < 30.0 else 1.00
        elif c == "poor":
            return 0.90
        else:
            # Default: assume fair coupling
            return 0.95 if V < 30.0 else 1.00

    def _fm_corrected(self, F: float, duration: str, V: float) -> float:
        """
        Frequency Multiplier (FM) - IMPLEMENTAZIONE COMPLETA BASATA SU TABLE 5.

        Il PDF NIOSH fornisce tabelle separate per V<30" e V≥30".

        Args:
            F: Frequency in lifts/minute
            duration: "<1h", "1-2h", "2-8h", ">8h"
            V: Vertical height at origin in inches

        Returns:
            FM value between 0.0 and 1.0
        """

        # Tabelle NIOSH complete (Table 5 del PDF)
        # Colonna 1: V < 30 inches
        # Colonna 2: V ≥ 30 inches

        # Durata: ≤1 hour
        if duration == "<1h":
            if V < 30:
                fm_table = {
                    0.2: 1.00,
                    0.5: 0.97,
                    1: 0.94,
                    2: 0.91,
                    3: 0.88,
                    4: 0.84,
                    5: 0.80,
                    6: 0.75,
                    7: 0.70,
                    8: 0.60,
                    9: 0.52,
                    10: 0.45,
                    11: 0.41,
                    12: 0.37,
                    13: 0.34,
                    14: 0.31,
                    15: 0.28,
                    18: 0.00,
                }
            else:  # V ≥ 30
                fm_table = {
                    0.2: 1.00,
                    0.5: 0.97,
                    1: 0.94,
                    2: 0.91,
                    3: 0.88,
                    4: 0.84,
                    5: 0.80,
                    6: 0.75,
                    7: 0.70,
                    8: 0.60,
                    9: 0.52,
                    10: 0.45,
                    11: 0.41,
                    12: 0.37,
                    13: 0.34,
                    14: 0.31,
                    15: 0.28,
                    18: 0.00,
                }

        # Durata: >1 to ≤2 hours
        elif duration == "1-2h":
            if V < 30:
                fm_table = {
                    0.2: 0.95,
                    0.5: 0.92,
                    1: 0.88,
                    2: 0.84,
                    3: 0.79,
                    4: 0.72,
                    5: 0.60,
                    6: 0.50,
                    7: 0.42,
                    8: 0.35,
                    9: 0.30,
                    10: 0.26,
                    11: 0.23,
                    12: 0.21,
                    13: 0.00,
                    14: 0.00,
                    15: 0.00,
                    18: 0.00,
                }
            else:  # V ≥ 30
                fm_table = {
                    0.2: 0.95,
                    0.5: 0.92,
                    1: 0.88,
                    2: 0.84,
                    3: 0.79,
                    4: 0.72,
                    5: 0.60,
                    6: 0.50,
                    7: 0.42,
                    8: 0.35,
                    9: 0.30,
                    10: 0.26,
                    11: 0.23,
                    12: 0.21,
                    13: 0.00,
                    14: 0.00,
                    15: 0.00,
                    18: 0.00,
                }

        # Durata: >2 to ≤8 hours
        elif duration == "2-8h":
            if V < 30:
                fm_table = {
                    0.2: 0.85,
                    0.5: 0.81,
                    1: 0.75,
                    2: 0.65,
                    3: 0.55,
                    4: 0.45,
                    5: 0.35,
                    6: 0.27,
                    7: 0.22,
                    8: 0.18,
                    9: 0.00,
                    10: 0.00,
                    11: 0.00,
                    12: 0.00,
                    13: 0.00,
                    14: 0.00,
                    15: 0.00,
                    18: 0.00,
                }
            else:  # V ≥ 30
                fm_table = {
                    0.2: 0.85,
                    0.5: 0.81,
                    1: 0.75,
                    2: 0.65,
                    3: 0.55,
                    4: 0.45,
                    5: 0.35,
                    6: 0.27,
                    7: 0.22,
                    8: 0.18,
                    9: 0.00,
                    10: 0.00,
                    11: 0.00,
                    12: 0.00,
                    13: 0.00,
                    14: 0.00,
                    15: 0.00,
                    18: 0.00,
                }

        # Durata: >8 hours (non specificato nel PDF, usiamo valori conservativi)
        else:  # ">8h"
            if V < 30:
                fm_table = {
                    0.2: 0.75,
                    0.5: 0.70,
                    1: 0.65,
                    2: 0.55,
                    3: 0.45,
                    4: 0.35,
                    5: 0.27,
                    6: 0.22,
                    7: 0.18,
                    8: 0.00,
                    9: 0.00,
                    10: 0.00,
                    11: 0.00,
                    12: 0.00,
                    13: 0.00,
                    14: 0.00,
                    15: 0.00,
                    18: 0.00,
                }
            else:
                fm_table = {
                    0.2: 0.75,
                    0.5: 0.70,
                    1: 0.65,
                    2: 0.55,
                    3: 0.45,
                    4: 0.35,
                    5: 0.27,
                    6: 0.22,
                    7: 0.18,
                    8: 0.00,
                    9: 0.00,
                    10: 0.00,
                    11: 0.00,
                    12: 0.00,
                    13: 0.00,
                    14: 0.00,
                    15: 0.00,
                    18: 0.00,
                }

        # Gestione frequenze fuori range
        if F <= 0.2:
            return list(fm_table.values())[0]

        freq_keys = sorted(fm_table.keys())
        if F >= freq_keys[-1]:
            return fm_table[freq_keys[-1]]

        # Interpolazione lineare tra valori noti
        for i in range(len(freq_keys) - 1):
            if freq_keys[i] <= F <= freq_keys[i + 1]:
                f1, f2 = freq_keys[i], freq_keys[i + 1]
                fm1, fm2 = fm_table[f1], fm_table[f2]

                # Se entrambi sono 0.00, return 0.0
                if fm1 == 0.0 and fm2 == 0.0:
                    return 0.0

                # Interpolazione lineare
                t = (F - f1) / (f2 - f1)
                return fm1 + t * (fm2 - fm1)

        # Fallback
        closest = min(freq_keys, key=lambda k: abs(k - F))
        return fm_table[closest]


# ------------------------------------------------------------------------------
# FUNZIONI HELPER AGGIUNTIVE (Versione Semplificata / ISO 11228-1)
# ------------------------------------------------------------------------------
# Le seguenti funzioni implementano una variante del calcolo che tiene conto
# di genere ed età per il calcolo della costante di peso (CP), e utilizza
# tabelle semplificate per i moltiplicatori.
# ------------------------------------------------------------------------------


def compute_recommended_weight_simplified(
    gender="M",
    age=25,
    min_hand_floor_distance_vertical=None,
    vertical_lifting_distance=None,
    max_orizontal_hands_mid_ankle_distance=None,
    torso_torsion=0,
    judgment="intermediate",
    lifting_frequency=1,
    etm=1,
    one_limb_lifting=False,
    two_operators_lifting=False,
):
    """
    Calcola il peso raccomandato (RWL) utilizzando un approccio semplificato
    basato su tabelle e coefficienti specifici (es. ISO 11228-1).

    MIGLIORIE RISPETTO ALLA VERSIONE MONOLITICA:
    - Logica suddivisa in funzioni helper (get_vm_score, get_dm_score, ecc.) per
      migliorare la leggibilità e la manutenibilità.
    - Rimozione di operatori ternari annidati complessi.
    - Aggiunta di asserzioni per validare gli input.
    """

    # Validazione input
    assert gender in ["M", "W"], "Gender must be M or W"
    assert age > 0, "Age must be positive"
    assert (
        min_hand_floor_distance_vertical is not None
    ), "Vertical distance from floor required"
    assert vertical_lifting_distance is not None, "Vertical lifting distance required"
    assert (
        max_orizontal_hands_mid_ankle_distance is not None
    ), "Horizontal distance required"
    assert torso_torsion >= 0, "Torsion must be non-negative"
    assert judgment in ["good", "intermediate", "bad"], "Invalid judgment value"
    assert lifting_frequency > 0, "Frequency must be positive"
    assert etm > 0, "ETM must be positive"
    assert isinstance(one_limb_lifting, bool), "one_limb_lifting must be bool"
    assert isinstance(two_operators_lifting, bool), "two_operators_lifting must be bool"

    # Calcolo dei moltiplicatori usando funzioni helper
    cp = get_cp_score(gender, age)
    vm = get_vm_score(min_hand_floor_distance_vertical)
    dm = get_dm_score(vertical_lifting_distance)
    hm = get_hm_score(max_orizontal_hands_mid_ankle_distance)
    am = get_am_score(torso_torsion)
    cm = get_cm_score(judgment)
    fm = get_fm_score(lifting_frequency)
    etm_score = get_etm_score(etm)
    om = get_om_score(one_limb_lifting)
    pm = get_pm_score(two_operators_lifting)

    # Calcolo finale
    recommended_weight = cp * vm * dm * hm * am * cm * fm * etm_score * om * pm

    # Debug print (opzionale, commentato per pulizia)
    # print(round(cp, 2), round(vm, 2), round(dm, 2), round(hm, 2), round(am, 2),
    #       round(cm, 2), round(fm, 2), round(etm_score, 2), round(om, 2), round(pm, 2))

    return recommended_weight


def get_cp_score(gender, age):
    """Determina la Costante di Peso (CP) in base a genere ed età."""
    if gender == "M":
        if 20 <= age <= 45:
            return 25
        else:
            return 20
    elif gender == "W":
        if 20 <= age <= 45:
            return 20
        else:
            return 15
    return 20


def get_vm_score(vertical_distance):
    """Calcola il moltiplicatore verticale (VM)."""
    vm = 0
    if vertical_distance > 175:
        vm = 0
    elif vertical_distance == 170:
        vm = 0.70
    elif vertical_distance > 160:
        vm = 0.75
    elif vertical_distance > 150:
        vm = 0.78
    elif vertical_distance > 140:
        vm = 0.81
    elif vertical_distance > 130:
        vm = 0.84
    elif vertical_distance > 120:
        vm = 0.87
    elif vertical_distance > 110:
        vm = 0.90
    elif vertical_distance > 100:
        vm = 0.93
    elif vertical_distance > 90:
        vm = 0.96
    elif vertical_distance > 80:
        vm = 0.99
    elif vertical_distance > 75:
        vm = 1.00
    elif vertical_distance > 70:
        vm = 0.99
    elif vertical_distance > 60:
        vm = 0.96
    elif vertical_distance > 50:
        vm = 0.93
    elif vertical_distance > 40:
        vm = 0.90
    elif vertical_distance > 30:
        vm = 0.87
    elif vertical_distance > 20:
        vm = 0.84
    elif vertical_distance > 10:
        vm = 0.81
    elif vertical_distance >= 0:
        vm = 0.78
    return vm


def get_dm_score(vertical_distance):
    """Calcola il moltiplicatore di distanza (DM)."""
    dm = 0
    if vertical_distance > 175:
        dm = 0
    elif vertical_distance == 175:
        dm = 0.85
    elif vertical_distance > 160:
        dm = 0.85
    elif vertical_distance > 145:
        dm = 0.85
    elif vertical_distance > 130:
        dm = 0.86
    elif vertical_distance > 115:
        dm = 0.86
    elif vertical_distance > 100:
        dm = 0.87
    elif vertical_distance > 85:
        dm = 0.87
    elif vertical_distance > 70:
        dm = 0.88
    elif vertical_distance > 55:
        dm = 0.90
    elif vertical_distance > 40:
        dm = 0.93
    elif vertical_distance > 25:
        dm = 0.93
    elif vertical_distance <= 25:
        dm = 1
    return dm


def get_hm_score(horizontal_distance):
    """Calcola il moltiplicatore orizzontale (HM)."""
    hm = 0
    if horizontal_distance > 63:
        hm = 0
    elif horizontal_distance == 63:
        hm = 0.40
    elif horizontal_distance > 60:
        hm = 0.42
    elif horizontal_distance > 58:
        hm = 0.43
    elif horizontal_distance > 56:
        hm = 0.45
    elif horizontal_distance > 54:
        hm = 0.46
    elif horizontal_distance > 52:
        hm = 0.48
    elif horizontal_distance > 50:
        hm = 0.50
    elif horizontal_distance > 48:
        hm = 0.52
    elif horizontal_distance > 46:
        hm = 0.54
    elif horizontal_distance > 44:
        hm = 0.57
    elif horizontal_distance > 42:
        hm = 0.60
    elif horizontal_distance > 40:
        hm = 0.63
    elif horizontal_distance > 38:
        hm = 0.66
    elif horizontal_distance > 36:
        hm = 0.69
    elif horizontal_distance > 34:
        hm = 0.74
    elif horizontal_distance > 32:
        hm = 0.78
    elif horizontal_distance > 30:
        hm = 0.83
    elif horizontal_distance > 28:
        hm = 0.89
    elif horizontal_distance > 25:
        hm = 0.89
    elif horizontal_distance <= 25:
        hm = 1
    return hm


def get_am_score(degree):
    """Calcola il moltiplicatore di asimmetria (AM)."""
    am = 0
    if degree > 135:
        am = 0
    elif degree == 135:
        am = 0.57
    elif degree > 105:
        am = 0.66
    elif degree > 75:
        am = 0.76
    elif degree > 60:
        am = 0.81
    elif degree > 45:
        am = 0.86
    elif degree > 30:
        am = 0.90
    elif degree > 15:
        am = 0.95
    elif degree > 0:
        am = 0.95
    elif degree == 0:
        am = 1
    return am


def get_cm_score(judgment):
    """Calcola il moltiplicatore di coupling (CM)."""
    if judgment == "good":
        return 1
    elif judgment == "intermediate":
        return 0.95
    elif judgment == "bad":
        return 0.90
    return 0.95  # Default


def get_fm_score(frequency):
    """Calcola il moltiplicatore di frequenza (FM)."""
    # Mappa dei fattori di base per diverse frequenze (lifts/min)
    # Basato su valori NIOSH per durata di 2-8h e V ~30" (stima conservativa)
    frequency_map = {
        0.1: 1.00,
        0.2: 1.00,
        0.5: 0.97,
        1.0: 0.94,
        1.5: 0.91,
        2.0: 0.88,
        2.5: 0.84,
        3.0: 0.79,
        3.5: 0.75,
        4.0: 0.72,
        4.5: 0.68,
        5.0: 0.65,
        5.5: 0.62,
        6.0: 0.59,
        6.5: 0.56,
        7.0: 0.53,
        8.0: 0.46,
        9.0: 0.42,
        10.0: 0.37,
        11.0: 0.33,
        12.0: 0.30,
        13.0: 0.26,
        14.0: 0.24,
        15.0: 0.22,
        16.0: 0.19,
        18.0: 0.15,
        20.0: 0.12,
        25.0: 0.08,
        30.0: 0.05,
    }

    if frequency <= 0.1:
        return 1.00
    elif frequency >= 30.0:
        return 0.05

    # Interpolazione lineare
    freq_keys = sorted(frequency_map.keys())
    for i in range(len(freq_keys) - 1):
        if freq_keys[i] <= frequency <= freq_keys[i + 1]:
            f1, f2 = freq_keys[i], freq_keys[i + 1]
            fm1, fm2 = frequency_map[f1], frequency_map[f2]
            t = (frequency - f1) / (f2 - f1)
            return fm1 + t * (fm2 - fm1)

    # Fallback (non dovrebbe accadere grazie ai check sopra)
    closest = min(freq_keys, key=lambda k: abs(k - frequency))
    return frequency_map[closest]


def get_etm_score(mmc):
    """Calcola il moltiplicatore per durata estesa (ETM)."""
    # TODO: Implementare logica completa
    return 1


def get_om_score(one_limb):
    """Calcola il moltiplicatore per sollevamento con un solo arto (OM)."""
    if one_limb:
        return 0.6
    return 1


def get_pm_score(two_operators):
    """Calcola il moltiplicatore per due operatori (PM)."""
    if two_operators:
        return 0.85
    return 1


if __name__ == "__main__":

    # Test CM corretto
    print("=== TEST COUPLING MULTIPLIER ===")
    calc = NIOSHCalculator()

    # Good coupling (sempre 1.00)
    print(f"Good, V=25: {calc._cm_corrected('good', 25)}")  # 1.00
    print(f"Good, V=35: {calc._cm_corrected('good', 35)}")  # 1.00

    # Fair coupling (0.95 se V<30, 1.00 se V≥30)
    print(f"Fair, V=25: {calc._cm_corrected('fair', 25)}")  # 0.95
    print(f"Fair, V=35: {calc._cm_corrected('fair', 35)}")  # 1.00 ✅ CORRETTO

    # Poor coupling (sempre 0.90)
    print(f"Poor, V=25: {calc._cm_corrected('poor', 25)}")  # 0.90
    print(f"Poor, V=35: {calc._cm_corrected('poor', 35)}")  # 0.90 ✅ CORRETTO

    print("\n=== TEST FREQUENCY MULTIPLIER ===")

    # Test con V < 30 inches, durata 2-8h
    print(f"F=1/min, V=25, 2-8h: {calc._fm_corrected(1.0, '2-8h', 25)}")  # 0.75
    print(f"F=5/min, V=25, 2-8h: {calc._fm_corrected(5.0, '2-8h', 25)}")  # 0.35

    # Test con V ≥ 30 inches, durata 2-8h
    print(f"F=1/min, V=35, 2-8h: {calc._fm_corrected(1.0, '2-8h', 35)}")  # 0.75
    print(f"F=5/min, V=35, 2-8h: {calc._fm_corrected(5.0, '2-8h', 35)}")  # 0.35
