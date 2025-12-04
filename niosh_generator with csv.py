import csv
import itertools
import json
import requests
from pathlib import Path
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

class NIOSHExampleGenerator:
    def __init__(
        self,
        csv_file,
        output_file="esempi_niosh.txt",
        ollama_url="http://localhost:11434",
    ):
        """
        Inizializza il generatore di esempi NIOSH con FPS

        Args:
            csv_file: Percorso del file CSV
            output_file: File di output per gli esempi
            ollama_url: URL dell'API Ollama (default locale)
        """
        self.csv_file = csv_file
        self.output_file = output_file
        self.ollama_url = ollama_url
        self.combinations_used = set()
        self.vectorizer = TfidfVectorizer()

    def load_csv_data(self):
        data = {}
        with open(self.csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col, value in row.items():
                    if value is None:
                        continue
                    value = value.strip()
                    if value:
                        data.setdefault(col, []).append(value)

        # rimuovi duplicati mantenendo l'ordine
        for col in data:
            seen = set()
            unique = []
            for item in data[col]:
                if item not in seen:
                    unique.append(item)
                    seen.add(item)
            data[col] = unique

        return data

    def combination_to_text(self, combination):
        """
        Converte una combinazione in testo per l'embedding

        Args:
            combination: Dizionario con la combinazione

        Returns:
            str: Rappresentazione testuale della combinazione
        """
        # Unisci tutti i valori in un'unica stringa per l'embedding
        return " ".join([str(v) for v in combination.values()])

    def get_most_dissimilar_combinations_fps(self, all_combinations, k):
        """
        Seleziona le k combinazioni più dissimili usando Farthest Point Sampling

        Args:
            all_combinations: Lista di dizionari con le combinazioni
            k: Numero di combinazioni da selezionare

        Returns:
            list: Combinazioni selezionate per massima diversità
        """
        if not all_combinations or k <= 0:
            return []

        # Se richiediamo più combinazioni di quelle disponibili, restituisci tutte
        if k >= len(all_combinations):
            return all_combinations

        print(f"Generazione embeddings per {len(all_combinations)} combinazioni...")

        # Converti le combinazioni in testo
        texts = [self.combination_to_text(combo) for combo in all_combinations]

        # Genera embeddings TF-IDF
        embeddings = self.vectorizer.fit_transform(texts).toarray()
        N = embeddings.shape[0]

        print(f"Applicazione Farthest Point Sampling (FPS)...")

        # Array per tracciare gli indici selezionati
        selected_indices = []

        # 1. Inizializza con un punto casuale
        first_idx = np.random.randint(0, N)
        selected_indices.append(first_idx)

        # Array per tracciare la distanza minima di ogni punto dal set selezionato
        min_dists = np.full(N, np.inf)

        # 2. Iterativamente seleziona il punto più lontano
        for i in range(k - 1):
            # Calcola la distanza dal punto appena selezionato a tutti gli altri
            current_selected = embeddings[selected_indices[-1]].reshape(1, -1)
            dists = cosine_distances(embeddings, current_selected).flatten()

            # Aggiorna le distanze minime
            min_dists = np.minimum(min_dists, dists)

            # Seleziona il punto con la massima distanza minima
            # Questo è il punto più lontano dal set corrente
            next_idx = np.argmax(min_dists)
            selected_indices.append(next_idx)

            if (i + 1) % 10 == 0:
                print(f"  Selezionate {i + 1}/{k - 1} combinazioni diverse...")

        # Restituisci le combinazioni selezionate
        diverse_combinations = [all_combinations[idx] for idx in selected_indices]

        print(
            f"✓ FPS completato: {len(diverse_combinations)} combinazioni massimamente diverse"
        )

        return diverse_combinations

    def generate_combinations(
        self, data, selected_columns=None, max_combinations=None, use_fps=True
    ):
        """
        Genera combinazioni uniche con FPS per massima diversità

        Args:
            data: Dizionario con i dati delle colonne
            selected_columns: Lista di colonne da usare (None = tutte)
            max_combinations: Numero massimo di combinazioni da generare
            use_fps: Se True, usa Farthest Point Sampling (consigliato)
        """
        if selected_columns is None:
            selected_columns = list(data.keys())

        # Filtra solo le colonne selezionate
        filtered_data = {col: data[col] for col in selected_columns if col in data}

        # Genera tutte le combinazioni possibili
        column_names = list(filtered_data.keys())
        column_values = [filtered_data[col] for col in column_names]

        print(f"Generazione di tutte le combinazioni possibili...")
        all_combinations = []
        for combo in itertools.product(*column_values):
            combo_dict = dict(zip(column_names, combo))
            combo_key = tuple(combo_dict[col] for col in column_names)

            if combo_key not in self.combinations_used:
                all_combinations.append(combo_dict)
                self.combinations_used.add(combo_key)

        print(f"Totale combinazioni possibili: {len(all_combinations)}")

        # Applica FPS se richiesto
        if use_fps and all_combinations and max_combinations:
            diverse_combinations = self.get_most_dissimilar_combinations_fps(
                all_combinations, k=min(max_combinations, len(all_combinations))
            )
            return diverse_combinations
        else:
            # Nessun FPS: usa tutte le combinazioni o limita casualmente
            if max_combinations:
                random.shuffle(all_combinations)
                return all_combinations[:max_combinations]
            return all_combinations

    async def generate_scenario_with_ollama_async(self, session, combination):
        """
        Versione asincrona: genera uno scenario NIOSH usando Ollama con llama3.2

        Args:
            session: istanza di aiohttp.ClientSession condivisa
            combination: Dizionario con la combinazione di valori
        """

        payload = {
            "model": "niosh_scenario_model",  # il nome definito nel Modelfile
            "prompt": json.dumps(combination, ensure_ascii=False),
            "stream": False,
            "temperature": 0.7,
            "num_predict": 150,
        }

        try:
            # Timeout totale per la singola richiesta
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=timeout,
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    scenario = result.get("response", "").strip()
                    scenario = scenario.replace("\n", " ").strip()
                    return scenario
                else:
                    print(f"Errore Ollama (status={resp.status})")
                    return None

        except Exception as e:
            print(f"Errore nella generazione async: {e}")
            return None

    async def generate_and_save_examples_async(
        self,
        num_examples=None,
        selected_columns=None,
        use_fps=True,
        max_concurrent=8,
    ):
        """
        Versione asincrona:
        Genera e salva gli esempi nel file di testo usando asyncio + aiohttp.

        Args:
            num_examples: Numero di esempi da generare (None = tutti).
            selected_columns: Colonne da usare per le combinazioni.
            use_fps: Se True, usa Farthest Point Sampling per massima diversità.
            max_concurrent: Numero massimo di richieste contemporanee a Ollama.
        """
        print("=" * 70)
        print("GENERATORE DI SCENARI NIOSH CON FARTHEST POINT SAMPLING (ASYNC)")
        print("=" * 70)

        print("\nCaricamento dati dal CSV...")
        data = self.load_csv_data()

        print(f"Colonne disponibili: {list(data.keys())}")

        if use_fps:
            print(f"\n Modalità: FARTHEST POINT SAMPLING (FPS)")
            print("   → Massima diversità semantica garantita")
            print("   → Algoritmo: Greedy k-Center approximation")
        else:
            print(f"\n Modalità: SELEZIONE CASUALE")
            print("   → Nessun filtro di diversità applicato")

        combinations = self.generate_combinations(
            data, selected_columns, num_examples, use_fps
        )

        total = len(combinations)
        print(f"\n{'=' * 70}")
        print(f"Totale combinazioni da generare: {total}")
        print(f"Concurrency (max richieste simultanee): {max_concurrent}")
        print(f"{'=' * 70}\n")

        scenarios = [None] * total

        sem = asyncio.Semaphore(max_concurrent)

        async with aiohttp.ClientSession() as session:

            async def worker(idx, combo):
                async with sem:
                    scenario = await self.generate_scenario_with_ollama_async(
                        session, combo
                    )
                    return idx, scenario

            tasks = [worker(i, combo) for i, combo in enumerate(combinations)]

            # Progress bar singola, pulita, compatibile
            for coro in tqdm(asyncio.as_completed(tasks), total=total, desc="Generazione scenari"):
                idx, scenario = await coro
                scenarios[idx] = scenario

        # Scrittura su file in ordine deterministico
        with open(self.output_file, "w", encoding="utf-8") as f:
            valid_count = 0
            for i, scenario in enumerate(scenarios, 1):
                if scenario:
                    f.write(scenario + "\n")
                    valid_count += 1
                else:
                    print(f"[{i}/{total}] ATTENZIONE: scenario mancante, salto la riga")

        print(f"{'=' * 70}")
        print(f" COMPLETATO (ASYNC)!")
        print(
            f"   {valid_count}/{total} scenari validi salvati in '{self.output_file}'"
        )
        print(f"{'=' * 70}")

    def generate_scenario_with_ollama(self, combination):
        """
        Genera uno scenario NIOSH usando Ollama con llama3.2

        Args:
            combination: Dizionario con la combinazione di valori
        """
        # Crea il prompt per generare lo scenario
        prompt = f"""You are a deterministic generator of ergonomic lifting scenarios.

Your task is to produce ONE sentence for each input combination:
Generate one sentence using: {json.dumps(combination, ensure_ascii=False)}

You MUST select RANDOMLY one of the three scenario types:
• SINGLE TASK
• REPETITIVE SINGLE TASK
• MULTI-TASK SEQUENCE

The selection of the type MUST be random and MUST NOT depend on the content of the input combination.

----------------------------------------------------------
SENTENCE RULES (MANDATORY)
----------------------------------------------------------
• Output ONLY the sentence. No comments, no explanations, no formatting.
• Use clear, natural English.
• Maximum length: 330 characters.
• Follow the structure: subject + action + location + object.
• Do NOT include any numbers, dimensions, durations, or weights.
• Use generic wording for objects (e.g., “boxes”, “racks”, “containers”).
• No commas unless needed to join actions in a multi-task sentence.

----------------------------------------------------------
SCENARIO TYPES (YOU MUST RANDOMLY PICK ONE)
----------------------------------------------------------

1. SINGLE TASK
   - Exactly one action in one place on one object.
   - Example pattern:
     “The worker lifts boxes from a pallet in the loading area.”

2. REPETITIVE SINGLE TASK
   - One action repeated.
   - Use words such as:
     “repeatedly”, “continuously”, “throughout the shift”.
   - Example pattern:
     “The worker repeatedly lifts trays at the packing station.”

3. MULTI-TASK SEQUENCE
   - More than one distinct action joined in a single sentence.
   - Use connectors:
     “then”, “and next”, “followed by”.
   - Example pattern:
     “The worker lifts bins from the floor then places crates on a shelf.”

----------------------------------------------------------
RULE ENFORCEMENT
----------------------------------------------------------
• Randomly choose the scenario type BEFORE generating the sentence.
• Never invent numbers, weights, measurements, or durations.
• Never break the single-sentence rule.
• Never exceed 330 characters.
• Output ONLY the sentence.

"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "niosh_scenario_model",  # il nome definito nel Modelfile
                    "prompt": json.dumps(combination, ensure_ascii=False),
                    "stream": False,
                    "temperature": 0.7,
                    "num_predict": 60,
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                scenario = result.get("response", "").strip()
                # Pulisci la risposta da eventuali caratteri extra
                scenario = scenario.replace("\n", " ").strip()
                return scenario
            else:
                print(f"Errore Ollama: {response.status_code}")
                return None

        except Exception as e:
            print(f"Errore nella generazione: {e}")
            return None

    def generate_and_save_examples(
        self,
        num_examples=None,
        selected_columns=None,
        use_fps=True,
        num_workers=8,
    ):
        """
        Genera e salva gli esempi nel file di testo usando una pipeline multithreading.

        Args:
            num_examples: Numero di esempi da generare (None = tutti).
            selected_columns: Colonne da usare per le combinazioni.
            use_fps: Se True, usa Farthest Point Sampling per massima diversità.
            num_workers: Numero di thread per la generazione parallela.
        """
        print("=" * 70)
        print("GENERATORE DI SCENARI NIOSH CON FARTHEST POINT SAMPLING (MULTITHREAD)")
        print("=" * 70)

        print("\nCaricamento dati dal CSV...")
        data = self.load_csv_data()

        print(f"Colonne disponibili: {list(data.keys())}")

        if use_fps:
            print(f"\n Modalità: FARTHEST POINT SAMPLING (FPS)")
            print("   → Massima diversità semantica garantita")
            print("   → Algoritmo: Greedy k-Center approximation")
        else:
            print(f"\n Modalità: SELEZIONE CASUALE")
            print("   → Nessun filtro di diversità applicato")

        combinations = self.generate_combinations(
            data, selected_columns, num_examples, use_fps
        )

        total = len(combinations)
        print(f"\n{'=' * 70}")
        print(f"Totale combinazioni da generare: {total}")
        print(f"Thread di lavoro: {num_workers}")
        print(f"{'=' * 70}\n")

        # Array per mantenere l'output in ordine
        scenarios = [None] * total

        def worker(index_combo):
            """Funzione eseguita da ciascun thread."""
            idx, combo = index_combo
            print(f"[THREAD] Generazione scenario per indice {idx + 1}/{total}")
            scenario = self.generate_scenario_with_ollama(combo)
            return idx, scenario

        # Multithreading per chiamare Ollama in parallelo
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker, (i, combo)): i
                for i, combo in enumerate(combinations)
            }

            for future in as_completed(futures):
                idx, scenario = future.result()
                if scenario:
                    scenarios[idx] = scenario
                    print(f"[{idx + 1}/{total}] ✓ Scenario generato")
                else:
                    print(f"[{idx + 1}/{total}] ✗ Errore nella generazione")

        # Scrittura su file in ordine deterministico
        with open(self.output_file, "w", encoding="utf-8") as f:
            valid_count = 0
            for i, scenario in enumerate(scenarios, 1):
                if scenario:
                    f.write(scenario + "\n")
                    valid_count += 1
                else:
                    print(f"[{i}/{total}] ATTENZIONE: scenario mancante, salto la riga")

        print(f"{'=' * 70}")
        print(f"✅ COMPLETATO!")
        print(
            f"   {valid_count}/{total} scenari validi salvati in '{self.output_file}'"
        )
        print(f"{'=' * 70}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    CSV_FILE = base_dir / "dati_niosh.csv"
    OUTPUT_FILE = base_dir / "scenari_niosh_fps.txt"

    # === CHIEDI IN CONSOLE QUANTE COMBINAZIONI GENERARE ===
    while True:
        try:
            user_input = input("Quante combinazioni vuoi generare (es. 100)? ").strip()
            max_combo = int(user_input)
            if max_combo <= 0:
                print("Inserisci un numero maggiore di zero.")
                continue
            break
        except ValueError:
            print("Per favore inserisci un numero valido.")

    generator = NIOSHExampleGenerator(
        csv_file=str(CSV_FILE),
        output_file=str(OUTPUT_FILE)
    )

    # VERSIONE ASINCRONA con FPS
    asyncio.run(
        generator.generate_and_save_examples_async(
            num_examples=max_combo,                
            selected_columns=["subject", "action", "place", "object"],
            use_fps=True,
            max_concurrent=8
        )
    )


    # Se vuoi tenere anche la versione sync come alternativa:
    # generator.generate_and_save_examples(
    #     num_examples=None,
    #     selected_columns=["subject", "action", "place", "object"],
    #     use_fps=True,
    #     num_workers=8,
    # )
