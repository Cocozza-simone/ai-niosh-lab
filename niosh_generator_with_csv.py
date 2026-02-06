import csv
import itertools
import json
import random
import asyncio
import aiohttp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


class NIOSHExampleGenerator:
    def __init__(
        self,
        csv_file,
        output_file="scenari_niosh_fps3.txt",
        ollama_url="http://localhost:11434",
    ):
        self.csv_file = csv_file
        self.output_file = output_file
        self.ollama_url = ollama_url
        self.vectorizer = TfidfVectorizer()
        self.combinations_used = set()

    # =====================================================
    # CARICAMENTO CSV
    # =====================================================
    def load_csv_data(self):
        data = {}
        with open(self.csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col, value in row.items():
                    if value:
                        data.setdefault(col, []).append(value.strip())

        # dedup
        for col in data:
            unique = []
            seen = set()
            for v in data[col]:
                if v not in seen:
                    unique.append(v)
                    seen.add(v)
            data[col] = unique
        return data

    # =====================================================
    # COMBINAZIONI + FARTEST POINT SAMPLING
    # =====================================================
    def combination_to_text(self, combination):
        return " ".join(str(v) for v in combination.values())
    def get_most_dissimilar_combinations_fps(self, all_combinations, k, semantic_threshold=0.0):
        """
        Seleziona fino a k combinazioni più dissimili usando Farthest Point Sampling,
        rispettando la soglia minima di dissimilarità (cosine distance).
        """
        if k >= len(all_combinations):
            return all_combinations

        texts = [self.combination_to_text(c) for c in all_combinations]
        embeddings = self.vectorizer.fit_transform(texts).toarray()
        N = embeddings.shape[0]

        selected = [np.random.randint(0, N)]
        min_dists = np.full(N, np.inf)

        for _ in range(k - 1):
            last = embeddings[selected[-1]].reshape(1, -1)
            dists = cosine_distances(embeddings, last).flatten()
            min_dists = np.minimum(min_dists, dists)

            # Trova indici candidati che rispettano la soglia semantica
            candidates = [i for i in range(N) if i not in selected and min_dists[i] >= semantic_threshold]
            if not candidates:
                break  # non ci sono più combinazioni sufficientemente diverse

            # Scegli il più lontano tra i candidati
            next_idx = max(candidates, key=lambda i: min_dists[i])
            selected.append(next_idx)

        return [all_combinations[i] for i in selected]

    """ 
    def get_most_dissimilar_combinations_fps(self, all_combinations, k):
        if k >= len(all_combinations):
            return all_combinations

        texts = [self.combination_to_text(c) for c in all_combinations]
        embeddings = self.vectorizer.fit_transform(texts).toarray()
        N = embeddings.shape[0]

        selected = [np.random.randint(0, N)]
        min_dists = np.full(N, np.inf)

        for _ in range(k - 1):
            last = embeddings[selected[-1]].reshape(1, -1)
            dists = cosine_distances(embeddings, last).flatten()
            min_dists = np.minimum(min_dists, dists)
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)

        return [all_combinations[i] for i in selected]

    # =====================================================
    # GENERAZIONE COMBINAZIONI solo top y
    # =====================================================
        def generate_combinations(
        self, data, selected_columns=None, max_combinations=None, use_fps=True
    ):
        if selected_columns is None:
            selected_columns = list(data.keys())

        filtered = {c: data[c] for c in selected_columns if c in data}
        names = list(filtered.keys())
        values = [filtered[c] for c in names]

        all_combinations = []
        for combo in itertools.product(*values):
            combo_dict = dict(zip(names, combo))
            key = tuple(combo_dict[c] for c in names)
            if key not in self.combinations_used:
                self.combinations_used.add(key)
                all_combinations.append(combo_dict)

        if use_fps and max_combinations:
            return self.get_most_dissimilar_combinations_fps(
                all_combinations, max_combinations
            )

        if max_combinations:
            random.shuffle(all_combinations)
            return all_combinations[:max_combinations]

        return all_combinations
"""
    # =====================================================
    # GENERAZIONE COMBINAZIONI con top y semantic threshold
    # =====================================================
    def generate_combinations(
        self,
        data,
        selected_columns=None,
        max_combinations=None,
        use_fps=True,
        semantic_threshold=0.3,  # soglia cosine similarity minima
    ):
        if selected_columns is None:
            selected_columns = list(data.keys())

        filtered = {c: data[c] for c in selected_columns if c in data}
        names = list(filtered.keys())
        values = [filtered[c] for c in names]

        # ===============================
        # CREAZIONE TUTTE LE COMBINAZIONI
        # ===============================
        all_combinations = []
        for combo in itertools.product(*values):
            combo_dict = dict(zip(names, combo))
            key = tuple(combo_dict[c] for c in names)
            if key not in self.combinations_used:
                self.combinations_used.add(key)
                all_combinations.append(combo_dict)

        if not all_combinations:
            return []

        # ===============================
        # FILTRO SEMANTICO
        # ===============================
        if semantic_threshold > 0:
            texts = [self.combination_to_text(c) for c in all_combinations]
            embeddings = self.vectorizer.fit_transform(texts).toarray()
            filtered_combinations = []
            kept_indices = []

            for i, emb_i in enumerate(embeddings):
                if all(
                    cosine_distances(emb_i.reshape(1, -1), embeddings[j].reshape(1, -1))[0][0] 
                    >= semantic_threshold 
                    for j in kept_indices
                ):
                    kept_indices.append(i)
                    filtered_combinations.append(all_combinations[i])

            all_combinations = filtered_combinations

        # ===============================
        # FARTEST POINT SAMPLING
        # ===============================
        if use_fps and max_combinations:
            return self.get_most_dissimilar_combinations_fps(
                all_combinations,
                max_combinations,
                semantic_threshold=semantic_threshold  # passiamo la soglia
            )

        # ===============================
        # RANDOM TRUNCATION
        # ===============================
        if max_combinations:
            random.shuffle(all_combinations)
            return all_combinations[:max_combinations]

        return all_combinations

    # =====================================================
    # GENERATORE TIPIZZATO DELLO SCENARIO
    # =====================================================
    async def generate_specific_type_async(self, session, combination, scenario_type):

        # ============================================================
        # PROMPT MIGLIORATO PER GENERARE SCENARI REALISTICI
        # ============================================================

        instructions = {
            "single": (
                "Generate a SINGLE-TASK manual handling scenario. "
                "The scenario must describe ONE clear physical action involving the subject, the action, the place, "
                "and the object. No repetition, no multi-step sequences, no connectors such as 'then', 'next', "
                "'afterwards', or 'followed by'. "
                "Do NOT add tools, equipment, or new objects. "
                "Use simple warehouse/industrial natural language. "
                "Write ONE coherent sentence only."
            ),
            "repetitive": (
                "Generate a REPETITIVE-TASK manual handling scenario. "
                "The subject must perform the action REPEATEDLY throughout the task. "
                "Use repetition markers such as 'continuously', 'repeatedly', 'throughout the shift', "
                "'for most of the shift'. "
                "The sentence must NOT include step sequences (no 'then', 'next', 'followed by'). "
                "Use simple warehouse/industrial natural language. "
                "Write ONE coherent sentence only."
            ),
            "multi": (
                "Generate a MULTI-TASK SEQUENCE manual handling scenario. "
                "The scenario must describe a sequence of AT LEAST THREE distinct actions performed by the subject. "
                "Use connectors such as 'then', 'followed by', 'and next', 'and finally'. "
                "The narrative must be ONE SINGLE long sentence describing a chained multi-step sequence. "
                "Do NOT use repetition markers such as 'continuously' or 'repeatedly'. "
                "Write ONE coherent multi-step industrial scenario."
            ),
        }

        # ============================================================
        # COSTRUZIONE PROMPT
        # ============================================================

        prompt = (
            f"{instructions[scenario_type]}\n\n"
            f"YOU MUST USE THIS EXACT SUBJECT (do NOT change it): {combination['subject']}\n"
            f"Subject is mandatory and must appear exactly as provided.\n\n"
            f"Action: {combination['action']}\n"
            f"Place: {combination['place']}\n"
            f"Object: {combination['object']}\n\n"
            f"Requirements:\n"
            f"- Maintain industrial/warehouse tone.\n"
            f"- No invented tools or extra details.\n"
            f"- No changing the subject, action, place, or object.\n"
            f"- Output ONE single sentence.\n\n"
            f"Write one coherent scenario in natural language."
        )

        payload = {
            "model": "llama3.2",  # o qualunque modello tu abbia installato in Ollama
            "prompt": prompt,
            "temperature": 0.7,
            "num_predict": 300,
            "stream": False,
        }

        # ============================================================
        # CHIAMATA AL MODELLO + PULIZIA OUTPUT
        # ============================================================

        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with session.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=timeout
            ) as resp:

                if resp.status == 200:
                    result = await resp.json()
                    scen = result.get("response", "").strip()

                    # normalizza e ritorna in una riga
                    scen = scen.replace("\n", " ").strip()
                    scen = " ".join(scen.split())  # rimuove doppi spazi

                    return scen

                return None

        except Exception:
            return None

    # =====================================================
    # GENERATORE DI BLOCCHI (GARANZIA TARGET)
    # =====================================================
    async def generate_block(
        self, combinations, scenario_type, target, max_concurrent=16
    ):
        results = []
        sem = asyncio.Semaphore(max_concurrent)
        random.shuffle(combinations)

        async with aiohttp.ClientSession() as session:
            idx = 0
            pbar = tqdm(total=target, desc=f"{scenario_type.upper()}")

            async def worker(combo):
                async with sem:
                    return await self.generate_specific_type_async(
                        session, combo, scenario_type
                    )

            while len(results) < target:
                combo = combinations[idx % len(combinations)]
                idx += 1

                scen = await worker(combo)
                if scen:
                    results.append(scen)
                    pbar.update(1)

            pbar.close()
            return results

    # =====================================================
    # GENERAZIONE BILANCIATA
    # =====================================================
    async def generate_balanced_fast(
        self, combinations, target_each, max_concurrent=16
    ):

        singles = await self.generate_block(
            combinations, "single", target_each, max_concurrent
        )
        repetitives = await self.generate_block(
            combinations, "repetitive", target_each, max_concurrent
        )
        multis = await self.generate_block(
            combinations, "multi", target_each, max_concurrent
        )

        return singles + repetitives + multis


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    base = Path(__file__).parent
    CSV_FILE = base / "dati_niosh.csv"
    OUTPUT_FILE = base / "scenari_niosh_fps3.txt"
    
    gen = NIOSHExampleGenerator(
        csv_file=str(CSV_FILE),
        output_file=str(OUTPUT_FILE),
    )

    # ===============================
    # CARICAMENTO DATI
    # ===============================
    data = gen.load_csv_data()

    # ===============================
    # generate_combinations con  top y
    # ===============================
    """ combinations = gen.generate_combinations(
        data,
        selected_columns=["subject", "action", "place", "object"],
        max_combinations=None,
        use_fps=True,
     
    ) """

    # ===============================
    # generate_combinations con semantic threshold e top y
    # ===============================
    combinations = gen.generate_combinations(
        data,
        selected_columns=["subject", "action", "place", "object"],
        max_combinations=None,
        use_fps=True,
        semantic_threshold=0.30       # threshold semantico
    )
    # ===============================
    # ANALISI DIVERSITÀ SEMANTICA
    # ===============================
    texts = [gen.combination_to_text(c) for c in combinations]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # matrice distanze coseno
    dist_matrix = cosine_distances(X)

    threshold = 0.30  # soglia di diversità semantica
    unique_indices = []

    for i in range(len(dist_matrix)):
        if not any(dist_matrix[i][j] < threshold for j in unique_indices):
            unique_indices.append(i)

    print("\n🧠 ANALISI SEMANTICA")
    print("Combinazioni totali:", len(combinations))
    print("Combinazioni realmente diverse (threshold =", threshold, "):", len(unique_indices))
    print("Riduzione effettiva:", 
      round(100 * (1 - len(unique_indices) / len(combinations)), 2), "%")

    # ===============================
    # LOG NUMERICO CHIARO
    # ===============================
    totale_teorico = (
        len(data["subject"]) *
        len(data["action"]) *
        len(data["place"]) *
        len(data["object"])
    )

    print("\n📊 STATISTICHE COMBINAZIONI")
    print("Totale combinazioni teoriche:", totale_teorico)
    print("Combinazioni generate:", len(combinations))

    # ===============================
    # INPUT UTENTE
    # ===============================
    target_each = int(
        input("\nQuanti scenari per categoria (single/repetitive/multi)? > ")
    )

    totale_scenari = 3 * target_each

    print("\n📈 STATISTICHE OUTPUT")
    print("Scenari finali generati:", totale_scenari)

    # ===============================
    # BLOCCO DURO: NO RIUSO COMBINAZIONI
    # ===============================
    if target_each > len(combinations):
        raise ValueError(
            f"\n❌ ERRORE DI CONFIGURAZIONE\n"
            f"- Combinazioni uniche disponibili: {len(combinations)}\n"
            f"- target_each richiesto: {target_each}\n\n"
            f"Ogni combinazione può essere usata UNA SOLA VOLTA per categoria.\n"
            f"Riduci target_each a ≤ {len(combinations)} "
            f"oppure aumenta le combinazioni nel CSV."
        )

    # ===============================
    # GENERAZIONE SCENARI
    # ===============================
    scenarios = asyncio.run(
        gen.generate_balanced_fast(
            combinations=combinations,
            target_each=target_each,
            max_concurrent=24,
        )
    )

    # ===============================
    # SALVATAGGIO FILE
    # ===============================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(s + "\n")

    print("\n✓ COMPLETATO! File salvato in:", OUTPUT_FILE)
