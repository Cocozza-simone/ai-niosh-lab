import os
import json
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe


try:
    import orjson

    def fast_load(fp):
        return orjson.loads(fp.read())

except ImportError:

    def fast_load(fp):
        return json.load(fp)


# ------------------------------------------------------------
# PARALLEL JSON LOADER
# ------------------------------------------------------------

def parse_file(path):
    try:
        with open(path, "rb") as f:
            data = fast_load(f)
        return path, data
    except Exception as e:
        return path, {"_error": str(e)}

# ------------------------------------------------------------
# FALLBACK DETECTOR (Gemini Judge)
# ------------------------------------------------------------

def is_fallback_eval(eval_block: dict) -> bool:
    """
    Ritorna True se il blocco proviene dal fallback del Gemini Judge.
    Il fallback è chiaramente identificabile da:
        winner = "B"
        weighted_scores = { "A": 0, "B": 10 }
    """
    if not isinstance(eval_block, dict):
        return False

    winner = eval_block.get("winner")
    if winner != "B":
        return False

    ws = eval_block.get("weighted_scores", {})
    if not isinstance(ws, dict):
        return False

    if ws.get("A") == 0 and ws.get("B") == 10:
        return True

    return False


def detect_fallback_in_report(report_json: dict) -> bool:
    """
    Controlla tutte le sezioni valutate (specific + combined)
    per capire se QUALSIASI parte del file è andata in fallback.
    """
    ev = report_json.get("evaluation", {})
    if not isinstance(ev, dict):
        return False

    KEYS = [
        "jd_specific_eval", "jd_combined_eval",
        "ja_specific_eval", "ja_combined_eval",
        "ha_specific_eval", "ha_combined_eval",
        "rs_specific_eval", "rs_combined_eval",
    ]

    for key in KEYS:
        block = ev.get(key)
        if block and is_fallback_eval(block):
            return True

    return False

# ------------------------------------------------------------
# REGEX COMPILATE PER DETECTION MODELLI
# ------------------------------------------------------------

MODEL_PATTERNS = [
    re.compile(r"(llama[\w:\-]*)"),
    re.compile(r"(gemma[\w:\-]*)"),
    re.compile(r"(gemini[\w:\-]*)"),
    re.compile(r"(qwen[\w:\-]*)"),
    re.compile(r"(mistral[\w:\-]*)"),
    re.compile(r"(phi[\w:\-]*)"),
    re.compile(r"(gpt[\w:\-]*)"),
]

SECTIONS = [
    "job_description",
    "job_analysis",
    "hazard_assessment",
    "redesign_suggestions",
]


# ------------------------------------------------------------
# DYNAMIC DETECTION
# ------------------------------------------------------------

def detect_model_and_task(path: Path):
    parts = [p.lower() for p in path.parts]

    # MODELLO
    model_name = "unknown"
    for part in parts:
        for patt in MODEL_PATTERNS:
            m = patt.search(part)
            if m:
                model_name = m.group(1)
                break
        if model_name != "unknown":
            break

    # TASK
    TASK_CANDIDATES = {"single", "multi", "repetitive"}

    task_type = next((p for p in parts if p in TASK_CANDIDATES), "unknown")

    # fallback: cartella accanto al modello
    if task_type == "unknown" and model_name != "unknown":
        try:
            idx = parts.index(model_name)
            if idx + 1 < len(parts):
                cand = parts[idx + 1]
                if "." not in cand:
                    task_type = cand
        except ValueError:
            pass

    return model_name, task_type


# ------------------------------------------------------------
# LOAD JSONs
# ------------------------------------------------------------

def load_all_json(base_dir, mode="specific"):
    """
    mode:
      - specific → usa solo *_specific_eval
      - combined → usa solo *_combined_eval
      - both → crea media specific+combined
    """

    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Directory '{base_dir}' non trovata.")

    json_files = list(base.rglob("*.json"))
    if not json_files:
        return []

    data = []

    max_workers = min(32, (os.cpu_count() or 8) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path, obj in executor.map(parse_file, json_files):

            if "_error" in obj:
                print(f"[ERROR] {path}: {obj['_error']}")
                continue

            model_name, task_type = detect_model_and_task(Path(path))

            entry = {
                "file": str(path),
                "model_name": model_name,
                "task_type": task_type,
                "input": obj.get("input", ""),
                "timestamp": obj.get("timestamp", ""),
                "fallback": detect_fallback_in_report(obj),
            }

            ev = obj.get("evaluation", {})

            def extract(block):
                """
                Ritorna:
                - AI weighted score B
                - raw dimension scores B
                """
                if not isinstance(block, dict):
                    return None, None

                weighted = block.get("weighted_scores", {})
                scores = block.get("scores", {})

                ai_score = weighted.get("B")
                dim_scores = scores.get("B") if isinstance(scores, dict) else None

                return ai_score, dim_scores

            SECTION_MAP = {
                "job_description": ("jd_combined_eval", "jd_specific_eval"),
                "job_analysis": ("ja_combined_eval", "ja_specific_eval"),
                "hazard_assessment": ("ha_combined_eval", "ha_specific_eval"),
                "redesign_suggestions": ("rs_combined_eval", "rs_specific_eval"),
            }

            for section, (combined_key, specific_key) in SECTION_MAP.items():

                block_combined = ev.get(combined_key)
                block_specific = ev.get(specific_key)

                score_combined, scores_dim_combined = extract(block_combined)
                score_specific, scores_dim_specific = extract(block_specific)

                # =============== MODALITÀ ===============
                if mode == "specific":
                    final_score = score_specific
                    final_dim_scores = scores_dim_specific

                elif mode == "combined":
                    final_score = score_combined
                    final_dim_scores = scores_dim_combined

                elif mode == "both":
                    # media delle due se entrambe presenti
                    if score_combined is not None and score_specific is not None:
                        final_score = (score_combined + score_specific) / 2
                    else:
                        final_score = score_combined or score_specific

                    # media dimensionale
                    if isinstance(scores_dim_combined, dict) and isinstance(scores_dim_specific, dict):
                        final_dim_scores = {
                            k: (scores_dim_combined.get(k, 0) + scores_dim_specific.get(k, 0)) / 2
                            for k in ["structure", "coherence", "style", "semantic_alignment"]
                        }
                    else:
                        final_dim_scores = scores_dim_combined or scores_dim_specific

                entry[f"{section}_score_AI"] = final_score
                entry[f"{section}_scores"] = final_dim_scores
                entry["section_name"] = section

            data.append(entry)

    return data

# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

def mean_std(series):
    s = series.dropna()
    if s.empty:
        return None, None
    return round(s.mean(), 3), (round(s.std(), 3) if len(s) > 1 else 0.0)


def compute_global_stats(df):
    stats = {}
    AI_COLS = [f"{sec}_score_AI" for sec in SECTIONS]

    for sec in SECTIONS:
        col = f"{sec}_score_AI"
        mean, std = mean_std(df[col])
        stats[sec] = {"mean": mean, "std": std, "count": df[col].dropna().shape[0]}

    df["total_AI"] = df[AI_COLS].mean(axis=1)
    mean, std = mean_std(df["total_AI"])

    stats["overall"] = {
        "global_mean": mean,
        "global_std": std,
        "total_examples": df["total_AI"].dropna().shape[0],
    }

    return stats


def compute_group_stats(df, group_cols):
    df = df.copy()
    AI_COLS = [f"{sec}_score_AI" for sec in SECTIONS]
    df["total"] = df[AI_COLS].mean(axis=1)

    rows = []

    for name, group in df.groupby(group_cols):
        row = {}

        if isinstance(name, tuple):
            for i, col in enumerate(group_cols):
                row[col] = name[i]
        else:
            row[group_cols[0]] = name

        row["count"] = len(group)

        for sec in SECTIONS:
            col = f"{sec}_score_AI"
            mean, std = mean_std(group[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std

        mean, std = mean_std(group["total"])
        row["total_mean"] = mean
        row["total_std"] = std

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

def save_outputs(df, global_stats, stats_model, stats_task, stats_both):
    df.to_csv("evaluation_summary.csv", index=False)

    pd.DataFrame(
        [
            {"section": sec, **vals}
            for sec, vals in global_stats.items()
            if sec != "overall"
        ]
    ).to_csv("evaluation_by_section.csv", index=False)

    stats_model.to_csv("comparison_by_model.csv", index=False)
    stats_task.to_csv("comparison_by_task_type.csv", index=False)
    stats_both.to_csv("comparison_model_tasktype.csv", index=False)

    with open("evaluation_stats.json", "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2)

    print("\nOutput generati:")
    print(" - evaluation_summary.csv")
    print(" - evaluation_by_section.csv")
    print(" - evaluation_stats.json")
    print(" - comparison_by_model.csv")
    print(" - comparison_by_task_type.csv")
    print(" - comparison_model_tasktype.csv")


def plot_section_performance_overview(df, output_path):

    sections = SECTIONS
    means = []
    stds = []

    for sec in sections:
        col = f"{sec}_score_AI"
        vals = df[col].dropna()

        means.append(vals.mean() if len(vals) else 0)
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0)

    sections_clean = [s.replace("_", " ").title() for s in sections]

    fig, ax = plt.subplots(figsize=(16, 7), dpi=320)
    fig.patch.set_facecolor("white")

    y = np.arange(len(sections))

    # BARRE CON ERRORI BEN VISIBILI
    bars = ax.barh(
        y,
        means,
        xerr=stds,
        color="#4AA36A",
        edgecolor="black",
        linewidth=2,
        capsize=16,
        error_kw=dict(
            lw=3,
            capthick=3,
            ecolor="black"
        )
    )

    # TICKS
    ax.set_yticks(y)
    ax.set_yticklabels(sections_clean, fontsize=16)

    ax.set_xlim(0, 10)
    ax.set_xlabel("Mean Score", fontsize=18)
    ax.set_title(
        "Section Performance Overview (Mean ± Std)",
        fontsize=22,
        fontweight="bold"
    )

    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # ANNOTAZIONI NUMERICHE (μ, σ)
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            mean + 0.12,
            i,
            f"μ={mean:.2f}  σ={std:.2f}",
            va="center",
            fontsize=14,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=320)
    plt.close()

def plot_dimension_scores_heatmap(df, output_path):

    dim_names = ["structure", "coherence", "style", "semantic_alignment"]
    sections = SECTIONS

    matrix = []

    # Calcolo della matrice sezione × dimensione
    for sec in sections:
        col = f"{sec}_scores"
        rows = df[col]

        row_vals = []
        for dim in dim_names:
            values = [
                s[dim] for s in rows
                if isinstance(s, dict) and dim in s
            ]
            row_vals.append(np.mean(values) if values else 0)

        matrix.append(row_vals)

    matrix = np.array(matrix)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)

    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=10)

    # Assi
    ax.set_xticks(range(len(dim_names)))
    ax.set_yticks(range(len(sections)))

    ax.set_xticklabels(dim_names, fontsize=14, rotation=45)
    ax.set_yticklabels([s.replace("_", " ").title() for s in sections],
                       fontsize=16)

    # Testo numerico nero + BOLD
    for i in range(len(sections)):
        for j in range(len(dim_names)):
            ax.text(
                j, i, f"{matrix[i, j]:.1f}",
                ha="center", va="center",
                color="black",
                fontsize=12,
                fontweight="bold"  
            )

    plt.colorbar(im)
    ax.set_title("Dimension Scores Heatmap", fontsize=22, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
def print_full_console_report(global_stats, df, stats_model, stats_task, stats_both):

    print("\n==============================")
    print("   GEMINI JUDGE — REPORT")
    print("==============================\n")

    for sec in SECTIONS:
        vals = global_stats[sec]
        print(f">> {sec.upper()}")
        print(f"   media:  {vals['mean']}")
        print(f"   std:    {vals['std']}")
        print(f"   count:  {vals['count']}\n")

    ov = global_stats["overall"]
    print("OVERALL")
    print(f"   Media globale: {ov['global_mean']}")
    print(f"   Std globale:   {ov['global_std']}")
    print(f"   Esempi totali: {ov['total_examples']}\n")

    print("\n==============================")
    print("     CONFRONTO MODELLI")
    print("==============================")

    print("\nPER MODELLO")
    for _, row in stats_model.iterrows():
        print(f"\n>> Modello: {row['model_name']}")
        print(f"   Esempi: {row['count']}")
        print(f"   Media totale: {row['total_mean']}")
        print(f"   Std: {row['total_std']}")

    print("\nPER TIPO DI TASK")
    for _, row in stats_task.iterrows():
        print(f"\n>> Task: {row['task_type']}")
        print(f"   Esempi: {row['count']}")
        print(f"   Media totale: {row['total_mean']}")
        print(f"   Std: {row['total_std']}")

    print("\nMODELLO × TASK TYPE")
    for _, row in stats_both.iterrows():
        print(f"\n>> Modello: {row['model_name']} | Task: {row['task_type']}")
        print(f"   Esempi: {row['count']}")
        print(f"   Media totale: {row['total_mean']}")
        print(f"   Std: {row['total_std']}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIOSH Aggregator ")
    parser.add_argument(
        "--dir",
        type=str,
        default="batch_reports",
        help="Directory dei JSON da analizzare",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["specific", "combined", "both"],
        help="Modalità di aggregazione",
    )
    args = parser.parse_args()

    print(f"Caricamento JSON da '{args.dir}'...")

    records = load_all_json(args.dir, mode=args.mode)
    if not records:
        print("Nessun JSON trovato.")
        return

    df = pd.DataFrame(records)

    # Filtro dei record validi: almeno UNO score presente
    AI_COLS = [f"{sec}_score_AI" for sec in SECTIONS]
    valid_mask = df[AI_COLS].notna().any(axis=1)
    df_valid = df[valid_mask].copy()

    df_valid = df_valid[df_valid["model_name"].notna()]
    df_valid = df_valid[df_valid["task_type"].notna()]
    
    total = len(df_valid)
    fallback_count = df_valid[df_valid["fallback"]].shape[0]
    perc = (fallback_count / total) * 100 if total > 0 else 0
    # --- ASSERTION DI SICUREZZA ---
    MAX_FALLBACK_PERC = 5.0  # soglia massima accettabile

    if perc > MAX_FALLBACK_PERC:
        raise RuntimeError(
            f"Fallback troppo alti: {perc:.2f}% (soglia {MAX_FALLBACK_PERC}%). "
            "Il judge non sta producendo JSON valido. Interruzione analisi."
        )
    else:
        print(f"Caricati {len(df)} file.")
        print(f"Esempi validi: {len(df_valid)}")

        global_stats = compute_global_stats(df_valid)

        stats_model = compute_group_stats(df_valid, ["model_name"])
        stats_task = compute_group_stats(df_valid, ["task_type"])
        stats_both = compute_group_stats(df_valid, ["model_name", "task_type"])

        print_full_console_report(global_stats, df_valid, stats_model, stats_task, stats_both)
        # ------------------------------------------------------------
        # CARTELLA PLOTS
        # ------------------------------------------------------------
        PLOTS_DIR = Path("plots")
        PLOTS_DIR.mkdir(exist_ok=True)
        save_outputs(df_valid, global_stats, stats_model, stats_task, stats_both)

        # Generazione grafico delle sezioni
        output_plot = Path("section_performance_overview.png")
        # ==============================
        # GENERAZIONE GRAFICI
        # ==============================

        plot_section_performance_overview(
            df_valid, 
            PLOTS_DIR / "section_performance_overview.png"
        )
        print("Salvato grafico delle sezioni")
        
        plot_dimension_scores_heatmap(
            df_valid, 
            PLOTS_DIR / "heatmap.png"
        )
        print("Salvato heatmap final scores")
        
        print("\nGrafici salvati nella cartella 'plots/'")


    
if __name__ == "__main__":
    main()
