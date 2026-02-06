"""
analyzer.py - Strumento di Business Intelligence per Report NIOSH AI

Scopo:
1. VALIDAZIONE SCIENTIFICA: Genera metriche e grafici per dimostrare l'accuratezza.
2. DEBUGGING MIRATO: Isola i casi fallimentari e cerca correlazioni di errore.
3. ANALISI AVANZATE:
   - Bland-Altman plot per accordo Human vs AI (LI)
   - Intraclass Correlation Coefficient (ICC) per la concordanza
   - Regressione lineare per modellare l'errore (AI - Human)
   - Distinzione per modello (colonna "Model") in metriche e grafici

Uso:
  python analyzer.py out_eval.jsonl
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data(filepath):
    """Carica il file JSONL e appiattisce la struttura per l'analisi."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            # Estrazione sicura dei dati annidati
            try:
                # Dati Base
                idx = record.get("example_index")
                model_name = record.get("model", "unknown")

                # Metriche Numeriche (Risk - LI)
                risk_data = record["numeric_distance"]["risk"].get("max_li", {})
                h_li = risk_data.get("human")
                a_li = risk_data.get("ai")
                li_diff_abs = risk_data.get("abs_diff")

                # Errore con segno (AI - Human)
                li_diff_signed = None
                if h_li is not None and a_li is not None:
                    try:
                        li_diff_signed = float(a_li) - float(h_li)
                    except Exception:
                        li_diff_signed = None

                # Parametri di Input (per cercare pattern di errore)
                param_data = record["numeric_distance"]["parameters"]
                weight = param_data.get("weight", {}).get("human")
                h_dist = param_data.get("horizontal_origin", {}).get("human")

                # Valutazione Qualitativa (Gemini)
                gemini = record.get("gemini_eval", {})
                quality_score = gemini.get("overall_quality_score")
                text_score = gemini.get("textual_alignment_score")
                sem = record.get("semantic_validation", {})
                bleu = sem.get("BLEU")
                sbert_sim = sem.get("SBERT_similarity")
                comet_like = sem.get("COMET_like")
                entail = sem.get("NLI_entailment")
                data.append(
                    {
                     "index": idx,
                     "Model": model_name,
                     "Human_LI": h_li,
                     "AI_LI": a_li,
                     "LI_Error_Abs": li_diff_abs,
                     "LI_Error_Signed": li_diff_signed,
                     "Weight": weight,
                     "H_Dist": h_dist,
                     "Quality_Score": quality_score,
                     "Text_Score": text_score,

                     # 🆕 METRICHE SEMANTICHE
                     "BLEU": bleu,
                     "SBERT_similarity": sbert_sim,
                     "COMET_score": comet_like,
                     "NLI_entailment": entail
                    }
                )
            except Exception as e:
                print(f"Errore parsing riga {record.get('example_index', '?')}: {e}")

    df = pd.DataFrame(data)

    # Ensure numeric types for critical columns to avoid TypeError
    cols_to_numeric = [
        "Human_LI",
        "AI_LI",
        "LI_Error_Abs",
        "LI_Error_Signed",
        "Weight",
        "H_Dist",
        "Quality_Score",
        "Text_Score",
    ]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Model in stringa per sicurezza
    if "Model" in df.columns:
        df["Model"] = df["Model"].astype(str)

    return df


# ------------------------------------------------------------------
# FUNZIONI DI SUPPORTO STATISTICO: ICC, Bland-Altman, Regressione
# ------------------------------------------------------------------

def compute_icc_2_1(values1, values2):
    """
    Calcola ICC(2,1) (two-way random effects, absolute agreement, single rater).
    Adattato per 2 'rater' (Human vs AI).
    Restituisce np.nan se i dati sono insufficienti.
    """
    data = np.column_stack([values1, values2]).astype(float)
    # Rimuovi righe con NaN
    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]

    n, k = data.shape
    if n < 2 or k != 2:
        return np.nan

    # Medie
    mean_raters = data.mean(axis=0)
    mean_subjects = data.mean(axis=1)
    grand_mean = data.mean()

    # Somme dei quadrati
    ss_total = ((data - grand_mean) ** 2).sum()
    ss_subjects = (k * ((mean_subjects - grand_mean) ** 2)).sum()
    ss_raters = (n * ((mean_raters - grand_mean) ** 2)).sum()
    ss_error = ss_total - ss_subjects - ss_raters

    df_subjects = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)

    if df_subjects <= 0 or df_raters <= 0 or df_error <= 0:
        return np.nan

    ms_subjects = ss_subjects / df_subjects
    ms_raters = ss_raters / df_raters
    ms_error = ss_error / df_error

    # ICC(2,1)
    icc = (ms_subjects - ms_error) / (
        ms_subjects
        + (k - 1) * ms_error
        + (k * (ms_raters - ms_error) / n)
    )
    return icc


def bland_altman_plot(df):
    """
    Bland-Altman plot per Human_LI vs AI_LI.
    Dimostra accordo tra le misure biomeccaniche.
    I punti sono colorati per 'Model'.
    """
    data = df.dropna(subset=["Human_LI", "AI_LI"]).copy()
    if data.empty:
        print("[Bland-Altman] Nessun dato disponibile per il grafico.")
        return

    data["Mean_LI"] = (data["Human_LI"] + data["AI_LI"]) / 2.0
    data["Diff_LI"] = data["AI_LI"] - data["Human_LI"]

    mean_diff = data["Diff_LI"].mean()
    sd_diff = data["Diff_LI"].std(ddof=1)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="Mean_LI",
        y="Diff_LI",
        hue="Model",
        data=data,
        alpha=0.7
    )

    plt.axhline(mean_diff, color="red", linestyle="--", label=f"Bias = {mean_diff:.3f}")
    plt.axhline(loa_upper, color="gray", linestyle="--", label=f"+1.96 SD = {loa_upper:.3f}")
    plt.axhline(loa_lower, color="gray", linestyle="--", label=f"-1.96 SD = {loa_lower:.3f}")

    plt.xlabel("Media LI (Human & AI)")
    plt.ylabel("Differenza LI (AI - Human)")
    plt.title("Bland-Altman Plot per LI (Human vs AI)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bland_altman_LI.png")
    print("-> Bland-Altman plot salvato come 'bland_altman_LI.png'")


def regression_error_model(df):
    """
    Analisi di regressione per modellare l'errore LI (AI - Human)
    in funzione di Human_LI, Weight e H_Dist.
    Stampa i coefficienti globali e genera un grafico con retta di regressione
    (Errore vs Human_LI) colorata per modello.
    """
    cols_needed = ["Human_LI", "Weight", "H_Dist", "LI_Error_Signed", "Model"]
    df_reg = df[cols_needed].dropna()
    if df_reg.empty:
        print("[Regressione] Nessun dato sufficiente per la regressione.")
        return

    # Modello globale
    X = df_reg[["Human_LI", "Weight", "H_Dist"]].values
    y = df_reg["LI_Error_Signed"].values
    X_design = np.column_stack([np.ones(len(X)), X])  # colonna di intercept

    beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ beta
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print("\n--- REGRESSIONE GLOBALE DELL'ERRORE LI (AI - Human) ---")
    print(f"Intercetta:           {beta[0]:.4f}")
    print(f"Coefficiente Human_LI:{beta[1]:.4f}")
    print(f"Coefficiente Weight:  {beta[2]:.4f}")
    print(f"Coefficiente H_Dist:  {beta[3]:.4f}")
    print(f"R^2 globale:          {r2:.4f}")

    # Regressioni per modello (solo come riepilogo numerico)
    print("\n--- REGRESSIONE PER MODELLO ---")
    for model_name, g in df_reg.groupby("Model"):
        if len(g) < 5:
            print(f"Model={model_name}: dati insufficienti (<5 esempi) per regressione robusta.")
            continue
        X_m = g[["Human_LI", "Weight", "H_Dist"]].values
        y_m = g["LI_Error_Signed"].values
        X_m_design = np.column_stack([np.ones(len(X_m)), X_m])
        beta_m, _, _, _ = np.linalg.lstsq(X_m_design, y_m, rcond=None)
        y_m_pred = X_m_design @ beta_m
        ss_res_m = ((y_m - y_m_pred) ** 2).sum()
        ss_tot_m = ((y_m - y_m.mean()) ** 2).sum()
        r2_m = 1 - ss_res_m / ss_tot_m if ss_tot_m > 0 else np.nan

        print(
            f"Model={model_name} | Intercetta={beta_m[0]:.4f}, "
            f"Human_LI={beta_m[1]:.4f}, Weight={beta_m[2]:.4f}, "
            f"H_Dist={beta_m[3]:.4f}, R^2={r2_m:.4f}"
        )

    # Grafico: errore vs Human_LI con regressione per modello
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df_reg,
        x="Human_LI",
        y="LI_Error_Signed",
        hue="Model",
        alpha=0.7
    )
    # Aggiungo linea orizzontale a 0 (assenza di errore)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0, label="Errore = 0")

    plt.xlabel("Human LI")
    plt.ylabel("Errore LI (AI - Human)")
    plt.title("Errore LI vs Human LI (per modello)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regressione_errore_LI_vs_HumanLI.png")
    print("-> Grafico regressione errore salvato come 'regressione_errore_LI_vs_HumanLI.png'")


# ------------------------------------------------------------------
# ANALISI DI BASE + ESTENSIONI PER MODELLO
# ------------------------------------------------------------------

def generate_scientific_validation(df):
    """Fase 1: Dimostra che l'AI è accurata."""
    print("\n--- 1. VALIDAZIONE SCIENTIFICA (Metriche) ---")

    # Calcolo metriche chiave globali
    mae = df["LI_Error_Abs"].mean()
    rmse = np.sqrt(((df["Human_LI"] - df["AI_LI"]) ** 2).mean())
    correlation = df["Human_LI"].corr(df["AI_LI"])

    #  METRICHE SEMANTICHE
    mean_bleu = df["BLEU"].mean()
    mean_sbert = df["SBERT_similarity"].mean()
    mean_comet = df["COMET_score"].mean()
    mean_entail = df["NLI_entailment"].mean()

    print(f"Esempi Analizzati: {len(df)}")
    print(f"Errore Medio Assoluto (MAE) su LI: {mae:.4f}")
    print(f"Radice dell'Errore Quadratico Medio (RMSE): {rmse:.4f}")
    print(f"Correlazione Pearson (Human vs AI): {correlation:.4f} (1.0 = Perfetto)")
    print(f"BLEU medio: {mean_bleu:.3f}")
    print(f"SBERT similarity media: {mean_sbert:.3f}")
    print(f"COMET-like medio: {mean_comet:.3f}")
    print(f"NLI entailment medio: {mean_entail:.3f}")
    # Metriche per modello
    print("\n--- Metriche per MODEL ---")
    if "Model" in df.columns:
        for model_name, g in df.groupby("Model"):
            if g.empty:
                continue
            mae_m = g["LI_Error_Abs"].mean()
            rmse_m = np.sqrt(((g["Human_LI"] - g["AI_LI"]) ** 2)).mean()
            corr_m = g["Human_LI"].corr(g["AI_LI"])
            print(
                f"Model={model_name} | N={len(g)} | "
                f"MAE={mae_m:.4f} | RMSE={rmse_m:.4f} | Corr={corr_m:.4f}"
            )

    # Creazione Dashboard Grafica
    plt.figure(figsize=(14, 6))

    # Plot A: Scatter Plot (La prova visiva) - HUE per Model
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="Human_LI",
        y="AI_LI",
        hue="Model",
        data=df,
        alpha=0.6
    )

    # Linea di perfezione
    max_val = max(df["Human_LI"].max(), df["AI_LI"].max())
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect Match")

    plt.title(f"Accuratezza LI: Human vs AI (Corr globale: {correlation:.2f})")
    plt.xlabel("Human Lifting Index (Ground Truth)")
    plt.ylabel("AI Lifting Index")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot B: Distribuzione Qualità (Gemini) per modello
    plt.subplot(1, 2, 2)
    if "Model" in df.columns:
        sns.histplot(
            data=df,
            x="Quality_Score",
            hue="Model",
            bins=10,
            kde=True,
            multiple="layer",
            alpha=0.6
        )
    else:
        sns.histplot(df["Quality_Score"], bins=10, kde=True, color="green")

    plt.axvline(60, color="red", linestyle="--", label="Soglia Accettabilità (60)")
    plt.title("Distribuzione Punteggi Qualitativi (Gemini) per modello")
    plt.xlabel("Punteggio (0-100)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("report_validazione_scientifica.png")
    print("-> Grafico salvato come 'report_validazione_scientifica.png'")
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x="SBERT_similarity",
        y="LI_Error_Abs",
        hue="Model",
        data=df,
        alpha=0.7
    )
    plt.title("Correlazione SBERT similarity vs Errore LI")
    plt.xlabel("SBERT similarity")
    plt.ylabel("Errore LI (|AI − Human|)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("correlazione_sbert_vs_errore.png")
    print("-> Grafico salvato: correlazione_sbert_vs_errore.png")


def debugging_mirato(df):
    """Fase 2: Trova dove e perché l'AI sbaglia."""
    print("\n--- 2. DEBUGGING MIRATO (Analisi Errori) ---")

    # 1. Trova i "Casi Limite" (Worst Offenders)
    worst_cases = df[df["Quality_Score"] < 60].sort_values(by="Quality_Score")

    if not worst_cases.empty:
        print(f"\n[!] Trovati {len(worst_cases)} report critici (Score < 60):")
        print(
            worst_cases[
                ["index", "Model", "Human_LI", "AI_LI", "Quality_Score", "Weight"]
            ].to_string(index=False)
        )

        # Esporta per analisi manuale
        worst_cases.to_csv("casi_da_revisionare.csv", index=False)
        print("-> Dettagli salvati in 'casi_da_revisionare.csv'")
    else:
        print("\n[OK] Nessun report sotto la soglia di qualità 60.")

    # 2. Analisi dei Pattern (Correlazioni Nascoste)
    # Cerchiamo se l'errore aumenta all'aumentare del peso o della distanza
    print("\n[?] Analisi Cause Radice (Correlazione Errore):")

    # Correlazione tra Errore LI e Peso dell'oggetto
    corr_weight = df["LI_Error_Abs"].corr(df["Weight"])
    print(f"   Correlazione Errore vs Peso Oggetto: {corr_weight:.2f}")
    if corr_weight > 0.3:
        print("   -> ATTENZIONE: L'AI tende a sbagliare di più con oggetti pesanti.")

    # Correlazione tra Errore LI e Distanza Orizzontale
    corr_dist = df["LI_Error_Abs"].corr(df["H_Dist"])
    print(f"   Correlazione Errore vs Distanza Orizzontale: {corr_dist:.2f}")

    # Plot dei Pattern di Errore (colorato per Model)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        x="Weight",
        y="LI_Error_Abs",
        data=df,
        hue="Model",
        style="Model",
        alpha=0.7
    )
    plt.title("Pattern di Errore: L'errore dipende dal Peso? (per modello)")
    plt.xlabel("Peso Oggetto (lbs/kg)")
    plt.ylabel("Errore Assoluto LI (|Human - AI|)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("analisi_pattern_errore.png")
    print("-> Grafico pattern salvato come 'analisi_pattern_errore.png'")


def icc_analysis(df):
    """
    Calcola ICC globale e per modello, stampa i risultati.
    """
    print("\n--- 3. ANALISI ICC (Intraclass Correlation Coefficient) ---")

    # Globale
    icc_global = compute_icc_2_1(df["Human_LI"].values, df["AI_LI"].values)
    print(f"ICC(2,1) globale (Human vs AI): {icc_global:.4f}")

    # Per modello
    if "Model" in df.columns:
        for model_name, g in df.groupby("Model"):
            if len(g) < 3:
                print(f"Model={model_name}: dati insufficienti per ICC (<3 esempi).")
                continue
            icc_m = compute_icc_2_1(g["Human_LI"].values, g["AI_LI"].values)
            print(f"Model={model_name} | ICC(2,1): {icc_m:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python analyzer.py out_eval.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]

    # 1. Carica dati
    print(f"Caricamento dati da {input_file}...")
    df = load_data(input_file)

    if df.empty:
        print("Errore: Nessun dato valido trovato nel file.")
        return

    # 2. Analisi di base
    generate_scientific_validation(df)
    debugging_mirato(df)

    # 3. Nuove analisi richieste
    icc_analysis(df)
    bland_altman_plot(df)
    regression_error_model(df)

    print("\nAnalisi completata.")


if __name__ == "__main__":
    main()
