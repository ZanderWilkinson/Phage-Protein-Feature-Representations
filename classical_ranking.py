from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from config import RAW_DIR, PROCESSED_DIR, METRICS_DIR, FIG_DIR



CLASSICAL_FILES = [
    "train_kmer.csv",
    "train_GAAC.csv",
    "train_GDPC.csv",
    "train_GTPC.csv",
    "train_DDE.csv",
    "train_PAAC.csv",
    "train_APAAC.csv",
    "train_CTDC.csv",
    "train_CTDT.csv",
    "train_CTDD.csv",
    "train_CTriad.csv",
    "train_SOC.csv",
    "train_QS.csv",
]

COMP_DIR = PROCESSED_DIR / "classical" / "components"


def load_y() -> pd.Series:
    y = pd.read_csv(RAW_DIR / "phage_protein_student_train.csv")["label"]
    return y


def load_X(csv_name: str) -> pd.DataFrame:
    X = pd.read_csv(COMP_DIR / csv_name)

    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():

        n_nans = int(X.isna().sum().sum())
        raise ValueError(f"{csv_name} contains NaNs after numeric coercion: {n_nans}")
    return X


def main(seed: int = 42):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    y = load_y()


    idx = np.arange(len(y))
    idx_tr, idx_val = train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    results = []

    for fname in CLASSICAL_FILES:
        X = load_X(fname)

        X_tr = X.iloc[idx_tr, :]
        y_tr = y.iloc[idx_tr]
        X_val = X.iloc[idx_val, :]
        y_val = y.iloc[idx_val]


        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("rf", RandomForestClassifier(
                    n_estimators=400,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                )),
            ]
        )

        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)

        macro_f1 = f1_score(y_val, pred, average="macro")

        results.append({
            "descriptor_file": fname,
            "n_features": X.shape[1],
            "macro_f1_val": macro_f1,
        })

        print(f"{fname:28s}  features={X.shape[1]:5d}  macroF1={macro_f1:.4f}")

    df = pd.DataFrame(results).sort_values("macro_f1_val", ascending=False)
    out_csv = METRICS_DIR / "classical_descriptor_ranking.csv"
    df.to_csv(out_csv, index=False)
    print("\nSaved ranking:", out_csv)


    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.bar(df["descriptor_file"], df["macro_f1_val"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Macro F1 (validation)")
    plt.title("Validation performance per classical descriptor (RF probe)")
    plt.tight_layout()
    out_fig = FIG_DIR / "classical_descriptor_ranking.png"
    plt.savefig(out_fig, dpi=300)
    plt.show()
    print("Saved plot:", out_fig)


if __name__ == "__main__":
    main()