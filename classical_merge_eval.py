from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from config import RAW_DIR, PROCESSED_DIR

COMP_DIR = PROCESSED_DIR / "classical" / "components"

FILES = {
    "QSOrder": "train_QS.csv",
    "PAAC": "train_PAAC.csv",
    "APAAC": "train_APAAC.csv",
    "CTDD": "train_CTDD.csv",
    "Kmer": "train_kmer.csv",
}
def load_y():
    return pd.read_csv(RAW_DIR / "phage_protein_student_train.csv")["label"]

def load_X(fname):
    df = pd.read_csv(COMP_DIR / fname)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError(f"NaNs detected in {fname}")
    return df

def evaluate(X, y, idx_tr, idx_val, seed=42):
    X_tr = X.iloc[idx_tr]
    y_tr = y.iloc[idx_tr]
    X_val = X.iloc[idx_val]
    y_val = y.iloc[idx_val]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])

    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)
    return f1_score(y_val, pred, average="macro")

def main():
    y = load_y()

    idx = np.arange(len(y))
    idx_tr, idx_val = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    X = {name: load_X(fname) for name, fname in FILES.items()}

    results = {}


    results["QSOrder_only"] = evaluate(X["QSOrder"], y, idx_tr, idx_val)


    top3 = pd.concat([X["QSOrder"], X["PAAC"], X["APAAC"]], axis=1)
    results["Top3"] = evaluate(top3, y, idx_tr, idx_val)


    top5 = pd.concat(
        [X["QSOrder"], X["PAAC"], X["APAAC"], X["CTDD"], X["Kmer"]],
        axis=1
    )
    results["Top5"] = evaluate(top5, y, idx_tr, idx_val)

    print("\nClassical combination results (Macro F1 validation):")
    for k, v in results.items():
        print(f"{k:15s}: {v:.4f}")

if __name__ == "__main__":
    main()