from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from config import DATA_DIR, RAW_DIR


EMB_DIR = DATA_DIR / "embeddings"

PRETRAINED_FILE = "phage_protein_student_train_esm2_t33_650M_UR50D.csv"
FINETUNED_FILE = "phage_protein_student_train_esm2_t33_650M_UR50D_ft.csv"


def load_y():
    return pd.read_csv(RAW_DIR / "phage_protein_student_train.csv")["label"]


def load_embeddings(filename):
    df = pd.read_csv(EMB_DIR / filename)


    if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        df = df.iloc[:, 1:]

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

    X_pre = load_embeddings(PRETRAINED_FILE)
    X_ft = load_embeddings(FINETUNED_FILE)

    score_pre = evaluate(X_pre, y, idx_tr, idx_val)
    score_ft = evaluate(X_ft, y, idx_tr, idx_val)

    print("\nValidation Macro F1 (RF probe):")
    print(f"Pretrained ESM   : {score_pre:.4f}")
    print(f"Fine-tuned ESM   : {score_ft:.4f}")


if __name__ == "__main__":
    main()