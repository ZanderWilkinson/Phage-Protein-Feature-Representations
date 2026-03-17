

from pathlib import Path
import pandas as pd
from config import PROCESSED_DIR

COMP_DIR = PROCESSED_DIR / "classical" / "components"
OUT_DIR = PROCESSED_DIR / "classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP5_TRAIN = [
    "train_QS.csv",
    "train_PAAC.csv",
    "train_APAAC.csv",
    "train_CTDD.csv",
    "train_kmer.csv"
]

TOP5_TEST = [f.replace("train_", "test_") for f in TOP5_TRAIN]

def merge(files):
    dfs = [pd.read_csv(COMP_DIR / f) for f in files]
    n = dfs[0].shape[0]
    for df in dfs:
        assert df.shape[0] == n
    return pd.concat(dfs, axis=1)

X_train = merge(TOP5_TRAIN)
X_test = merge(TOP5_TEST)

X_train.to_csv(OUT_DIR / "train_classical_final.csv", index=False)
X_test.to_csv(OUT_DIR / "test_classical_final.csv", index=False)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

