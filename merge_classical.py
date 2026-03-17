from pathlib import Path
import pandas as pd
from config import PROCESSED_DIR

COMP_DIR = PROCESSED_DIR / "classical" / "components"
OUT_DIR = PROCESSED_DIR / "classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP5 = [
    "train_QS.csv",
    "train_PAAC.csv",
    "train_APAAC.csv",
    "train_CTDD.csv",
    "train_kmer.csv"
]

dfs = [pd.read_csv(COMP_DIR / f) for f in TOP5]


n = dfs[0].shape[0]
for df in dfs:
    assert df.shape[0] == n

X = pd.concat(dfs, axis=1)
X.to_csv(OUT_DIR / "train_classical_final.csv", index=False)

print("Saved classical final:", X.shape)