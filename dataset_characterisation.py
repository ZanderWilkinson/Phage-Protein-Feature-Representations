import pandas as pd
import matplotlib.pyplot as plt
from config import RAW_DIR, FIG_DIR

train = pd.read_csv(RAW_DIR / "phage_protein_student_train.csv")
counts = train["label"].value_counts().sort_values(ascending=False)

plt.figure(figsize=(10, 4))
counts.plot(kind="bar")
plt.ylabel("Number of proteins")
plt.title("Training set class distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution_train.png", dpi=300)
plt.show()



test = pd.read_csv(RAW_DIR / "phage_protein_student_test.csv")

train_prop = train["label"].value_counts(normalize=True)
test_prop = test["label"].value_counts(normalize=True)

df = pd.DataFrame({"Train": train_prop, "Test": test_prop}).fillna(0)

plt.figure(figsize=(10, 4))
df.plot(kind="bar")
plt.ylabel("Proportion")
plt.title("Class proportions in training and test sets")
plt.tight_layout()
plt.savefig(FIG_DIR / "train_test_class_proportions.png", dpi=300)
plt.show()



from Bio import SeqIO
from config import RAW_DIR

lengths = [
    len(record.seq)
    for record in SeqIO.parse(RAW_DIR / "phage_protein_student_train.fasta", "fasta")
]

plt.figure(figsize=(6, 4))
plt.hist(lengths, bins=50)
plt.xlabel("Protein length (amino acids)")
plt.ylabel("Count")
plt.title("Sequence length distribution (training set)")
plt.tight_layout()
plt.savefig(FIG_DIR / "sequence_length_distribution.png", dpi=300)
plt.show()



import numpy as np

summary = {
    "Train proteins": len(train),
    "Test proteins": len(test),
    "Number of classes": train["label"].nunique(),
    "Min class size (train)": counts.min(),
    "Max class size (train)": counts.max(),
    "Median sequence length": int(np.median(lengths)),
    "Max sequence length": int(np.max(lengths)),
}

for k, v in summary.items():
    print(f"{k}: {v}")
