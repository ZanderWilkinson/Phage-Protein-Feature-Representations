from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from lightgbm import LGBMClassifier

from config import RAW_DIR, PROCESSED_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "embeddings"


y_train = pd.read_csv(RAW_DIR / "phage_protein_student_train.csv")["label"]
y_test = pd.read_csv(RAW_DIR / "phage_protein_student_test.csv")["label"]

label_names = sorted(y_train.unique())
label_to_int = {label: i for i, label in enumerate(label_names)}

y_train_enc = y_train.map(label_to_int)
y_test_enc = y_test.map(label_to_int)


def load_classical():
    Xtr = pd.read_csv(PROCESSED_DIR / "classical" / "train_classical_final.csv")
    Xte = pd.read_csv(PROCESSED_DIR / "classical" / "test_classical_final.csv")
    return Xtr, Xte

def load_pretrained():
    Xtr = pd.read_csv(DATA_DIR / "phage_protein_student_train_esm2_t33_650M_UR50D.csv")
    Xte = pd.read_csv(DATA_DIR / "phage_protein_student_test_esm2_t33_650M_UR50D.csv")
    if not np.issubdtype(Xtr.iloc[:,0].dtype, np.number):
        Xtr = Xtr.iloc[:,1:]
        Xte = Xte.iloc[:,1:]
    return Xtr, Xte

def load_finetuned():
    Xtr = pd.read_csv(DATA_DIR / "phage_protein_student_train_esm2_t33_650M_UR50D_ft.csv")
    Xte = pd.read_csv(DATA_DIR / "phage_protein_student_test_esm2_t33_650M_UR50D_ft.csv")
    if not np.issubdtype(Xtr.iloc[:,0].dtype, np.number):
        Xtr = Xtr.iloc[:,1:]
        Xte = Xte.iloc[:,1:]
    return Xtr, Xte

representations = {
    "Classical": load_classical,
    "Pretrained": load_pretrained,
    "FineTuned": load_finetuned
}

for name, loader in representations.items():

    print("\n==============================")
    print("Representation:", name)
    print("==============================")

    X_train, X_test = loader()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    gbm = LGBMClassifier(n_estimators=500, random_state=42, verbose=-1)
    gbm.fit(X_train, y_train_enc)

    preds = gbm.predict(X_test)


    report = classification_report(
        y_test_enc,
        preds,
        target_names=label_names,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{name}_classification_report.csv")


    cm = confusion_matrix(y_test_enc, preds)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = np.array([
        f"{count}\n({pct:.2f})"
        for count, pct in zip(cm.flatten(), cm_norm.flatten())
    ]).reshape(cm.shape)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm_norm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5
    )

    plt.title(f"Confusion Matrix - {name} (GBM)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix.png", dpi=300)
    plt.show()
    print(cm[label_names.index("whiskers")])