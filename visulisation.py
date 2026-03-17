from pathlib import Path
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import RAW_DIR, PROCESSED_DIR
plt.style.use("seaborn-v0_8-whitegrid")
plt.savefig("filename.png", dpi=300)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"


y = pd.read_csv(RAW_DIR / "phage_protein_student_train.csv")["label"]


X_classical = pd.read_csv(PROCESSED_DIR / "classical" / "train_classical_final.csv")
X_pre = pd.read_csv(DATA_DIR / "phage_protein_student_train_esm2_t33_650M_UR50D.csv")
X_ft = pd.read_csv(DATA_DIR / "phage_protein_student_train_esm2_t33_650M_UR50D_ft.csv")


if not np.issubdtype(X_pre.iloc[:,0].dtype, np.number):
    X_pre = X_pre.iloc[:,1:]
if not np.issubdtype(X_ft.iloc[:,0].dtype, np.number):
    X_ft = X_ft.iloc[:,1:]

representations = {
    "Classical": X_classical,
    "Pretrained": X_pre,
    "FineTuned": X_ft
}

for name, X in representations.items():

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    labels = y.astype("category")
    class_names = labels.cat.categories
    class_codes = labels.cat.codes

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=class_codes,
        cmap="tab20",
        s=5
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    plt.title(f"PCA - {name}")


    handles = []
    for i, cls in enumerate(class_names):
        handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       label=cls,
                       markerfacecolor=plt.cm.tab20(i),
                       markersize=6)
        )

    plt.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=7
    )

    plt.tight_layout()
    plt.show()


    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=class_codes,
        cmap="tab20",
        s=5
    )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"UMAP - {name}")

    plt.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=7
    )

    plt.tight_layout()
    plt.show()