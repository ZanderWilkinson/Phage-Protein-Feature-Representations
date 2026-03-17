from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from config import RAW_DIR, PROCESSED_DIR

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data" / "embeddings"


y_train = pd.read_csv(RAW_DIR / "phage_protein_student_train.csv")["label"]
y_test = pd.read_csv(RAW_DIR / "phage_protein_student_test.csv")["label"]

label_codes, label_names = pd.factorize(y_train)
label_map = dict(enumerate(label_names))

y_train_enc = label_codes
y_test_enc = pd.Categorical(y_test, categories=label_names).codes


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
results = []

for name, loader in representations.items():


    print("Representation:", name)


    X_train, X_test = loader()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_enc)
    pred_rf = rf.predict(X_test)

    rf_f1 = f1_score(y_test_enc, pred_rf, average="macro")
    print("RF Macro F1:", rf_f1)

    results.append({
        "representation": name,
        "model": "RF",
        "macro_f1": rf_f1
    })


    gbm = LGBMClassifier(n_estimators=500, random_state=42, verbose=-1)
    gbm.fit(X_train, y_train_enc)
    pred_gbm = gbm.predict(X_test)

    gbm_f1 = f1_score(y_test_enc, pred_gbm, average="macro")
    print("GBM Macro F1:", gbm_f1)

    results.append({
        "representation": name,
        "model": "GBM",
        "macro_f1": gbm_f1
    })


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr_tensor = torch.tensor(y_train_enc, dtype=torch.long).to(device)

    Xte_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    yte_tensor = torch.tensor(y_test_enc, dtype=torch.long).to(device)

    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(X_train.shape[1], len(label_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        outputs = model(Xtr_tensor)
        loss = criterion(outputs, ytr_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(Xte_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    nn_f1 = f1_score(y_test_enc, preds, average="macro")
    print("NN Macro F1:", nn_f1)

    results.append({
        "representation": name,
        "model": "NN",
        "macro_f1": nn_f1
    })
output_dir = Path("results/metrics")
output_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(output_dir / "final_test_macro_f1.csv", index=False)
