# =============================================================
# PERBANDINGAN 3 MODEL DENGAN UNDERSAMPLING
# Pipeline SAMA seperti track SMOTE, kecuali langkah resampling:
# SMOTE (oversampling) diganti RandomUnderSampler (undersampling).
# Train menyusut dari 183.824 -> ~11rb baris (3 x kelas terkecil),
# sehingga SVM kernel RBF full-train bisa selesai dalam hitungan
# menit. Test set FULL & IDENTIK dengan track SMOTE
# (split random_state=42) sehingga confusion matrix comparable.
# Output -> models/undersampling/ + outputs/undersampling/
# =============================================================

import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             classification_report)
from imblearn.under_sampling import RandomUnderSampler
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"
OUT = BASE / "outputs" / "undersampling"
MODELS = BASE / "models" / "undersampling"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# 1. INPUT
df = pd.read_csv(DATA)
print("Ukuran data awal :", df.shape)

# 2. PREPROCESSING (identik SMOTE)
print("Missing value total:", df.isnull().sum().sum())
print("Duplikat:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
for kol in ["BMI", "MentHlth", "PhysHlth"]:
    Q1, Q3 = df[kol].quantile(0.25), df[kol].quantile(0.75)
    IQR = Q3 - Q1
    df[kol] = df[kol].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
print("Ukuran setelah preprocessing:", df.shape)

X = df.drop(columns=["Diabetes_012"])
y = df["Diabetes_012"]
feature_columns = list(X.columns)

# 3. SPLIT (identik SMOTE -> test set sama persis)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])

# 4. TRANSFORMASI
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, MODELS / "scaler.pkl")
joblib.dump(feature_columns, MODELS / "feature_columns.pkl")

# 5. RESAMPLING: UNDERSAMPLING (pengganti SMOTE, hanya train)
print("\nDistribusi SEBELUM undersampling:")
print(pd.Series(y_train).value_counts().sort_index().to_string())
rus = RandomUnderSampler(random_state=42)
X_tr_res, y_tr_res = rus.fit_resample(X_train_scaled, y_train)
print("\nDistribusi SESUDAH undersampling:")
print(pd.Series(y_tr_res).value_counts().sort_index().to_string())

# 6-8. TRAINING + EVALUASI 3 model
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(random_state=42, cache_size=500),
}
hasil = []
for nama, model in models.items():
    t0 = time.time()
    model.fit(X_tr_res, y_tr_res)
    print(f"{nama} dilatih dalam {time.time()-t0:.1f} detik.")
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    ak = accuracy_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rc = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"[{nama}] CM:\n{cm}")
    print(f"[{nama}] Akurasi: {ak:.4f} | Presisi: {pr:.4f} | Recall: {rc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["0", "1", "2"], yticklabels=["0", "1", "2"])
    plt.xlabel("Prediksi"); plt.ylabel("Aktual")
    plt.title(f"Confusion Matrix - {nama} (undersampling)\nAkurasi: {ak:.4f}")
    plt.tight_layout(); plt.savefig(OUT / f"confusion_matrix_{nama}.png"); plt.close()
    joblib.dump(model, MODELS / f"model_{nama}.pkl")
    hasil.append({"Model": nama, "Akurasi": ak, "Presisi_weighted": pr,
                  "Recall_weighted": rc, "F1_weighted": f1})

df_hasil = pd.DataFrame(hasil).sort_values("Akurasi", ascending=False)
print("\nRINGKASAN UNDERSAMPLING:\n", df_hasil.to_string(index=False))
df_hasil.to_csv(OUT / "hasil_perbandingan.csv", index=False)

plt.figure(figsize=(8, 5))
x = np.arange(len(df_hasil)); lebar = 0.2
for i, kol in enumerate(["Akurasi", "Presisi_weighted", "Recall_weighted", "F1_weighted"]):
    plt.bar(x + (i - 1.5) * lebar, df_hasil[kol], width=lebar, label=kol)
plt.xticks(x, df_hasil["Model"]); plt.ylim(0, 1); plt.ylabel("Skor")
plt.title("Perbandingan Kinerja (Undersampling): DT vs NB vs SVM")
plt.legend(); plt.tight_layout(); plt.savefig(OUT / "perbandingan_metrik.png"); plt.close()

open(MODELS / "model_terbaik.txt", "w").write(df_hasil.iloc[0]["Model"])
print(f"\nModel terbaik (undersampling): {df_hasil.iloc[0]['Model']}")
print("Selesai.")
