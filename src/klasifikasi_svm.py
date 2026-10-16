# =============================================================
# KLASIFIKASI DIABETES - BRFSS 2015 Health Indicators
# Metode: SVM (Support Vector Machine, kernel RBF)
# Alur PERSIS SAMA seperti klasifikasi_diabetes_kelompok2.py
# (Decision Tree), hanya klasifikatornya diganti:
# Input -> Preprocessing -> Split Data -> Transformasi (train & test)
# -> Resampling (oversampling SMOTE, hanya train) -> Training (SVM)
# -> Testing -> Evaluasi
#
# CATATAN: Dengan pipeline penuh (SMOTE -> ~500rb baris train),
# SVM kernel RBF membutuhkan waktu SANGAT lama (orde jam).
# Jika ingin lebih cepat, gunakan sampling atau LinearSVC,
# tetapi kode di bawah ini tetap memakai alur persis sama.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             classification_report)
from imblearn.over_sampling import SMOTE
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"
OUT = BASE / "outputs" / "smote"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. INPUT DATA
# -------------------------------------------------------------
FILE = DATA
df = pd.read_csv(FILE)
print("=" * 60)
print("1. INPUT DATA")
print("=" * 60)
print("Ukuran data awal :", df.shape)

# -------------------------------------------------------------
# 2. PREPROCESSING : missing value, duplikasi, outlier
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

# a) Deteksi missing value
print("\n[a] Missing value per kolom (total):", df.isnull().sum().sum())

# b) Deteksi & hapus duplikasi data
dup = df.duplicated().sum()
print(f"[b] Jumlah data duplikat : {dup}")
df = df.drop_duplicates().reset_index(drop=True)
print("    Ukuran setelah hapus duplikat :", df.shape)

# c) Deteksi & tangani outlier (metode IQR pada fitur numerik kontinu)
kolom_numerik = ["BMI", "MentHlth", "PhysHlth"]
print("[c] Deteksi outlier (IQR) pada:", kolom_numerik)
for kol in kolom_numerik:
    Q1 = df[kol].quantile(0.25)
    Q3 = df[kol].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR
    jml_outlier = ((df[kol] < batas_bawah) | (df[kol] > batas_atas)).sum()
    print(f"    {kol:10s}: {jml_outlier} outlier  (batas {batas_bawah:.1f} - {batas_atas:.1f})")
    # Capping: nilai di luar batas dipotong ke batas terdekat
    df[kol] = df[kol].clip(lower=batas_bawah, upper=batas_atas)
print("    Outlier ditangani dengan capping (clip ke batas IQR).")

# Pisahkan fitur (X) dan target (y)
X = df.drop(columns=["Diabetes_012"])
y = df["Diabetes_012"]

# -------------------------------------------------------------
# 3. SPLIT DATA : 80% training - 20% testing
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("3. SPLIT DATA (80% training : 20% testing)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Data training :", X_train.shape[0], "baris")
print("Data testing  :", X_test.shape[0], "baris")

# -------------------------------------------------------------
# 4. TRANSFORMASI : MinMaxScaler (fit di train, transform ke train & test)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRANSFORMASI (MinMaxScaler)")
print("=" * 60)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)   # fit HANYA pada data training
X_test = scaler.transform(X_test)         # test hanya di-transform
print("Semua fitur diskalakan ke rentang [0, 1].")

# -------------------------------------------------------------
# 5. RESAMPLING : Oversampling (SMOTE) - hanya pada data training
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("5. RESAMPLING (Oversampling - SMOTE)")
print("=" * 60)
# --- Distribusi SEBELUM SMOTE (data imbalance) ---
sebelum = pd.Series(y_train).value_counts().sort_index()
print("Distribusi kelas SEBELUM oversampling:")
print(sebelum.to_string())

plt.figure(figsize=(6, 4))
sns.barplot(x=sebelum.index, y=sebelum.values,
            palette='coolwarm', hue=sebelum.index, legend=False)
plt.xlabel("Kelas Diabetes_012")
plt.ylabel("Jumlah")
plt.title("Distribusi Sebelum SMOTE (Imbalance) - SVM")
plt.tight_layout()
plt.savefig(OUT / "distribusi_sebelum_smote_SVM.png")

# --- Proses oversampling ---
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# --- Distribusi SESUDAH SMOTE (seimbang) ---
sesudah = pd.Series(y_train).value_counts().sort_index()
print("\nDistribusi kelas SESUDAH oversampling:")
print(sesudah.to_string())

plt.figure(figsize=(6, 4))
sns.barplot(x=sesudah.index, y=sesudah.values,
            palette='coolwarm', hue=sesudah.index, legend=False)
plt.xlabel("Kelas Diabetes_012")
plt.ylabel("Jumlah")
plt.title("Distribusi Sesudah SMOTE (Seimbang) - SVM")
plt.tight_layout()
plt.savefig(OUT / "distribusi_sesudah_smote_SVM.png")

plt.show()

# -------------------------------------------------------------
# 6. TRAINING : SVM (SVC kernel RBF, default)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("6. TRAINING (SVM - SVC RBF)")
print("=" * 60)
model = SVC(random_state=42)
model.fit(X_train, y_train)
print("Model SVM selesai dilatih.")

# -------------------------------------------------------------
# 7. TESTING : prediksi pada data testing
# -------------------------------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------------------------------
# 8. EVALUASI : Confusion Matrix, Akurasi, Presisi, Recall, F1
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("8. EVALUASI")
print("=" * 60)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

akurasi = accuracy_score(y_test, y_pred)
presisi = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\nAkurasi   : {akurasi:.4f}")
print(f"Presisi   : {presisi:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nLaporan lengkap per kelas:")
print(classification_report(y_test, y_pred, zero_division=0))
