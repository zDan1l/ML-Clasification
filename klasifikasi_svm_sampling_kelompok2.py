# =============================================================
# KLASIFIKASI DIABETES - METODE SVM DENGAN SAMPLING (Opsi A)
# Pipeline SAMA PERSIS seperti klasifikasi_diabetes_kelompok2.py,
# kecuali SATU penyesuaian yang didokumentasikan:
#   - Data TEST: FULL (45.957 baris, split random_state=42) -> SAMA
#     seperti Decision Tree & Naive Bayes, sehingga confusion matrix
#     bisa dibandingkan apple-to-apple.
#   - Data TRAIN: diambil SAMPEL STRATIFIED 10.000 baris dari train
#     penuh, baru kemudian SMOTE. Alasan: SVC kernel RBF kompleksitas
#     O(n^2); train penuh pasca-SMOTE (~500rb baris) mustahil dilatih
#     di laptop (orde jam-hari / OOM).
# Langkah lain IDENTIK: missing value -> hapus duplikat ->
# capping outlier IQR (BMI, MentHlth, PhysHlth) -> split 80:20
# stratify(42) -> MinMaxScaler (fit train) -> SMOTE (train) ->
# SVC RBF -> testing -> evaluasi.
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
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             classification_report)
from imblearn.over_sampling import SMOTE
import joblib

N_SAMPEL_TRAIN = 10_000

# -------------------------------------------------------------
# 1. INPUT DATA
# -------------------------------------------------------------
FILE = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(FILE)
print("=" * 60)
print("1. INPUT DATA")
print("=" * 60)
print("Ukuran data awal :", df.shape)

# -------------------------------------------------------------
# 2. PREPROCESSING : missing value, duplikasi, outlier (IDENTIK)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)
print("\n[a] Missing value per kolom (total):", df.isnull().sum().sum())

dup = df.duplicated().sum()
print(f"[b] Jumlah data duplikat : {dup}")
df = df.drop_duplicates().reset_index(drop=True)
print("    Ukuran setelah hapus duplikat :", df.shape)

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
    df[kol] = df[kol].clip(lower=batas_bawah, upper=batas_atas)
print("    Outlier ditangani dengan capping (clip ke batas IQR).")

X = df.drop(columns=["Diabetes_012"])
y = df["Diabetes_012"]

# -------------------------------------------------------------
# 3. SPLIT DATA : 80% training - 20% testing (IDENTIK -> test sama
#    persis dengan DT & Naive Bayes)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("3. SPLIT DATA (80% training : 20% testing)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Data training (penuh) :", X_train.shape[0], "baris")
print("Data testing  (FULL, sama dgn DT/NB) :", X_test.shape[0], "baris")

# -------------------------------------------------------------
# 4. TRANSFORMASI : MinMaxScaler (fit di train penuh, IDENTIK)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRANSFORMASI (MinMaxScaler)")
print("=" * 60)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Semua fitur diskalakan ke rentang [0, 1].")

# -------------------------------------------------------------
# 4b. SAMPLING TRAIN (SATU-SATUNYA penyesuaian, terdokumentasi)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print(f"4b. SAMPLING TRAIN (stratified, n={N_SAMPEL_TRAIN})")
print("=" * 60)
X_tr_samp, _, y_tr_samp, _ = train_test_split(
    X_train_scaled, y_train, train_size=N_SAMPEL_TRAIN,
    random_state=42, stratify=y_train
)
print("Train sampel :", X_tr_samp.shape[0], "baris")
print(pd.Series(y_tr_samp).value_counts().sort_index().to_string())

# -------------------------------------------------------------
# 5. RESAMPLING : SMOTE hanya pada train sampel
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("5. RESAMPLING (Oversampling - SMOTE, pada train sampel)")
print("=" * 60)
smote = SMOTE(random_state=42)
X_tr_res, y_tr_res = smote.fit_resample(X_tr_samp, y_tr_samp)
print("Distribusi SESUDAH SMOTE:")
print(pd.Series(y_tr_res).value_counts().sort_index().to_string())
print(f"Ukuran train setelah SMOTE: {X_tr_res.shape}")

# -------------------------------------------------------------
# 6. TRAINING : SVM (SVC kernel RBF, default)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("6. TRAINING (SVM - SVC RBF)")
print("=" * 60)
model = SVC(random_state=42, cache_size=500)
t0 = time.time()
model.fit(X_tr_res, y_tr_res)
print(f"Model SVM selesai dilatih dalam {time.time()-t0:.1f} detik.")
joblib.dump(model, "model_SVM.pkl")

# -------------------------------------------------------------
# 7-8. TESTING + EVALUASI pada TEST FULL (sama dgn DT/NB)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("8. EVALUASI (pada test FULL 45.957 baris)")
print("=" * 60)
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
akurasi = accuracy_score(y_test, y_pred)
presisi = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nConfusion Matrix (SVM):")
print(cm)
print(f"\nAkurasi   : {akurasi:.4f}")
print(f"Presisi   : {presisi:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nLaporan lengkap per kelas:")
print(classification_report(y_test, y_pred, zero_division=0))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0", "1", "2"], yticklabels=["0", "1", "2"])
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title(f"Confusion Matrix - SVM (sampling train 10k)\nAkurasi: {akurasi:.4f}")
plt.tight_layout()
plt.savefig("confusion_matrix_SVM.png")
plt.close()

# Simpan ringkasan untuk dashboard
with open("hasil_svm.txt", "w") as f:
    f.write(f"akurasi={akurasi:.4f}\npresisi={presisi:.4f}\nrecall={recall:.4f}\nf1={f1:.4f}\n")
    f.write("cm=" + np.array2string(cm, separator=",") + "\n")

# Gabungkan ke hasil_perbandingan.csv
df_hasil = pd.read_csv("hasil_perbandingan.csv")
df_hasil = df_hasil[df_hasil["Model"] != "SVM"]
df_hasil = pd.concat([df_hasil, pd.DataFrame([{
    "Model": "SVM", "Akurasi": akurasi,
    "Presisi_weighted": presisi, "Recall_weighted": recall,
    "F1_weighted": f1}])], ignore_index=True)
df_hasil = df_hasil.sort_values("Akurasi", ascending=False)
df_hasil.to_csv("hasil_perbandingan.csv", index=False)
print("\nhasil_perbandingan.csv diperbarui:")
print(df_hasil.to_string(index=False))

plt.figure(figsize=(8, 5))
x = np.arange(len(df_hasil)); lebar = 0.2
for i, kol in enumerate(["Akurasi", "Presisi_weighted", "Recall_weighted", "F1_weighted"]):
    plt.bar(x + (i - 1.5) * lebar, df_hasil[kol], width=lebar, label=kol)
plt.xticks(x, df_hasil["Model"]); plt.ylim(0, 1); plt.ylabel("Skor")
plt.title("Perbandingan Kinerja: Decision Tree vs Naive Bayes vs SVM")
plt.legend(); plt.tight_layout(); plt.savefig("perbandingan_metrik.png"); plt.close()

open("model_terbaik.txt", "w").write(df_hasil.iloc[0]["Model"])
print(f"\nModel terbaik: {df_hasil.iloc[0]['Model']}")
print("Selesai.")
