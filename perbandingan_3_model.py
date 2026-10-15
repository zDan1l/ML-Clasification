# =============================================================
# PERBANDINGAN KLASIFIKASI DIABETES - BRFSS 2015
# Pipeline PERSIS SAMA seperti klasifikasi_diabetes_kelompok2.py,
# hanya klasifikatornya dibedakan:
#   1. Decision Tree
#   2. Naive Bayes (GaussianNB)
#   3. SVM (SVC RBF)
# Alur: Input -> Preprocessing -> Split -> Transformasi
# -> Resampling (SMOTE, hanya train) -> Training -> Testing -> Evaluasi
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
from imblearn.over_sampling import SMOTE
import joblib

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
# 2. PREPROCESSING : missing value, duplikasi, outlier
# (PERSIS seperti kode kelompok2)
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
feature_columns = list(X.columns)

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
# 4. TRANSFORMASI : MinMaxScaler (fit di train saja)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRANSFORMASI (MinMaxScaler)")
print("=" * 60)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Semua fitur diskalakan ke rentang [0, 1].")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

# -------------------------------------------------------------
# 5. RESAMPLING : SMOTE hanya pada data training
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("5. RESAMPLING (Oversampling - SMOTE)")
print("=" * 60)
sebelum = pd.Series(y_train).value_counts().sort_index()
print("Distribusi kelas SEBELUM oversampling:")
print(sebelum.to_string())

plt.figure(figsize=(6, 4))
sns.barplot(x=sebelum.index, y=sebelum.values,
            palette='coolwarm', hue=sebelum.index, legend=False)
plt.xlabel("Kelas Diabetes_012")
plt.ylabel("Jumlah")
plt.title("Distribusi Sebelum SMOTE (Imbalance)")
plt.tight_layout()
plt.savefig("distribusi_sebelum_smote.png")
plt.close()

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

sesudah = pd.Series(y_train_res).value_counts().sort_index()
print("\nDistribusi kelas SESUDAH oversampling:")
print(sesudah.to_string())

plt.figure(figsize=(6, 4))
sns.barplot(x=sesudah.index, y=sesudah.values,
            palette='coolwarm', hue=sesudah.index, legend=False)
plt.xlabel("Kelas Diabetes_012")
plt.ylabel("Jumlah")
plt.title("Distribusi Sesudah SMOTE (Seimbang)")
plt.tight_layout()
plt.savefig("distribusi_sesudah_smote.png")
plt.close()

print(f"\nUkuran train setelah SMOTE: {X_train_res.shape}")

# -------------------------------------------------------------
# 6-8. TRAINING + TESTING + EVALUASI untuk 3 model
# -------------------------------------------------------------
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(random_state=42),  # default kernel='rbf'
}

label_kelas = ["0: Non-Diabetes", "1: Pre-Diabetes", "2: Diabetes"]
hasil = []

for nama, model in models.items():
    print("\n" + "=" * 60)
    print(f"6. TRAINING ({nama})")
    print("=" * 60)
    t0 = time.time()
    model.fit(X_train_res, y_train_res)
    dur = time.time() - t0
    print(f"Model {nama} selesai dilatih dalam {dur:.1f} detik.")

    print(f"7. TESTING ({nama}) + 8. EVALUASI ({nama})")
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nConfusion Matrix ({nama}):\n{cm}")
    print(f"\nAkurasi   : {akurasi:.4f}")
    print(f"Presisi   : {presisi:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"\nLaporan lengkap per kelas ({nama}):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Simpan heatmap confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0", "1", "2"], yticklabels=["0", "1", "2"])
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title(f"Confusion Matrix - {nama}\nAkurasi: {akurasi:.4f}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{nama}.png")
    plt.close()

    # Simpan model
    joblib.dump(model, f"model_{nama}.pkl")

    hasil.append({
        "Model": nama,
        "Akurasi": akurasi,
        "Presisi_weighted": presisi,
        "Recall_weighted": recall,
        "F1_weighted": f1,
        "WaktuLatih_detik": dur,
    })

# Ringkasan perbandingan
df_hasil = pd.DataFrame(hasil).sort_values("Akurasi", ascending=False)
print("\n" + "=" * 60)
print("RINGKASAN PERBANDINGAN")
print("=" * 60)
print(df_hasil.to_string(index=False))
df_hasil.to_csv("hasil_perbandingan.csv", index=False)

# Plot perbandingan metrik
plt.figure(figsize=(8, 5))
x = np.arange(len(df_hasil))
lebar = 0.2
for i, kol in enumerate(["Akurasi", "Presisi_weighted", "Recall_weighted", "F1_weighted"]):
    plt.bar(x + (i - 1.5) * lebar, df_hasil[kol], width=lebar, label=kol)
plt.xticks(x, df_hasil["Model"])
plt.ylim(0, 1)
plt.ylabel("Skor")
plt.title("Perbandingan Kinerja: Decision Tree vs Naive Bayes vs SVM")
plt.legend()
plt.tight_layout()
plt.savefig("perbandingan_metrik.png")
plt.close()

# Tandai model terbaik (akurasi tertinggi) untuk dipakai di Streamlit
terbaik = df_hasil.iloc[0]["Model"]
print(f"\nModel terbaik (akurasi tertinggi): {terbaik}")
with open("model_terbaik.txt", "w") as f:
    f.write(terbaik)
print("Selesai. File: hasil_perbandingan.csv, confusion_matrix_*.png, perbandingan_metrik.png")
