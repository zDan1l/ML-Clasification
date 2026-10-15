# Evaluasi ulang model tersimpan (DT & NB) dengan pipeline identik
# untuk menghasilkan tabel perbandingan + confusion matrix numbers.
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import numpy as np

FILE = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(FILE)
df = df.drop_duplicates().reset_index(drop=True)
for kol in ["BMI", "MentHlth", "PhysHlth"]:
    Q1 = df[kol].quantile(0.25)
    Q3 = df[kol].quantile(0.75)
    IQR = Q3 - Q1
    df[kol] = df[kol].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

X = df.drop(columns=["Diabetes_012"])
y = df["Diabetes_012"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)

hasil = []
for nama in ["DecisionTree", "NaiveBayes"]:
    model = joblib.load(f"model_{nama}.pkl")
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    ak = accuracy_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rc = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"===== {nama} =====")
    print("Confusion Matrix:\n", cm)
    print(f"Akurasi: {ak:.4f} | Presisi: {pr:.4f} | Recall: {rc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    # heatmap ulang (pastikan file ada)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0","1","2"], yticklabels=["0","1","2"])
    plt.xlabel("Prediksi"); plt.ylabel("Aktual")
    plt.title(f"Confusion Matrix - {nama}\nAkurasi: {ak:.4f}")
    plt.tight_layout(); plt.savefig(f"confusion_matrix_{nama}.png"); plt.close()
    hasil.append({"Model": nama, "Akurasi": ak, "Presisi_weighted": pr, "Recall_weighted": rc, "F1_weighted": f1})

df_hasil = pd.DataFrame(hasil).sort_values("Akurasi", ascending=False)
print("\nRINGKASAN:\n", df_hasil.to_string(index=False))
df_hasil.to_csv("hasil_perbandingan.csv", index=False)

plt.figure(figsize=(8,5))
x = np.arange(len(df_hasil)); lebar = 0.2
for i, kol in enumerate(["Akurasi","Presisi_weighted","Recall_weighted","F1_weighted"]):
    plt.bar(x + (i-1.5)*lebar, df_hasil[kol], width=lebar, label=kol)
plt.xticks(x, df_hasil["Model"]); plt.ylim(0,1); plt.ylabel("Skor")
plt.title("Perbandingan Kinerja: Decision Tree vs Naive Bayes\n(SVM menyusul - training full-data belum selesai)")
plt.legend(); plt.tight_layout(); plt.savefig("perbandingan_metrik.png"); plt.close()

terbaik = df_hasil.iloc[0]["Model"]
open("model_terbaik.txt","w").write(terbaik)
print(f"\nModel terbaik sementara: {terbaik}")
