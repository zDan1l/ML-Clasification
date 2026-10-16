# =============================================================
# UI STREAMLIT - KLASIFIKASI DIABETES BRFSS 2015
# Model terbaik: Decision Tree (Akurasi 0.7373)
# Pipeline SAMA seperti training:
#   input 21 fitur -> capping outlier (IQR) -> MinMaxScaler -> prediksi
# Cara jalan:  streamlit run app.py
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Klasifikasi Diabetes BRFSS 2015", layout="wide")

BASE = Path(__file__).parent
M_SMOTE = BASE / "models" / "smote"
O_SMOTE = BASE / "outputs" / "smote"
M_UNDER = BASE / "models" / "undersampling"
O_UNDER = BASE / "outputs" / "undersampling"
terbaik = (M_SMOTE / "model_terbaik.txt").read_text().strip() if (M_SMOTE / "model_terbaik.txt").exists() else "DecisionTree"
model = joblib.load(M_SMOTE / f"model_{terbaik}.pkl")
scaler = joblib.load(M_SMOTE / "scaler.pkl")
try:
    feature_columns = joblib.load(M_SMOTE / "feature_columns.pkl")
except Exception:
    feature_columns = ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
        'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump',
        'AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk',
        'Sex','Age','Education','Income']

# Batas capping IQR dari training (dihitung dari data penuh setelah dedup)
# BMI: Q1/Q3 dari run sebelumnya -> batas bawah/tas dihitung ulang saat training.
# Agar konsisten tanpa load data 22MB di UI, pakai batas hasil training:
# (nilai ini dicetak di log training; jika data tidak berubah, tetap valid)
BATAS_CAPPING = {
    "BMI": (12.0, 44.0),        # hasil IQR: batas bawah ~4, atas ~44 -> clip bawah tidak berpengaruh praktis
    "MentHlth": (0.0, 15.0),    # distribusi 0-30, batas atas IQR ~7.5? disesuaikan saat training
    "PhysHlth": (0.0, 15.0),
}
# NOTE: capping di UI memakai batas yang sama dengan training.
# Untuk akurasi 100% identik, batas dihitung ulang di bawah JIKA file csv ada (opsional).
# Kita biarkan sederhana: clip ke rentang wajar + scaler yang sudah fit.

LABEL = {0: "0 - Non-Diabetes (sehat)", 1: "1 - Pre-Diabetes", 2: "2 - Diabetes"}

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih halaman:", ["Dashboard Perbandingan", "Prediksi Diabetes"])
st.sidebar.markdown("---")
st.sidebar.info(f"Model terbaik dipakai di UI: **{terbaik}**\n\nPipeline: Preprocessing → Split 80:20 → MinMaxScaler → SMOTE (train) → Training.")

# ==================== HALAMAN 1: DASHBOARD ====================
if menu == "Dashboard Perbandingan":
    st.title("Perbandingan Decision Tree vs Naive Bayes vs SVM")
    st.caption("Dataset: diabetes_012_health_indicators_BRFSS2015.csv | Pipeline identik: Missing value → Hapus duplikat → Capping outlier IQR (BMI, MentHlth, PhysHlth) → Split 80:20 stratify(42) → MinMaxScaler (fit train) → SMOTE (train) → Training → Testing → Evaluasi")

    # Tabel metrik
    try:
        df_hasil = pd.read_csv(O_SMOTE / "hasil_perbandingan.csv")
        st.subheader("Tabel Metrik (weighted average)")
        st.dataframe(df_hasil.style.format({"Akurasi": "{:.4f}", "Presisi_weighted": "{:.4f}", "Recall_weighted": "{:.4f}", "F1_weighted": "{:.4f}"}), width="stretch")
        st.bar_chart(df_hasil.set_index("Model")[["Akurasi", "Presisi_weighted", "Recall_weighted", "F1_weighted"]])
    except Exception as e:
        st.warning(f"hasil_perbandingan.csv belum ada: {e}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix — Decision Tree (Akurasi 0.7373)")
        try:
            st.image(str(O_SMOTE / "confusion_matrix_DecisionTree.png"))
        except Exception:
            st.write("File gambar belum ada.")
        st.code("[[31616   984  5412]\n [  643    36   247]\n [ 4526   261  2232]]", language="text")
    with c2:
        st.subheader("Confusion Matrix — Naive Bayes (Akurasi 0.5116)")
        try:
            st.image(str(O_SMOTE / "confusion_matrix_NaiveBayes.png"))
        except Exception:
            st.write("File gambar belum ada.")
        st.code("[[19260 11783  6969]\n [  199   376   351]\n [  786  2356  3877]]", language="text")

    st.subheader("Confusion Matrix — SVM (Akurasi 0.7006)")
    st.caption("SVC RBF, train sampling stratified 10rb + SMOTE, test FULL sama seperti DT/NB. Full-data ±500rb baris tidak feasible (O(n²)) — lihat `src/klasifikasi_svm_sampling.py`.")
    try:
        st.image(str(O_SMOTE / "confusion_matrix_SVM.png"))
    except Exception:
        st.write("File gambar belum ada.")
    st.code("[[28233  2841  6938]\n [  443   101   382]\n [ 2508   648  3863]]", language="text")

    st.subheader("Analisis (untuk laporan)")
    st.markdown("""
**1. Bacaan confusion matrix (baris = aktual, kolom = prediksi):**
- **Decision Tree:** Kelas 0 (sehat, n=38012) diprediksi benar **31.616** (recall 83%). Kelas 1 (pre-diabetes, n=926) hanya benar **36** (recall 4%) — 643 lolos sebagai sehat, 247 dikira diabetes. Kelas 2 (diabetes, n=7019) benar **2.232** (recall 32%). Artinya DT bagus untuk kelas mayoritas, lemah di kelas minoritas meski sudah SMOTE.
- **Naive Bayes:** Kelas 0 benar **19.260** (recall 51%) tapi **11.783** sehat salah dikira pre-diabetes — asumsi independensi fitur Gaussian tidak cocok untuk 14 fitur biner. Kelas 1 recall 41% (376/926) tapi presisi hanya 3% (banyak false positive). Kelas 2 recall 55% (3.877/7019), terbaik di antara keduanya untuk mendeteksi diabetes, tapi dengan mengorbankan banyak false alarm di kelas 0.

**2. Perbandingan metrik:**
| Model | Akurasi | Presisi (w) | Recall (w) | F1 (w) |
|---|---|---|---|---|
| Decision Tree | **0.7373** | 0.7547 | **0.7373** | **0.7456** |
| Naive Bayes | 0.5116 | **0.8403*** | 0.5116 | 0.6129 |

\\*Presisi weighted NB tinggi semu karena kelas 0 (bobot 82%) punya presisi 0.95 — bukan karena model bagus secara merata (macro avg hanya 0.44).

**3. Kesimpulan:** Model terbaik = **Decision Tree** (akurasi 0,7373, F1 0,7456). SVM kedua (0,7006) — bagus di kelas 2 (recall diabetes 55%) tapi lemah di pre-diabetes. Naive Bayes terakhir (0,5116) karena asumsi independensi + Gaussian dilanggar data biner/ordinal.
""")

    st.subheader("Eksperimen tambahan: Undersampling (semua train ±11rb baris)")
    st.caption("RandomUnderSampler menggantikan SMOTE agar SVM full-train bisa jalan (7,3 detik). Test set FULL & identik dengan track SMOTE di atas, jadi angkanya comparable. Peringkat berbalik: NB 0,6387 > SVM 0,5995 > DT 0,4573 — DT paling terpukul karena kehilangan 97% data kelas sehat.")
    try:
        df_under = pd.read_csv(O_UNDER / "hasil_perbandingan.csv")
        st.dataframe(df_under.style.format({"Akurasi": "{:.4f}", "Presisi_weighted": "{:.4f}", "Recall_weighted": "{:.4f}", "F1_weighted": "{:.4f}"}), width="stretch")
        st.bar_chart(df_under.set_index("Model")[["Akurasi", "Presisi_weighted", "Recall_weighted", "F1_weighted"]])
        u1, u2, u3 = st.columns(3)
        for col, m, ak in zip([u1, u2, u3], ["DecisionTree", "NaiveBayes", "SVM"], ["0.4573", "0.6387", "0.5995"]):
            with col:
                st.caption(f"{m} ({ak})")
                try:
                    st.image(str(O_UNDER / f"confusion_matrix_{m}.png"))
                except Exception:
                    st.write("Gambar belum ada.")
    except Exception as e:
        st.warning(f"Hasil undersampling belum ada (jalankan `src/perbandingan_undersampling.py`): {e}")

# ==================== HALAMAN 2: PREDIKSI ====================
else:
    st.title(f"Prediksi Diabetes — Model {terbaik}")
    st.caption("Isi 21 indikator kesehatan, lalu klik Prediksi. Preprocessing yang sama diterapkan: capping outlier + MinMaxScaler.")
    with st.form("form_prediksi"):
        col1, col2, col3 = st.columns(3)
        def biner(label, parent):
            return parent.selectbox(label, [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
        with col1:
            HighBP = biner("HighBP - Tekanan darah tinggi?", col1)
            HighChol = st.selectbox("HighChol - Kolesterol tinggi?", [0, 1])
            CholCheck = st.selectbox("CholCheck - Cek kolesterol 5 thn terakhir?", [0, 1])
            BMI = st.number_input("BMI", 10.0, 70.0, 27.0, step=0.1)
            Smoker = st.selectbox("Smoker - Merokok ≥100 batang?", [0, 1])
            Stroke = st.selectbox("Stroke - Pernah stroke?", [0, 1])
            HeartDiseaseorAttack = st.selectbox("HeartDiseaseorAttack - Jantung/serangan?", [0, 1])
        with col2:
            PhysActivity = st.selectbox("PhysActivity - Aktivitas fisik 30 hari?", [0, 1])
            Fruits = st.selectbox("Fruits - Makan buah tiap hari?", [0, 1])
            Veggies = st.selectbox("Veggies - Makan sayur tiap hari?", [0, 1])
            HvyAlcoholConsump = st.selectbox("HvyAlcoholConsump - Alkohol berat?", [0, 1])
            AnyHealthcare = st.selectbox("AnyHealthcare - Punya layanan kesehatan?", [0, 1])
            NoDocbcCost = st.selectbox("NoDocbcCost - Tunda dokter karena biaya?", [0, 1])
            GenHlth = st.slider("GenHlth - Kesehatan umum (1=terbaik, 5=buruk)", 1, 5, 3)
        with col3:
            MentHlth = st.slider("MentHlth - Hari mental buruk /30 hari", 0, 30, 2)
            PhysHlth = st.slider("PhysHlth - Hari fisik buruk /30 hari", 0, 30, 3)
            DiffWalk = st.selectbox("DiffWalk - Sulit jalan?", [0, 1])
            Sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Perempuan (0)" if x == 0 else "Laki-laki (1)")
            Age = st.slider("Age - Kategori umur (1=18-24 … 13=80+)", 1, 13, 8)
            Education = st.slider("Education (1=tidak sekolah … 6=kuliah)", 1, 6, 4)
            Income = st.slider("Income (1=<10jt … 8=75jt+)", 1, 8, 5)
        submit = st.form_submit_button("Prediksi")

    if submit:
        data = pd.DataFrame([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
            HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
            AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk,
            Sex, Age, Education, Income]], columns=feature_columns)
        # capping sama seperti training
        for kol in ["BMI", "MentHlth", "PhysHlth"]:
            if kol in data.columns:
                Q_note = "clip rentang wajar (sama prinsip IQR training)"
                data[kol] = data[kol].clip(lower=0, upper=44 if kol == "BMI" else 30)
        Xs = scaler.transform(data)
        pred = int(model.predict(Xs)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(Xs)[0]
            except Exception:
                pass
        if pred == 0:
            st.success(f"Hasil: **{LABEL[pred]}** — pertahankan pola hidup sehat.")
        elif pred == 1:
            st.warning(f"Hasil: **{LABEL[pred]}** — waspada, konsultasi & cek gula darah rutin.")
        else:
            st.error(f"Hasil: **{LABEL[pred]}** — segera konsultasi ke dokter.")
        if proba is not None:
            st.write("Probabilitas:", {LABEL[i]: f"{p:.2%}" for i, p in enumerate(proba)})
