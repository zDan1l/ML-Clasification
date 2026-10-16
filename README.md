# Klasifikasi Diabetes — BRFSS 2015

Perbandingan Decision Tree, Naive Bayes, dan SVM dengan pipeline identik:
missing value → hapus duplikat → capping outlier IQR → split 80:20 (stratify, seed 42) →
MinMaxScaler (fit train) → resampling (hanya train) → training → evaluasi.
UI Streamlit memakai model terbaik track SMOTE (Decision Tree, akurasi 0,7373).

| Track | Decision Tree | Naive Bayes | SVM |
|---|---|---|---|
| SMOTE (oversampling) | **0,7373** | 0,5116 | 0,7006* |
| Undersampling | 0,4573 | **0,6387** | 0,5995 |

\*SVM SMOTE memakai train sampling 10rb (full-data ±500rb baris tidak feasible untuk SVC RBF).
Undersampling (train ±11rb) justru membalik peringkat: DT paling terpukul karena kehilangan 97% data kelas sehat.

## Struktur

```
app.py             UI Streamlit (dashboard + prediksi)
task.txt           soal tugas
data/              dataset CSV
src/               script klasifikasi & perbandingan
models/smote|undersampling/   model .pkl, scaler, model_terbaik.txt
outputs/smote|undersampling/  confusion matrix, metrik CSV, grafik
docs/              laporan & arsip
```

## Cara jalan (dari awal)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Satu metode:

```bat
python src/klasifikasi_decision_tree.py
python src/klasifikasi_naive_bayes.py
python src/klasifikasi_svm.py              :: full-data, dokumentasi saja (sangat lama)
```

Perbandingan lengkap:

```bat
python src/perbandingan_smote.py           :: DT + NB + SVM(sampling), track utama
python src/perbandingan_undersampling.py   :: DT + NB + SVM full-train ±11rb baris
```

UI:

```bat
streamlit run app.py
```

Lalu buka `http://localhost:8501`.
