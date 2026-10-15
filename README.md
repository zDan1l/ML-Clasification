# Klasifikasi Diabetes — BRFSS 2015

Perbandingan 3 metode (Decision Tree, Naive Bayes, SVM) dengan pipeline identik:
missing value → hapus duplikat → capping outlier IQR → split 80:20 → MinMaxScaler → SMOTE (train) → training → evaluasi.
UI Streamlit memakai model terbaik (Decision Tree, akurasi 0,7373).

Hasil: Decision Tree 0,7373 | SVM 0,7006 | Naive Bayes 0,5116 (akurasi, detail di `hasil_perbandingan.csv`).

## Cara jalan (dari awal)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Jalankan klasifikasi (masing-masing menyimpan model `.pkl` + gambar confusion matrix):

```bat
python klasifikasi_diabetes_kelompok2.py      :: Decision Tree
python klasifikasi_naive_bayes_kelompok2.py   :: Naive Bayes
python klasifikasi_svm_sampling_kelompok2.py  :: SVM (train sampling 10rb, test full)
```

> `klasifikasi_svm_kelompok2.py` = SVM full-data (±500rb baris pasca-SMOTE), hanya untuk dokumentasi — praktis tidak selesai di laptop (SVC RBF O(n²)).

Jalankan UI:

```bat
streamlit run app.py
```

Lalu buka `http://localhost:8501` (Dashboard Perbandingan + Prediksi Diabetes).
