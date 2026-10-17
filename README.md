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

---

## Cara paling mudah menjalankan UI

**Klik dua kali file `run_app.bat`.** Selesai — browser terbuka otomatis di
`http://localhost:8501`. Launcher ini otomatis memakai Streamlit dari `.venv`,
jadi kamu tidak perlu mengaktifkan virtual environment dulu.

Jika `run_app.bat` belum ada / ingin manual, jalankan dari terminal (PowerShell / CMD) di folder proyek:

```bat
.venv\Scripts\streamlit.exe run app.py
```

> **Jangan** pakai perintah `streamlit run app.py` begitu saja. Di komputer ini
> perintah `streamlit` menunjuk ke **Python 3.13** yang tidak punya `scikit-learn`,
> `joblib`, dan `imblearn`, sehingga app langsung gagal dengan
> `ModuleNotFoundError: No module named 'joblib'`.
> Selalu panggil versi `.venv` seperti di atas.

Untuk menghentikan server: tekan `Ctrl+C` di jendela terminal.

> Saat **pertama kali** dijalankan, Streamlit menampilkan permintaan email
> (`Email:`). Cukup tekan **Enter** (biarkan kosong) — hanya muncul sekali.

---

## Setup pertama kali (sekali saja)

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Setelah ini, cukup pakai `run_app.bat` untuk membuka UI.

## Menjalankan training / perbandingan

Satu metode:

```bat
.venv\Scripts\python src\klasifikasi_decision_tree.py
.venv\Scripts\python src\klasifikasi_naive_bayes.py
.venv\Scripts\python src\klasifikasi_svm.py              :: full-data, dokumentasi saja (sangat lama)
```

Perbandingan lengkap:

```bat
.venv\Scripts\python src\perbandingan_smote.py           :: DT + NB + SVM(sampling), track utama
.venv\Scripts\python src\perbandingan_undersampling.py   :: DT + NB + SVM full-train ±11rb baris
```

## Struktur

```
app.py             UI Streamlit (Dashboard · Prediksi · Tentang)
run_app.bat        launcher UI sekali-klik
requirements.txt   daftar dependensi
.streamlit/        config.toml — konfigurasi tema UI
data/              dataset CSV BRFSS 2015
src/               script klasifikasi & perbandingan
models/            smote/ & undersampling/ — model .pkl, scaler, model_terbaik.txt
outputs/           smote/ & undersampling/ — confusion matrix, metrik CSV, grafik
```

## Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `ModuleNotFoundError: No module named 'joblib'` / `sklearn` | pakai `streamlit` Python 3.13 global | pakai `run_app.bat` atau `.venv\Scripts\streamlit.exe run app.py` |
| `'streamlit' is not recognized` | `.venv` belum dibuat | jalankan Setup pertama kali di atas |
| `ModuleNotFoundError: No module named 'streamlit'` | `.venv` belum di-`pip install` | `.venv\Scripts\python -m pip install -r requirements.txt` |
| Port 8501 sudah dipakai | server lain masih jalan | `run_app.bat` sudah pakai 8501; hentikan server lama atau ganti port: `.venv\Scripts\streamlit.exe run app.py --server.port 8502` |
| UI terbuka tapi gambar/angka kosong | belum menjalankan script perbandingan | jalankan `src/perbandingan_smote.py` dan `src/perbandingan_undersampling.py` |
