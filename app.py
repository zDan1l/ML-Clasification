# =============================================================
# UI STREAMLIT - KLASIFIKASI DIABETES BRFSS 2015
# Tema custom + dashboard perbandingan (2 track) + prediksi interaktif.
# Model terbaik track SMOTE: Decision Tree (akurasi 0,7373)
# Pipeline prediksi SAMA seperti training:
#   input 21 fitur -> capping outlier (IQR) -> MinMaxScaler -> prediksi
# Cara jalan:  streamlit run app.py
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import altair as alt

st.set_page_config(
    page_title="Diabetes Classifier | BRFSS 2015",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
TRACKS = {
    "SMOTE (Oversampling)": {"models": BASE / "models" / "smote",
                             "outputs": BASE / "outputs" / "smote",
                             "desc": "SMOTE menyeimbangkan kelas dengan sampel sintetis"},
    "Undersampling": {"models": BASE / "models" / "undersampling",
                      "outputs": BASE / "outputs" / "undersampling",
                      "desc": "RandomUnderSampler menyusutkan train ke ±11rb baris"},
}
NAMA_TAMPIL = {"DecisionTree": "Decision Tree", "NaiveBayes": "Naive Bayes", "SVM": "SVM"}
LABEL = {0: "Non-Diabetes (sehat)", 1: "Pre-Diabetes", 2: "Diabetes"}
WARNA_KELAS = {0: "#16A34A", 1: "#F59E0B", 2: "#DC2626"}

# -------------------------------------------------------------
# TEMA / CSS
# -------------------------------------------------------------
st.markdown("""
<style>
:root { --brand:#0F766E; --brand2:#0EA5E9; --ink:#0F172A; --muted:#64748B; }
.stApp { background:
    radial-gradient(1200px 600px at 10% -10%, #E0F2FE 0%, rgba(255,255,255,0) 55%),
    radial-gradient(1000px 500px at 100% 0%, #CCFBF1 0%, rgba(255,255,255,0) 50%),
    #F8FAFC; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
h1, h2, h3 { color: var(--ink); letter-spacing:-.02em; }

/* Hero */
.hero { padding: 26px 30px; border-radius: 20px; color:#fff;
    background: linear-gradient(120deg, #0F766E 0%, #0E7490 55%, #2563EB 100%);
    box-shadow: 0 18px 40px -18px rgba(15,118,110,.65); margin-bottom: 8px; }
.hero h1 { color:#fff; margin:0; font-size: 1.9rem; line-height:1.2; }
.hero p { color:#E0F2FE; margin:.45rem 0 0; font-size:.95rem; }
.hero .badge { display:inline-block; background:rgba(255,255,255,.18);
    border:1px solid rgba(255,255,255,.35); padding:3px 11px; border-radius:999px;
    font-size:.76rem; font-weight:600; margin-right:7px; margin-top:11px; }

/* Kartu */
.card { background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:18px 20px;
    box-shadow: 0 6px 18px -12px rgba(15,23,42,.28); }
.card-title { font-weight:700; color:var(--ink); font-size:1rem; margin-bottom:2px; }
.card-sub { color:var(--muted); font-size:.82rem; }

/* KPI */
.kpi { background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:16px 18px;
    box-shadow: 0 6px 18px -14px rgba(15,23,42,.30); height:100%; }
.kpi .lbl { color:var(--muted); font-size:.78rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.05em; }
.kpi .val { color:var(--brand); font-size:1.7rem; font-weight:800; line-height:1.25; }
.kpi .sub { color:var(--muted); font-size:.8rem; }

/* Hasil prediksi */
.result { border-radius:18px; padding:22px 24px; color:#fff; }
.result h2 { color:#fff; margin:0; }
.result .big { font-size:1.35rem; font-weight:800; }
.disclaimer { color:var(--muted); font-size:.8rem; margin-top:10px; }

/* Sidebar */
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E2E8F0; }
.side-brand { font-weight:800; font-size:1.12rem; color:var(--brand); }
.side-sub { color:var(--muted); font-size:.8rem; margin-top:-4px; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# LOADER (cached)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_track(nama):
    info = TRACKS[nama]
    mdir = info["models"]
    terbaik_file = mdir / "model_terbaik.txt"
    terbaik = terbaik_file.read_text().strip() if terbaik_file.exists() else "DecisionTree"
    scaler = joblib.load(mdir / "scaler.pkl") if (mdir / "scaler.pkl").exists() else None
    try:
        cols = joblib.load(mdir / "feature_columns.pkl")
    except Exception:
        cols = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    models = {}
    for key in NAMA_TAMPIL:
        f = mdir / f"model_{key}.pkl"
        if f.exists():
            models[key] = joblib.load(f)
    return {"terbaik": terbaik, "scaler": scaler, "cols": cols, "models": models}


@st.cache_data(show_spinner=False)
def load_hasil(output_dir):
    f = Path(output_dir) / "hasil_perbandingan.csv"
    return pd.read_csv(f) if f.exists() else None


def indonesiakan(df):
    d = df.rename(columns={"Presisi_weighted": "Presisi", "Recall_weighted": "Recall",
                           "F1_weighted": "F1-Score"}).copy()
    d["Model"] = d["Model"].map(lambda m: NAMA_TAMPIL.get(m, m))
    return d


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-brand">🩺 Diabetes Classifier</div>'
                '<div class="side-sub">BRFSS 2015 · Kelompok 2</div>', unsafe_allow_html=True)
    st.markdown("---")
    halaman = st.radio("Navigasi", ["📊 Dashboard", "🔮 Prediksi", "ℹ️ Tentang"],
                       label_visibility="collapsed")
    st.markdown("---")
    bundle_main = load_track("SMOTE (Oversampling)")
    st.markdown(
        f'<div class="card"><div class="card-title">Model terbaik</div>'
        f'<div class="card-sub">Track SMOTE</div>'
        f'<div style="font-size:1.15rem;font-weight:800;color:#0F766E;margin-top:6px">'
        f'{NAMA_TAMPIL.get(bundle_main["terbaik"], bundle_main["terbaik"])}</div></div>',
        unsafe_allow_html=True)
    st.caption("Alur: Preprocessing → Split 80:20 → MinMaxScaler → Resampling (train) → Training → Evaluasi")


# -------------------------------------------------------------
# HALAMAN 1 : DASHBOARD
# -------------------------------------------------------------
def halaman_dashboard():
    st.markdown(
        '<div class="hero"><h1>Perbandingan Decision Tree, Naive Bayes & SVM</h1>'
        '<p>Klasifikasi status diabetes pada Diabetes Health Indicators Dataset (BRFSS 2015, CDC) '
        'dengan pipeline identik dan dua skenario resampling.</p>'
        '<span class="badge">253.680 baris</span>'
        '<span class="badge">22 kolom</span><span class="badge">3 kelas</span>'
        '<span class="badge">Test set 45.957 baris</span></div>', unsafe_allow_html=True)
    st.write("")

    tab_smote, tab_under = st.tabs(["🔵 Track SMOTE (Oversampling)",
                                    "🟢 Track Undersampling"])
    with tab_smote:
        panel_track("SMOTE (Oversampling)", "🔵")
    with tab_under:
        panel_track("Undersampling", "🟢")


def panel_track(nama, emoji):
    info = TRACKS[nama]
    df = load_hasil(info["outputs"])
    if df is None:
        st.warning("`hasil_perbandingan.csv` belum ada. Jalankan script perbandingan terkait dulu.")
        return
    df = df.sort_values("Akurasi", ascending=False).reset_index(drop=True)

    # KPI cards
    juara = df.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Model Terbaik", NAMA_TAMPIL.get(juara["Model"], juara["Model"]), info["desc"])
    kpi(k2, "Akurasi Tertinggi", f"{juara['Akurasi']:.2%}", "weighted average")
    kpi(k3, "F1-Score Terbaik", f"{juara['F1_weighted']:.4f}", "weighted average")
    kpi(k4, "Model Dibandingkan", f"{len(df)}", "Decision Tree · NB · SVM")
    st.write("")

    # Tabel + grafik
    c_tab, c_chart = st.columns([1.05, 1.0], gap="large")
    with c_tab:
        st.markdown("##### 📋 Tabel Metrik")
        d = indonesiakan(df)
        st.dataframe(
            d.style.format({"Akurasi": "{:.4f}", "Presisi": "{:.4f}",
                            "Recall": "{:.4f}", "F1-Score": "{:.4f}"})
             .background_gradient(cmap="Greens", subset=["Akurasi", "F1-Score"]),
            width="stretch", hide_index=True)
        terbaik_nama = NAMA_TAMPIL.get(juara["Model"], juara["Model"])
        st.success(f"**{terbaik_nama}** unggul pada track ini "
                   f"(akurasi {juara['Akurasi']:.2%}, F1 {juara['F1_weighted']:.4f}).")
    with c_chart:
        st.markdown("##### 📈 Perbandingan Metrik")
        st.altair_chart(grafik_metrik(indonesiakan(df)), width="stretch")

    # Confusion matrix
    st.markdown("##### 🧩 Confusion Matrix")
    urut = [m for m in ["DecisionTree", "NaiveBayes", "SVM"] if m in set(df["Model"])]
    kol = st.columns(len(urut))
    for c, m in zip(kol, urut):
        f = info["outputs"] / f"confusion_matrix_{m}.png"
        akurasi_m = df.loc[df["Model"] == m, "Akurasi"]
        akurasi_m = f"{akurasi_m.iloc[0]:.4f}" if len(akurasi_m) else "-"
        with c:
            st.markdown(f"**{NAMA_TAMPIL[m]}** &nbsp;·&nbsp; akurasi `{akurasi_m}`")
            if f.exists():
                st.image(str(f), width="stretch")
            else:
                st.info("Gambar belum tersedia.")
    if nama.startswith("SMOTE"):
        st.caption("ℹ️ SVM di track SMOTE dilatih dengan sampel stratified 10rb (SVC RBF O(n²) "
                   "tidak feasible pada ±500rb baris); test set tetap penuh & identik dengan model lain.")

    # Analisis
    st.markdown("##### 🔎 Analisis")
    if nama.startswith("SMOTE"):
        st.markdown("""
- **Decision Tree** paling seimbang: kelas sehat dikenali sangat baik (recall 83%), tetapi kelas
  pradiabetes hampir selalu salah (hanya 36 dari 926 benar) karena datanya sangat sedikit.
- **Naive Bayes** akurasinya terendah (51,16%). Presisi weighted-nya terlihat tinggi (0,84) tapi itu
  semu — kelas sehat yang berbobot 82% memang presisi 0,95. Asumsi independensi & Gaussian dilanggar
  oleh dominasi 14 fitur biner. Meski begitu, recall diabetes-nya (55%) adalah yang tertinggi.
- **SVM** berada di tengah (70,06%): cukup baik mengenali diabetes (recall 55%) namun masih lemah
  pada kelas pradiabetes.
""")
        st.info("**Kesimpulan:** model terbaik track SMOTE = **Decision Tree** "
                "(akurasi 73,73%, F1 0,7456).")
    else:
        st.markdown("""
- **Naive Bayes** justru menjadi terbaik (63,87%) — ia butuh sedikit data, sehingga tahan ketika
  train menyusut menjadi ±11rb baris.
- **SVM** mengikuti di posisi kedua (59,95%) dan hanya butuh 7,3 detik untuk dilatih (vs. tidak
  selesai pada track SMOTE).
- **Decision Tree** paling terpukul (45,73%): kehilangan 97% data kelas sehat membuatnya gagal
  mengenali kelas mayoritas (recall kelas sehat hanya 47%).
""")
        st.warning("**Temuan penting:** pilihan teknik resampling membalik pemenang — "
                   "Decision Tree unggul pada SMOTE, sedangkan Naive Bayes unggul pada undersampling.")


def kpi(col, label, value, sub):
    with col:
        st.markdown(f'<div class="kpi"><div class="lbl">{label}</div>'
                    f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
                    unsafe_allow_html=True)


def grafik_metrik(df):
    melt = df.melt(id_vars="Model", var_name="Metrik", value_name="Skor")
    order = ["Akurasi", "Presisi", "Recall", "F1-Score"]
    return (
        alt.Chart(melt).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Model:N", axis=alt.Axis(title=None, labelAngle=0)),
            xOffset=alt.XOffset("Metrik:N", sort=order),
            y=alt.Y("Skor:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(title="Skor")),
            color=alt.Color("Metrik:N", sort=order,
                            scale=alt.Scale(range=["#0F766E", "#0EA5E9", "#F59E0B", "#8B5CF6"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "Metrik", alt.Tooltip("Skor:Q", format=".4f")],
        ).properties(height=330)
    )


# -------------------------------------------------------------
# HALAMAN 2 : PREDIKSI
# -------------------------------------------------------------
def halaman_prediksi():
    st.markdown(
        '<div class="hero"><h1>🔮 Prediksi Status Diabetes</h1>'
        '<p>Masukkan 21 indikator kesehatan. Data diproses dengan pipeline yang sama seperti '
        'saat pelatihan: capping outlier (IQR) → MinMaxScaler → model.</p></div>',
        unsafe_allow_html=True)
    st.write("")

    # Pilih track & model
    c1, c2 = st.columns([2, 1])
    with c1:
        track = st.selectbox("Skenario resampling (track)", list(TRACKS.keys()))
    bundle = load_track(track)
    tersedia = list(bundle["models"].keys())
    with c2:
        default_idx = tersedia.index(bundle["terbaik"]) if bundle["terbaik"] in tersedia else 0
        pilih = st.selectbox("Model", tersedia, index=default_idx,
                             format_func=lambda m: NAMA_TAMPIL.get(m, m))

    acc = None
    df_h = load_hasil(TRACKS[track]["outputs"])
    if df_h is not None:
        r = df_h.loc[df_h["Model"] == pilih, "Akurasi"]
        acc = r.iloc[0] if len(r) else None
    st.caption(f"Track **{track}** · model **{NAMA_TAMPIL.get(pilih, pilih)}**"
               + (f" · akurasi {acc:.4f}" if acc is not None else "")
               + f" · {TRACKS[track]['desc']}.")
    st.write("")

    with st.form("form_prediksi"):
        st.markdown("#### 🫀 Kondisi Metabolik & Jantung")
        a1, a2, a3 = st.columns(3)
        with a1:
            HighBP = ya_tidak("Tekanan darah tinggi? (HighBP)", a1)
            HighChol = ya_tidak("Kolesterol tinggi? (HighChol)", a1)
        with a2:
            CholCheck = ya_tidak("Cek kolesterol 5 thn terakhir? (CholCheck)", a2)
            Stroke = ya_tidak("Pernah stroke? (Stroke)", a2)
        with a3:
            HeartDiseaseorAttack = ya_tidak("Penyakit jantung / serangan? (HeartDisease)", a3)
            BMI = st.number_input("BMI (indeks massa tubuh)", 10.0, 70.0, 27.0, step=0.1,
                                  help="Berat (kg) dibagi tinggi (m) kuadrat. Normal 18,5–24,9.")

        st.markdown("#### 🏃 Gaya Hidup & Nutrisi")
        b1, b2, b3 = st.columns(3)
        with b1:
            Smoker = ya_tidak("Merokok ≥100 batang seumur hidup? (Smoker)", b1)
            PhysActivity = ya_tidak("Aktivitas fisik 30 hari terakhir? (PhysActivity)", b1)
        with b2:
            Fruits = ya_tidak("Makan buah setiap hari? (Fruits)", b2)
            Veggies = ya_tidak("Makan sayur setiap hari? (Veggies)", b2)
        with b3:
            HvyAlcoholConsump = ya_tidak("Konsumsi alkohol berat? (HvyAlcohol)", b3)

        st.markdown("#### 🏥 Akses & Kesehatan Umum")
        d1, d2, d3 = st.columns(3)
        with d1:
            AnyHealthcare = ya_tidak("Punya layanan kesehatan? (AnyHealthcare)", d1)
            NoDocbcCost = ya_tidak("Tunda ke dokter karena biaya? (NoDocbcCost)", d1)
        with d2:
            GenHlth = st.slider("Kesehatan umum (GenHlth)", 1, 5, 3,
                                help="1 = sangat baik … 5 = buruk")
            DiffWalk = ya_tidak("Sulit berjalan / naik tangga? (DiffWalk)", d2)
        with d3:
            MentHlth = st.slider("Hari kesehatan mental buruk /30 hari (MentHlth)", 0, 30, 2)
            PhysHlth = st.slider("Hari kesehatan fisik buruk /30 hari (PhysHlth)", 0, 30, 3)

        st.markdown("#### 👤 Demografi")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            Sex = e1.selectbox("Jenis kelamin (Sex)", [0, 1],
                               format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
        with e2:
            Age = e2.slider("Kelompok umur (Age)", 1, 13, 8,
                            help="1 = 18–24 … 13 = 80+")
        with e3:
            Education = e3.slider("Pendidikan (Education)", 1, 6, 4,
                                  help="1 = tidak sekolah … 6 = tamat kuliah")
        with e4:
            Income = e4.slider("Pendapatan (Income)", 1, 8, 5,
                               help="1 = <10 juta … 8 = ≥75 juta per tahun")

        st.write("")
        submit = st.form_submit_button("🔮  Prediksi Sekarang", width="stretch", type="primary")

    if submit:
        data = pd.DataFrame([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
            HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
            AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk,
            Sex, Age, Education, Income]], columns=bundle["cols"])
        for kol, atas in [("BMI", 44.0), ("MentHlth", 30.0), ("PhysHlth", 30.0)]:
            if kol in data.columns:
                data[kol] = data[kol].clip(lower=0, upper=atas)
        Xs = bundle["scaler"].transform(data) if bundle["scaler"] is not None else data.values
        model = bundle["models"][pilih]
        pred = int(model.predict(Xs)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(Xs)[0]
            except Exception:
                proba = None

        st.write("")
        if pred == 0:
            warna, ikon, pesan = "#16A34A", "✅", "Pertahankan pola hidup sehat Anda."
        elif pred == 1:
            warna, ikon, pesan = "#F59E0B", "⚠️", "Waspada — konsultasikan & cek gula darah rutin."
        else:
            warna, ikon, pesan = "#DC2626", "🚨", "Segera konsultasikan ke dokter."
        st.markdown(
            f'<div class="result" style="background:linear-gradient(120deg,{warna},#0F172A)">'
            f'<div style="opacity:.85;font-weight:600">Hasil prediksi · {NAMA_TAMPIL.get(pilih,pilih)}</div>'
            f'<h2>{ikon} {LABEL[pred]}</h2><div>{pesan}</div></div>', unsafe_allow_html=True)

        if proba is not None:
            st.markdown("##### Probabilitas per kelas")
            for i, p in enumerate(proba):
                c1, c2 = st.columns([1, 3])
                c1.markdown(f"**{LABEL[i]}**")
                c2.progress(min(max(float(p), 0.0), 1.0), text=f"{p:.1%}")

        st.markdown('<div class="disclaimer">⚠️ Hasil ini merupakan keluaran model machine learning '
                    'dan <b>bukan diagnosis medis</b>. Selalu konsultasikan kondisi Anda dengan '
                    'tenaga kesehatan profesional.</div>', unsafe_allow_html=True)


def ya_tidak(label, parent):
    return parent.selectbox(label, [0, 1],
                            format_func=lambda x: "Ya" if x == 1 else "Tidak")


# -------------------------------------------------------------
# HALAMAN 3 : TENTANG
# -------------------------------------------------------------
def halaman_tentang():
    st.markdown(
        '<div class="hero"><h1>ℹ️ Tentang Proyek</h1>'
        '<p>Ringkasan dataset, pipeline, dan hasil perbandingan ketiga algoritma.</p></div>',
        unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### 📦 Dataset")
        st.markdown("""
- **Nama:** Diabetes Health Indicators (BRFSS 2015), CDC.
- **Ukuran:** 253.680 baris × 22 kolom (21 fitur + target).
- **Target:** `Diabetes_012` → 0 tidak diabetes, 1 pradiabetes, 2 diabetes.
- **Distribusi awal:** 84,24% / 1,83% / 13,93% (sangat tidak seimbang).
""")
        st.markdown("##### ⚙️ Pipeline")
        st.markdown("""
1. Input data (CSV)
2. Preprocessing: missing value, hapus duplikat (23.899 baris), capping outlier IQR (BMI, MentHlth, PhysHlth)
3. Split 80:20 (`stratify=y`, `random_state=42`)
4. Transformasi MinMaxScaler (fit hanya di train)
5. Resampling hanya pada train (SMOTE atau undersampling)
6. Training → Testing → Evaluasi
""")
    with c2:
        st.markdown("##### 🏆 Hasil Perbandingan")
        for nama in TRACKS:
            df = load_hasil(TRACKS[nama]["outputs"])
            if df is None:
                continue
            st.markdown(f"**{nama}**")
            st.dataframe(indonesiakan(df.sort_values("Akurasi", ascending=False))
                         .style.format({"Akurasi": "{:.4f}", "Presisi": "{:.4f}",
                                        "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
                         width="stretch", hide_index=True)
        st.markdown("##### 🧰 Teknologi")
        st.markdown("Python · scikit-learn · imbalanced-learn · pandas · matplotlib/seaborn · "
                    "Streamlit. Kode tersedia di folder `src/`.")


# -------------------------------------------------------------
# ROUTER
# -------------------------------------------------------------
if halaman == "📊 Dashboard":
    halaman_dashboard()
elif halaman == "🔮 Prediksi":
    halaman_prediksi()
else:
    halaman_tentang()

st.markdown('<div class="disclaimer" style="text-align:center;margin-top:26px">'
            'UTS Praktikum Pembelajaran Mesin · Kelompok 2 · Universitas Airlangga</div>',
            unsafe_allow_html=True)
