# =============================================================
# GENERATOR LAPORAN DOCX - Perbandingan 3 Metode Klasifikasi Diabetes
# Format mengikuti referensi docs/Laporan_Klasifikasi_Diabetes.docx:
# Times New Roman 12 justify, Heading 1 (13), Heading 2 (12), A4.
# Cara jalan:  .venv/Scripts/python src/buat_laporan.py
# Output: docs/Laporan_Perbandingan_3_Model.docx
# =============================================================
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = Path(__file__).resolve().parent.parent
OUT_SMOTE = BASE / "outputs" / "smote"
OUT_UNDER = BASE / "outputs" / "undersampling"
DOC = BASE / "docs" / "Laporan_Perbandingan_3_Model.docx"

FONT = "Times New Roman"
GAMBAR = []  # (nomor, judul) untuk DAFTAR GAMBAR
_no_gambar = 0

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Cm(2.54)

normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
for h, size in [("Heading 1", 13), ("Heading 2", 12)]:
    s = doc.styles[h]
    s.font.name = FONT
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = RGBColor(0, 0, 0)
    s.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
h5 = doc.styles["Heading 5"]
h5.font.size = Pt(11)
h5.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER


def body(teks):
    p = doc.add_paragraph(teks)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def center(teks, size=12, bold=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(teks)
    r.bold = bold
    r.font.size = Pt(size)
    r.font.name = FONT
    return p


def h1(t):
    doc.add_page_break()
    return doc.add_heading(t, level=1)


def h2(t):
    return doc.add_heading(t, level=2)


def caption(t):
    return doc.add_heading(t, level=5)


def kode(*lines):
    for ln in lines:
        p = doc.add_paragraph()
        r = p.add_run(ln if ln else " ")
        r.font.name = "Courier New"
        r.font.size = Pt(10)


def gambar(path, judul):
    # Placeholder saja: gambar dimasukkan manual di Word.
    # Nomor + caption + DAFTAR GAMBAR tetap dibuat otomatis.
    global _no_gambar
    _no_gambar += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"[TEMPAT GAMBAR {_no_gambar}: {judul}]")
    r.italic = True
    r.font.name = FONT
    r.font.size = Pt(12)
    caption(f"Gambar {_no_gambar}. {judul}")
    GAMBAR.append((_no_gambar, judul))


def tabel(header, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htxt in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        r = c.paragraphs[0].add_run(htxt)
        r.bold = True
        r.font.name = FONT
        r.font.size = Pt(12)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = t.rows[i].cells[j]
            c.text = ""
            r = c.paragraphs[0].add_run(val)
            r.font.name = FONT
            r.font.size = Pt(12)
    doc.add_paragraph()
    return t


def no_border(table):
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        borders = table._tbl.tblPr.first_child_found_in("w:tblBorders")
        if borders is None:
            borders = OxmlElement("w:tblBorders")
            table._tbl.tblPr.append(borders)
        el = OxmlElement(tag)
        el.set(qn("w:val"), "nil")
        borders.append(el)


# ================= COVER =================
center("LAPORAN UTS MACHINE LEARNING", 15, True)
center("PERBANDINGAN ALGORITMA DECISION TREE, NAIVE BAYES, DAN SVM "
       "UNTUK KLASIFIKASI PENYAKIT DIABETES DENGAN PENANGANAN DATA "
       "TIDAK SEIMBANG (SMOTE DAN UNDERSAMPLING)", 13, True)
doc.add_paragraph(); doc.add_paragraph()
center("Kelompok 2", 11, True)
center("Disusun Oleh :", 11, True)
doc.add_paragraph()
anggota = [("Ziyaa Danil M", "434241034"), ("M. Ainul Yaqin", "434241036"),
           ("Vicky Candra Arifian", "434241038")]
tc = doc.add_table(rows=3, cols=2)
tc.alignment = WD_TABLE_ALIGNMENT.CENTER
no_border(tc)
for i, (nama, nim) in enumerate(anggota):
    for j, val in enumerate((nama, nim)):
        c = tc.rows[i].cells[j]
        c.text = ""
        r = c.paragraphs[0].add_run(val)
        r.bold = True
        r.font.name = FONT
        r.font.size = Pt(11)
doc.add_paragraph()
center("PROGRAM STUDI D4 TEKNIK INFORMATIKA", 14, True)
center("UNIVERSITAS AIRLANGGA", 14, True)
center("SURABAYA", 14, True)
center("2026", 14, True)
doc.add_page_break()
center("DAFTAR ISI", 13, True)
doc.add_page_break()
center("DAFTAR GAMBAR", 13, True)
for _n, _t in []:
    pass

# ================= BAB I =================
h1("BAB I  PENDAHULUAN")
h2("1.1  Latar Belakang")
body("Diabetes melitus merupakan suatu gangguan metabolisme yang ditandai dengan hiperglikemia yang diakibatkan "
     "oleh ketidakmampuan tubuh untuk mensekresi dan merespons insulin (Olisah et al., 2022). Diabetes melitus "
     "menjadi salah satu penyakit kronis penyebab kematian terbesar kelima di seluruh dunia dengan total 1,6 juta "
     "kematian (Olisah et al., 2022). Kondisi serupa juga terjadi di Indonesia. Berdasarkan data International "
     "Diabetes Federation (2023), prevalensi diabetes pada populasi dewasa di Indonesia mencapai 11,3%, dengan "
     "total kasus mencapai 20.426.400 dari 185.217.400 populasi dewasa. Angka ini menempatkan Indonesia sebagai "
     "salah satu negara dengan beban diabetes tertinggi di dunia. Oleh karena itu, deteksi dini terhadap risiko "
     "diabetes sangat penting agar penanganan dapat dilakukan lebih cepat dan komplikasi dapat dicegah.")
body("Seiring berkembangnya teknologi, data indikator kesehatan masyarakat dalam jumlah besar dapat dimanfaatkan "
     "untuk membangun model prediksi menggunakan teknik data mining, khususnya klasifikasi. Pendekatan konvensional "
     "berbasis aturan medis sering kali kurang mampu menangani data kesehatan yang kompleks dan berdimensi tinggi, "
     "sehingga diperlukan pendekatan berbasis data.")
body("Salah satu tantangan utama pada data kesehatan adalah kondisi data yang tidak seimbang (imbalanced), yaitu "
     "jumlah data pada kelas tertentu jauh lebih sedikit dibanding kelas lainnya. Kondisi ini dapat menyebabkan model "
     "cenderung memihak kelas mayoritas, sehingga metrik akurasi menjadi tinggi namun recall pada kelas minoritas "
     "rendah. Hal ini berimplikasi pada gagalnya deteksi dini pasien diabetes. Untuk mengatasi permasalahan tersebut, "
     "laporan ini menerapkan dua teknik penyeimbangan data, yaitu Synthetic Minority Over-sampling Technique (SMOTE) "
     "dan Random Undersampling, yang dikombinasikan dengan tiga algoritma klasifikasi: Decision Tree, Naive Bayes, "
     "dan Support Vector Machine (SVM). Ketiga model dibandingkan menggunakan Confusion Matrix, Akurasi, Presisi, "
     "Recall, dan F1-Score, kemudian model terbaik diimplementasikan ke dalam aplikasi web Streamlit untuk prediksi "
     "interaktif.")
h2("1.2  Rumusan Masalah")
body("Bagaimana tahapan pra-pemrosesan data yang diperlukan sebelum data digunakan untuk klasifikasi?")
body("Bagaimana cara menangani data yang tidak seimbang pada dataset diabetes dengan SMOTE dan undersampling?")
body("Bagaimana perbandingan performa algoritma Decision Tree, Naive Bayes, dan SVM dalam mengklasifikasikan status diabetes?")
body("Bagaimana mengimplementasikan model terbaik ke dalam aplikasi web untuk prediksi diabetes?")
h2("1.3  Tujuan")
body("Membangun model klasifikasi status diabetes menggunakan algoritma Decision Tree, Naive Bayes, dan SVM.")
body("Menerapkan pra-pemrosesan data, transformasi (MinMaxScaler), dan penyeimbangan data (SMOTE serta undersampling).")
body("Membandingkan performa ketiga model menggunakan Confusion Matrix, Akurasi, Presisi, Recall, dan F1-Score.")
body("Mengimplementasikan model dengan akurasi tertinggi ke dalam aplikasi web Streamlit.")

# ================= BAB II =================
h1("BAB II  LANDASAN TEORI")
h2("2.1  Data Mining dan Klasifikasi")
body("Data mining adalah proses menggali informasi atau pola yang bermanfaat dari sekumpulan data berukuran besar. "
     "Data mining telah banyak diterapkan di sektor kesehatan dan medis, termasuk untuk perencanaan dan manajemen "
     "keperawatan (Attari et al., 2026). Salah satu tugas utama dalam data mining adalah klasifikasi, yaitu proses "
     "membangun model untuk memprediksi kelas atau kategori dari suatu data berdasarkan fitur-fiturnya. Klasifikasi "
     "termasuk dalam pembelajaran terarah (supervised learning) karena model dilatih menggunakan data yang telah "
     "memiliki label.")
h2("2.2  Algoritma Decision Tree")
body("Decision Tree (pohon keputusan) adalah algoritma klasifikasi yang berbentuk struktur pohon, terdiri atas simpul "
     "akar (root node), simpul cabang (internal node), dan simpul daun (leaf node). Setiap internal node "
     "merepresentasikan pengujian terhadap suatu fitur, cabang merepresentasikan hasil pengujian, dan leaf node "
     "merepresentasikan kelas akhir (Sharma & Kumar, 2026).")
body("Dalam membangun pohon, algoritma memilih fitur pemisah terbaik berdasarkan ukuran ketidakmurnian (impurity), "
     "seperti Entropy dan Information Gain atau indeks Gini. Semakin murni pembagian yang dihasilkan sebuah fitur, "
     "semakin baik fitur tersebut dijadikan pemisah. Nilai Entropy dan Gini dihitung dengan rumus berikut:")
body("Entropy(S) = \u2212 \u03a3 p\u1d62 log\u2082 p\u1d62")
body("Gini(S) = 1 \u2212 \u03a3 (p\u1d62)\u00b2")
body("dengan p\u1d62 adalah proporsi data yang termasuk kelas ke-i. Kelebihan Decision Tree adalah mudah dipahami dan "
     "diinterpretasikan, sedangkan kekurangannya adalah rentan terhadap overfitting terutama bila pohon tumbuh "
     "terlalu dalam.")
h2("2.3  Algoritma Naive Bayes")
body("Naive Bayes adalah algoritma klasifikasi probabilistik berdasarkan teorema Bayes dengan asumsi kebebasan "
     "(independensi) antar-fitur. Varian Gaussian Naive Bayes mengasumsikan setiap fitur kontinu menyebar normal "
     "Gaussian pada tiap kelas, sehingga peluang fitur dihitung dari rata-rata dan variansi data latih. "
     "Kelebihannya adalah cepat dan hanya membutuhkan sedikit data latih, sedangkan kelemahannya muncul ketika "
     "asumsi independensi dilanggar, misalnya pada data yang didominasi fitur biner yang saling berkorelasi "
     "(Rish, 2001).")
body("P(X|C) = (1 / \u221a(2\u03c0\u03c3\u00b2)) \u00b7 e^( \u2212 (X \u2212 \u03bc)\u00b2 / 2\u03c3\u00b2 )")
h2("2.4  Algoritma Support Vector Machine (SVM)")
body("Support Vector Machine (SVM) adalah algoritma klasifikasi yang mencari hyperplane dengan margin maksimum "
     "antara kelas-kelas data. Untuk data yang tidak terpisah secara linear, digunakan fungsi kernel (misalnya RBF) "
     "yang memetakan data ke ruang berdimensi lebih tinggi. Kelebihan SVM adalah efektif pada data berdimensi "
     "tinggi, sedangkan kekurangannya adalah kompleksitas komputasi sekitar O(n\u00b2) sehingga lambat pada data "
     "latih berukuran sangat besar (Cortes & Vapnik, 1995).")
h2("2.5  Pra-pemrosesan Data (Preprocessing)")
body("Pra-pemrosesan adalah tahap membersihkan dan menyiapkan data agar berkualitas sebelum digunakan. Pada laporan "
     "ini, tahap ini mencakup tiga hal:")
body("Missing value: nilai kosong pada data yang dapat mengganggu proses pelatihan model.")
body("Duplikasi data: baris data yang tercatat lebih dari satu kali sehingga perlu dihapus agar tidak membebani "
     "dan membiaskan model.")
body("Outlier: nilai ekstrem yang menyimpang jauh dari sebaran data. Outlier dideteksi menggunakan metode IQR "
     "(Interquartile Range) dengan batas bawah dan batas atas sebagai berikut:")
body("IQR = Q3 \u2212 Q1")
body("Batas Bawah = Q1 \u2212 1,5 \u00d7 IQR   ;   Batas Atas = Q3 + 1,5 \u00d7 IQR")
h2("2.6  Transformasi Data (MinMaxScaler)")
body("Transformasi data dilakukan untuk menyeragamkan skala antar-fitur. MinMaxScaler adalah teknik normalisasi yang "
     "mengubah nilai setiap fitur ke dalam rentang 0 sampai 1 menggunakan rumus:")
body("X' = ( X \u2212 X\u2098\u1d62\u2099 ) / ( X\u2098\u2090\u2093 \u2212 X\u2098\u1d62\u2099 )")
body("Normalisasi penting agar fitur dengan rentang nilai besar (misalnya BMI) tidak mendominasi fitur dengan rentang "
     "kecil, sehingga model dapat belajar secara lebih adil. Namun, MinMaxScaler tidak menghilangkan pengaruh outlier "
     "karena nilai ekstrem tetap memengaruhi nilai minimum dan maksimum yang digunakan dalam penskalaan "
     "(scikit-learn developers, 2026).")
h2("2.7  Data Tidak Seimbang: SMOTE dan Undersampling")
body("Data tidak seimbang (imbalanced) terjadi ketika distribusi jumlah data antar-kelas sangat timpang. Model yang "
     "dilatih pada data semacam ini cenderung memihak kelas mayoritas dan buruk dalam mengenali kelas minoritas. "
     "SMOTE menangani masalah ini dengan membuat sampel sintetis kelas minoritas melalui interpolasi antara sampel "
     "minoritas dan tetangga terdekatnya, bukan sekadar menduplikasi data (Chawla et al., 2002).")
body("Berbeda dengan penggandaan biasa, SMOTE membuat data sintetis baru pada kelas minoritas dengan cara melakukan "
     "interpolasi di antara data minoritas yang berdekatan (tetangga terdekat). Dengan demikian jumlah data setiap "
     "kelas menjadi seimbang tanpa sekadar menyalin data yang sama. SMOTE hanya diterapkan pada data latih untuk "
     "menghindari kebocoran data (data leakage).")
body("Sebagai pembanding, laporan ini juga menerapkan Random Undersampling, yaitu membuang data kelas mayoritas "
     "secara acak hingga jumlahnya setara dengan kelas minoritas. Undersampling membuat data latih jauh lebih kecil "
     "sehingga algoritma berat seperti SVM dapat dilatih, tetapi mengorbankan sebagian besar informasi kelas "
     "mayoritas.")
h2("2.8  Evaluasi: Confusion Matrix dan Metrik")
body("Confusion Matrix adalah tabel yang membandingkan hasil prediksi model dengan nilai sebenarnya. Dari tabel ini "
     "diperoleh empat komponen dasar: True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative "
     "(FN). Berdasarkan komponen tersebut dihitung metrik evaluasi berikut:")
body("Akurasi = (TP + TN) / (TP + TN + FP + FN)")
body("Presisi = TP / (TP + FP)")
body("Recall = TP / (TP + FN)")
body("F1-Score = 2 \u00d7 (Presisi \u00d7 Recall) / (Presisi + Recall)")
body("Akurasi menunjukkan proporsi prediksi benar secara keseluruhan; presisi menunjukkan ketepatan prediksi positif; "
     "recall menunjukkan kemampuan menemukan seluruh data positif; dan F1-Score adalah rata-rata harmonik antara "
     "presisi dan recall. Pada data tidak seimbang, accuracy dapat memberikan penilaian yang terlalu optimistis karena "
     "kelas mayoritas mendominasi; karena itu, F1-score lebih sesuai digunakan sebagai metrik utama "
     "(Chicco & Jurman, 2025; Olisah et al., 2022).")

# ================= BAB III =================
h1("BAB III  METODOLOGI")
h2("3.1  Dataset")
body("Dataset yang digunakan adalah Diabetes Health Indicators Dataset (BRFSS 2015) yang bersumber dari survei CDC "
     "Amerika Serikat. Dataset ini memiliki 253.680 baris dan 22 kolom, terdiri atas 21 fitur indikator kesehatan "
     "dan 1 kolom target Diabetes_012. Kolom target memiliki tiga kelas, yaitu kelas 0 (tidak diabetes), kelas 1 "
     "(pradiabetes), dan kelas 2 (diabetes).")
body("Distribusi awal kelas target ditunjukkan pada tabel berikut.")
tabel(["Kelas", "Jumlah Data", "Persentase"],
      [["0 (Tidak Diabetes)", "213.703", "84,24%"],
       ["1 (Pradiabetes)", "4.631", "1,83%"],
       ["2 (Diabetes)", "35.346", "13,93%"]])
h2("3.2  Alat dan Bahan")
body("Program dibuat menggunakan bahasa pemrograman Python dengan bantuan pustaka pandas dan numpy untuk pengolahan "
     "data, scikit-learn untuk pemodelan dan evaluasi (DecisionTreeClassifier, GaussianNB, SVC), imbalanced-learn "
     "(imblearn) untuk SMOTE dan RandomUnderSampler, serta matplotlib dan seaborn untuk visualisasi. Aplikasi web "
     "dibangun menggunakan Streamlit dan dijalankan pada lingkungan virtual environment (.venv).")
h2("3.3  Alur Laporan")
body("Program disusun mengikuti alur kerja: Input Data \u2192 Preprocessing \u2192 Split Data (80:20) \u2192 "
     "Transformasi (MinMaxScaler) \u2192 Resampling (SMOTE atau Undersampling, hanya pada data latih) \u2192 Training "
     "(Decision Tree / Naive Bayes / SVM) \u2192 Testing \u2192 Evaluasi. Transformasi dan resampling hanya diterapkan "
     "pada data latih guna mencegah kebocoran data. Model dengan akurasi tertinggi kemudian diimplementasikan ke "
     "dalam aplikasi web Streamlit yang memuat dashboard perbandingan dan form prediksi.")
caption("Flowchart alur program klasifikasi")

# ================= BAB IV =================
h1("BAB IV  IMPLEMENTASI DAN PEMBAHASAN")
h2("4.1  Import Library")
body("Tahap pertama adalah mengimpor seluruh library yang dibutuhkan. Potongan kode berikut memuat pustaka untuk "
     "pengolahan data, pembagian data, transformasi, ketiga algoritma, metrik evaluasi, resampling, serta "
     "visualisasi.")
caption("Kode import library yang dibutuhkan")
kode("import pandas as pd",
     "from sklearn.model_selection import train_test_split",
     "from sklearn.preprocessing import MinMaxScaler",
     "from sklearn.tree import DecisionTreeClassifier",
     "from sklearn.naive_bayes import GaussianNB",
     "from sklearn.svm import SVC",
     "from sklearn.metrics import confusion_matrix, accuracy_score,",
     "    precision_score, recall_score, f1_score, classification_report",
     "from imblearn.over_sampling import SMOTE",
     "from imblearn.under_sampling import RandomUnderSampler")
h2("4.2  Input Data")
body("Dataset dibaca menggunakan fungsi read_csv dari pandas, kemudian ditampilkan ukuran datanya untuk memastikan "
     "data berhasil dimuat. Hasilnya menunjukkan data berukuran 253.680 baris dan 22 kolom.")
caption("Kode tahap Input Data")
kode('df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")',
     'print("Ukuran data awal :", df.shape)')
caption("Output tahap Input Data")
kode("Ukuran data awal : (253680, 22)")
h2("4.3  Preprocessing")
body("Tahap pra-pemrosesan terdiri dari deteksi missing value, penghapusan duplikasi, dan penanganan outlier "
     "menggunakan metode IQR pada fitur BMI, MentHlth, dan PhysHlth. Nilai yang berada di luar batas IQR ditangani "
     "dengan teknik capping, yaitu dipotong ke batas terdekat sehingga tidak mengganggu proses pelatihan.")
caption("Kode tahap Preprocessing")
kode('df = df.drop_duplicates().reset_index(drop=True)',
     'for kol in ["BMI", "MentHlth", "PhysHlth"]:',
     "    Q1, Q3 = df[kol].quantile(0.25), df[kol].quantile(0.75)",
     "    IQR = Q3 - Q1",
     "    df[kol] = df[kol].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)")
caption("Output tahap Preprocessing")
kode("Missing value per kolom (total): 0",
     "Jumlah data duplikat : 23899",
     "Ukuran setelah hapus duplikat : (229781, 22)",
     "BMI       : 5638 outlier   (batas 12.0 - 44.0)",
     "MentHlth  : 36163 outlier  (batas -3.0 - 5.0)",
     "PhysHlth  : 34347 outlier  (batas -6.0 - 10.0)")
body("Hasil pemeriksaan menunjukkan tidak terdapat missing value (0), ditemukan 23.899 baris duplikat yang kemudian "
     "dihapus sehingga data tersisa 229.781 baris, serta outlier pada fitur BMI, MentHlth, dan PhysHlth yang telah "
     "ditangani dengan capping.")
h2("4.4  Split Data")
body("Data dibagi menjadi 80% data latih dan 20% data uji menggunakan train_test_split. Parameter stratify=y menjaga "
     "proporsi tiap kelas tetap sama pada kedua bagian, sedangkan random_state=42 memastikan hasil pembagian dapat "
     "direproduksi.")
caption("Kode tahap Split Data")
kode("X_train, X_test, y_train, y_test = train_test_split(",
     "    X, y, test_size=0.20, random_state=42, stratify=y)")
body("Hasil pembagian menghasilkan 183.824 baris data latih dan 45.957 baris data uji.")
caption("Output tahap Split Data")
kode("Data training : 183824 baris",
     "Data testing  : 45957 baris")
h2("4.5  Transformasi (MinMaxScaler)")
body("Seluruh fitur dinormalisasi ke rentang 0\u20131 menggunakan MinMaxScaler. Objek scaler di-fit hanya pada data "
     "latih, lalu digunakan untuk mentransformasi data latih maupun data uji. Cara ini menjaga agar informasi dari "
     "data uji tidak ikut memengaruhi proses transformasi.")
caption("Kode tahap Transformasi")
kode("scaler = MinMaxScaler()",
     "X_train = scaler.fit_transform(X_train)",
     "X_test = scaler.transform(X_test)")
h2("4.6  Resampling")
body("Karena data sangat tidak seimbang, dilakukan penyeimbangan pada data latih dengan dua skenario. Skenario "
     "pertama menggunakan oversampling SMOTE: sebelum SMOTE jumlah data kelas 0, 1, dan 2 berturut-turut adalah "
     "152.043; 3.703; dan 28.078, sedangkan setelah SMOTE ketiga kelas menjadi seimbang dengan jumlah masing-masing "
     "152.043 data (total 456.129 baris). Perubahan ini terlihat jelas pada kedua grafik berikut.")
gambar(OUT_SMOTE / "distribusi_sebelum_smote.png", "Distribusi kelas SEBELUM SMOTE (imbalance)")
gambar(OUT_SMOTE / "distribusi_sesudah_smote.png", "Distribusi kelas SESUDAH SMOTE (seimbang)")
body("Skenario kedua menggunakan Random Undersampling sebagai pembanding: data kelas mayoritas dibuang acak hingga "
     "setara kelas terkecil, sehingga data latih menyusut menjadi 3 \u00d7 3.703 = 11.109 baris. Skenario ini "
     "membuat SVM kernel RBF dapat dilatih dengan data penuh (tanpa sampling tambahan), tetapi mengorbankan 97% "
     "data kelas sehat.")
h2("4.7  Training (Decision Tree, Naive Bayes, dan SVM)")
body("Ketiga model dilatih dengan konfigurasi dasar (default) di atas data latih yang telah diseimbangkan. "
     "Decision Tree membentuk aturan keputusan bercabang berdasarkan nilai fitur; Naive Bayes menghitung peluang "
     "Gaussian tiap fitur pada tiap kelas; SVM mencari hyperplane margin maksimum dengan kernel RBF.")
caption("Kode tahap Training ketiga model")
kode('models = {"DecisionTree": DecisionTreeClassifier(random_state=42),',
     '          "NaiveBayes": GaussianNB(),',
     '          "SVM": SVC(random_state=42)}',
     "for nama, model in models.items():",
     "    model.fit(X_train_res, y_train_res)")
body("Catatan implementasi: pada track SMOTE, SVM tidak dapat dilatih dengan 456.129 baris (kompleksitas O(n\u00b2), "
     "orde jam hingga kehabisan memori), sehingga digunakan sampel stratified 10.000 baris dari data latih yang "
     "kemudian di-SMOTE menjadi 24.813 baris; data uji tetap penuh dan identik. Pada track undersampling, ketiga "
     "model dilatih penuh (SVM selesai dalam 7,3 detik).")
h2("4.8  Testing")
body("Model yang telah dilatih digunakan untuk memprediksi kelas pada 45.957 baris data uji yang belum pernah dilihat "
     "sebelumnya. Data uji pada kedua track identik sehingga confusion matrix dapat dibandingkan secara langsung "
     "(apple-to-apple).")
caption("Kode tahap Testing")
kode("y_pred = model.predict(X_test)")
h2("4.9  Evaluasi Track SMOTE")
body("Model dievaluasi menggunakan Confusion Matrix serta metrik Akurasi, Presisi, Recall, dan F1-Score dengan "
     "rata-rata weighted.")
caption("Kode tahap Evaluasi")
kode("print(confusion_matrix(y_test, y_pred))",
     'akurasi = accuracy_score(y_test, y_pred)',
     'presisi = precision_score(y_test, y_pred, average="weighted")',
     'recall = recall_score(y_test, y_pred, average="weighted")',
     'f1 = f1_score(y_test, y_pred, average="weighted")')
body("Ringkasan nilai metrik ketiga model pada track SMOTE ditunjukkan pada tabel berikut.")
tabel(["Model", "Akurasi", "Presisi (w)", "Recall (w)", "F1 (w)"],
      [["Decision Tree", "0,7373", "0,7547", "0,7373", "0,7456"],
       ["Naive Bayes", "0,5116", "0,8403", "0,5116", "0,6129"],
       ["SVM", "0,7006", "0,8022", "0,7006", "0,7407"]])
gambar(OUT_SMOTE / "confusion_matrix_DecisionTree.png", "Confusion Matrix Decision Tree (SMOTE)")
gambar(OUT_SMOTE / "confusion_matrix_NaiveBayes.png", "Confusion Matrix Naive Bayes (SMOTE)")
gambar(OUT_SMOTE / "confusion_matrix_SVM.png", "Confusion Matrix SVM (SMOTE, train sampling 10rb)")
body("Decision Tree memperoleh akurasi 73,73% dengan F1-Score weighted 0,7456: kelas 0 dikenali sangat baik "
     "(31.616 dari 38.012, recall 83%), tetapi kelas 1 (pradiabetes) hanya benar 36 dari 926 (recall 4%). "
     "Naive Bayes hanya mencapai akurasi 51,16%: 11.783 data sehat salah dikira pradiabetes karena asumsi "
     "independensi dan Gaussian dilanggar oleh dominasi fitur biner, meskipun recall kelas 2 mencapai 55%. "
     "SVM menempati posisi kedua (akurasi 70,06%, F1 0,7407) dengan recall diabetes 55% (3.863 dari 7.019).")
gambar(OUT_SMOTE / "perbandingan_metrik.png", "Grafik perbandingan metrik ketiga model (SMOTE)")
caption("Output kode Evaluasi")
kode("Model terbaik (akurasi tertinggi): DecisionTree")
h2("4.10  Eksperimen Pembanding: Undersampling")
body("Pada track undersampling, peringkat model berbalik: Naive Bayes menjadi terbaik (akurasi 63,87%, F1 0,6913), "
     "diikuti SVM (59,95%) dan Decision Tree terakhir (45,73%). Decision Tree paling terpukul karena kehilangan 97% "
     "data kelas sehat sehingga gagal mengenali kelas mayoritas (recall kelas 0 hanya 47%), sedangkan Naive Bayes "
     "yang membutuhkan sedikit data justru paling tahan terhadap penyusutan data latih.")
body("Ringkasan metrik track undersampling ditunjukkan pada tabel berikut.")
tabel(["Model", "Akurasi", "Presisi (w)", "Recall (w)", "F1 (w)"],
      [["Naive Bayes", "0,6387", "0,8189", "0,6387", "0,6913"],
       ["SVM", "0,5995", "0,8365", "0,5995", "0,6800"],
       ["Decision Tree", "0,4573", "0,7835", "0,4573", "0,5575"]])
gambar(OUT_UNDER / "confusion_matrix_DecisionTree.png", "Confusion Matrix Decision Tree (undersampling)")
gambar(OUT_UNDER / "confusion_matrix_NaiveBayes.png", "Confusion Matrix Naive Bayes (undersampling)")
gambar(OUT_UNDER / "confusion_matrix_SVM.png", "Confusion Matrix SVM (undersampling)")
gambar(OUT_UNDER / "perbandingan_metrik.png", "Grafik perbandingan metrik ketiga model (undersampling)")
h2("4.11  Implementasi Aplikasi Web (Streamlit)")
body("Model terbaik track SMOTE, yaitu Decision Tree, diimplementasikan ke dalam aplikasi web Streamlit (app.py) "
     "yang memuat dua halaman: Dashboard Perbandingan yang menampilkan tabel metrik, grafik, confusion matrix, dan "
     "analisis kedua track; serta halaman Prediksi Diabetes berupa form 21 indikator kesehatan yang menerapkan "
     "preprocessing identik (capping outlier dan MinMaxScaler) sebelum memprediksi kelas 0 (sehat), 1 "
     "(pradiabetes), atau 2 (diabetes) beserta probabilitasnya. Aplikasi dijalankan dengan perintah "
     "streamlit run app.py dan dapat diakses pada http://localhost:8501.")

# ================= BAB V =================
h1("BAB V  PENUTUP")
h2("5.1  Kesimpulan")
body("Program klasifikasi diabetes berhasil dibangun dan dijalankan sesuai alur, mulai dari input data hingga "
     "evaluasi, untuk tiga algoritma sekaligus.")
body("Tahap preprocessing menghapus 23.899 data duplikat dan menangani outlier dengan capping IQR, sedangkan "
     "SMOTE menyeimbangkan ketiga kelas data latih menjadi masing-masing 152.043 data dan undersampling menjadi "
     "masing-masing 3.703 data.")
body("Pada track SMOTE, Decision Tree menjadi model terbaik (akurasi 73,73%, F1 0,7456), diikuti SVM (70,06%) dan "
     "Naive Bayes (51,16%). Pada track undersampling peringkat berbalik: Naive Bayes terbaik (63,87%), diikuti SVM "
     "(59,95%) dan Decision Tree (45,73%), yang membuktikan bahwa pilihan teknik resampling menentukan pemenang.")
body("Model Decision Tree track SMOTE diimplementasikan ke aplikasi web Streamlit yang menyediakan dashboard "
     "perbandingan dan prediksi interaktif.")
h2("5.2  Saran")
body("Melakukan penyetelan parameter (hyperparameter tuning), misalnya membatasi kedalaman Decision Tree, mengatur "
     "parameter C dan gamma SVM, atau mencoba kernel linear.")
body("Mencoba algoritma lain seperti Random Forest atau XGBoost yang umumnya memberikan hasil lebih baik pada data "
     "tabular.")
body("Menangani kelas pradiabetes secara khusus, misalnya dengan penggabungan kelas atau teknik resampling yang "
     "mempertahankan informasi kelas mayoritas seperti SMOTE-Tomek.")

h1("DAFTAR PUSTAKA")
body("Attari, M. Y. N., Ghadim, A. A., Ala, A., Simic, V., Tinka, D., & Pamucar, D. (2026). Application of "
     "healthcare data mining techniques to planning for nursing length of stay in surgical departments. Systems "
     "and Soft Computing, 9, 200535. https://doi.org/10.1016/j.sasc.2026.200535")
body("Olisah, C. C., Smith, L., & Smith, M. (2022). Diabetes mellitus prediction and diagnosis from a data "
     "preprocessing and machine learning perspective. Computer Methods and Programs in Biomedicine, 220, 106773. "
     "https://doi.org/10.1016/j.cmpb.2022.106773")
body("International Diabetes Federation. (2023, April 25). Indonesia. "
     "https://idf.org/our-network/regions-and-members/western-pacific/members/indonesia/")
body("Alsulami, A., Khalifa, R., & Alghamdi, W. (2026). Improving performance and accuracy in decision trees: A "
     "literature survey on impurity functions. International Journal of Advanced Computer Science and Applications "
     "(IJACSA), 17(4), 984\u2013995.")
body("scikit-learn developers. (2026). MinMaxScaler. scikit-learn. "
     "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html")
body("Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority "
     "Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321\u2013357. "
     "https://doi.org/10.1613/jair.953")
body("Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score "
     "and accuracy in binary classification evaluation. BMC Genomics, 21, 6. "
     "https://doi.org/10.1186/s12864-019-6413-7")
body("Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273\u2013297. "
     "https://doi.org/10.1023/A:1022627411411")
body("Rish, I. (2001). An empirical study of the naive Bayes classifier. IJCAI 2001 Workshop on Empirical Methods "
     "in Artificial Intelligence, 3(22), 41\u201346.")

DOC.parent.mkdir(parents=True, exist_ok=True)

# Isi DAFTAR GAMBAR (disisipkan setelah judulnya, urutan dijaga)
for _n, _t in reversed(GAMBAR):
    for _p in doc.paragraphs:
        if _p.text.strip() == "DAFTAR GAMBAR":
            _new = OxmlElement("w:p")
            _p._element.addnext(_new)
            from docx.text.paragraph import Paragraph as _Para
            _np = _Para(_new, doc)
            _r = _np.add_run(f"Gambar {_n}. {_t}")
            _r.font.name = FONT
            _r.font.size = Pt(12)
            break

doc.save(str(DOC))
print("Tersimpan:", DOC, "| gambar:", _no_gambar)
