# Cardiovascular Disease Prediction Using Machine Learning

## Deskripsi Proyek

Proyek ini mengimplementasikan sistem prediksi penyakit kardiovaskular (*Cardiovascular Disease*) menggunakan empat algoritma *machine learning* yang berbeda. Sistem ini dibangun sebagai bagian dari studi komparatif model prediktif dalam konteks skripsi, dengan tujuan mengevaluasi performa masing-masing algoritma pada dataset klinis pasien.

Sistem terdiri dari dua komponen utama:
1. **Pipeline pelatihan model** (`train_model.py`) -- melatih empat model klasifikasi secara otomatis dari dataset mentah.
2. **Antarmuka prediksi berbasis web** (`app.py`) -- aplikasi Streamlit yang memungkinkan pengguna memasukkan data klinis dan mendapatkan hasil prediksi secara *real-time*.

---

## Sumber Dataset

| Atribut | Detail |
|---|---|
| **Nama** | Cardiovascular Disease Dataset |
| **Sumber** | [Kaggle -- Svetlana Ulianova](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset) |
| **Jumlah Data Awal** | 70.000 baris, 13 kolom |
| **Jumlah Data Setelah Pembersihan** | ~69.000 baris (setelah penghapusan duplikat dan *outlier*) |
| **Target Variabel** | `cardio` (0 = tidak ada penyakit, 1 = ada penyakit kardiovaskular) |
| **Distribusi Kelas** | Seimbang (~50% positif, ~50% negatif) |

### Fitur Dataset

| No | Fitur | Tipe | Deskripsi |
|----|-------|------|-----------|
| 1 | `age` | Numerik | Usia pasien (dikonversi dari hari ke tahun) |
| 2 | `female` | Biner (0/1) | Jenis kelamin perempuan (hasil *one-hot encoding*) |
| 3 | `male` | Biner (0/1) | Jenis kelamin laki-laki (hasil *one-hot encoding*) |
| 4 | `height` | Numerik (cm) | Tinggi badan pasien |
| 5 | `weight` | Numerik (kg) | Berat badan pasien |
| 6 | `bmi` | Numerik | *Body Mass Index*, dihitung dari `weight / (height/100)^2` |
| 7 | `ap_hi` | Numerik (mmHg) | Tekanan darah sistolik |
| 8 | `ap_lo` | Numerik (mmHg) | Tekanan darah diastolik |
| 9 | `cholesterol` | Ordinal (1-3) | Kadar kolesterol: 1 = Normal, 2 = Di Atas Normal, 3 = Jauh Di Atas Normal |
| 10 | `gluc` | Ordinal (1-3) | Kadar glukosa: 1 = Normal, 2 = Di Atas Normal, 3 = Jauh Di Atas Normal |
| 11 | `smoke` | Biner (0/1) | Status merokok |
| 12 | `alco` | Biner (0/1) | Konsumsi alkohol |
| 13 | `active` | Biner (0/1) | Aktivitas fisik |

---

## Tahapan Metodologi

### 1. Preprocessing dan Pembersihan Data

Proses pembersihan data yang dilakukan meliputi:

- **Konversi usia**: Data usia awal dalam satuan hari dikonversi ke tahun dengan rumus `age / 365.25`.
- **Encoding jenis kelamin**: Kolom `gender` diubah menjadi dua kolom biner (`female`, `male`) menggunakan *one-hot encoding*.
- **Penghapusan duplikat**: Sebanyak 75 baris duplikat dihapus dari dataset.
- **Penambahan fitur BMI**: *Body Mass Index* dihitung dan ditambahkan sebagai fitur baru.
- **Penghapusan outlier BMI**: Data dengan BMI > 60 atau BMI < 15 dihapus karena dianggap sebagai kesalahan pencatatan.
- **Penghapusan outlier tekanan darah**: Data dengan `ap_hi > 220`, `ap_lo > 180`, `ap_hi < 40`, atau `ap_lo < 40` dihapus.
- **Penghapusan kolom tidak relevan**: Kolom `id` dan `gender` (asli) dihapus dari dataset.

### 2. Analisis Data Eksploratif (EDA)

Temuan utama dari analisis data:

- **Distribusi kelas**: Persentase pasien dengan penyakit kardiovaskular adalah ~50%, menunjukkan dataset yang seimbang.
- **Distribusi gender**: 65% pasien adalah perempuan, 35% laki-laki.
- **Korelasi usia**: Ditemukan hubungan signifikan antara usia dan penyakit kardiovaskular; pasien lanjut usia lebih berisiko.
- **Korelasi BMI**: Pasien dengan BMI lebih tinggi menunjukkan kecenderungan lebih besar terhadap penyakit kardiovaskular.
- **Korelasi kolesterol**: ~60% pasien dengan kadar kolesterol "jauh di atas normal" memiliki penyakit kardiovaskular.
- **Kadar glukosa**: Korelasi positif antara kadar glukosa tinggi dan penyakit kardiovaskular.

### 3. Analisis Probabilistik

Beberapa probabilitas kondisional yang dihitung dari dataset:

| Kondisi | Probabilitas Penyakit Kardiovaskular |
|---|---|
| Usia >= 50 tahun | 55,46% |
| BMI >= 37 | 68,78% |
| Krisis hipertensi (tekanan darah sangat tinggi) | 89,29% |
| Merokok atau mengonsumsi alkohol | 47,96% |
| Tidak aktif secara fisik | 53,28% |

### 4. Pelatihan dan Evaluasi Model

Dataset dibagi menjadi *training set* dan *test set* menggunakan `train_test_split` dari scikit-learn. Empat algoritma klasifikasi dilatih dan dievaluasi.

---

## Algoritma Machine Learning

### 4.1 Random Forest Classifier

Random Forest terdiri dari sejumlah *decision tree* yang masing-masing memberikan prediksi secara independen. Kelas dengan suara terbanyak dari seluruh *tree* menjadi prediksi akhir.

**Hyperparameter:**
| Parameter | Nilai |
|---|---|
| `n_estimators` | 51 |
| `max_depth` | 10 |
| `random_state` | 0 |

**Hasil:**
- Testing Accuracy: **72,92%**
- Average Cross-Validation Accuracy: **73,44%**

---

### 4.2 Support Vector Machine (SVM)

SVM memetakan setiap data ke dalam ruang *n*-dimensi dan mencari *hyperplane* optimal yang memisahkan dua kelas dengan margin maksimum. Kernel RBF (*Radial Basis Function*) digunakan untuk menangani data yang tidak terpisah secara linear.

**Hyperparameter:**
| Parameter | Nilai |
|---|---|
| `C` | 100 |
| `gamma` | 0.00001 |
| `kernel` | RBF |
| `random_state` | 42 |
| `probability` | True |

**Catatan:** Model SVM menunjukkan presisi yang lebih tinggi namun *type-one error* (false negative) lebih besar dibandingkan Random Forest, sehingga performa keseluruhannya dinilai kurang optimal untuk konteks medis.

---

### 4.3 K-Nearest Neighbors (KNN)

KNN mengklasifikasikan data baru berdasarkan mayoritas kelas dari *k* tetangga terdekat. Jarak antar data dihitung menggunakan algoritma Ball Tree.

**Hyperparameter:**
| Parameter | Nilai |
|---|---|
| `n_neighbors` | 300 |
| `weights` | uniform |
| `leaf_size` | 1 |
| `algorithm` | ball_tree |

**Hasil:**
- Testing Accuracy: **71,84%**
- Average Cross-Validation Accuracy: **72,24%**

---

### 4.4 XGBoost (Extreme Gradient Boosting)

XGBoost menggunakan *decision tree* dengan kerangka *gradient boosting*. Berbeda dengan Random Forest yang melatih *tree* secara paralel, XGBoost melatih *tree* secara sekuensial, di mana setiap *tree* baru dilatih untuk memperbaiki kesalahan model sebelumnya.

**Hyperparameter:**
| Parameter | Nilai |
|---|---|
| `n_estimators` | 150 |
| `max_depth` | 4 |
| `learning_rate` | 0.13 |
| `gamma` | 0.24 |
| `reg_lambda` | 50.0 |
| `scale_pos_weight` | 1 |
| `seed` | 0 |

**Hasil:**
- Testing Accuracy: **73,20%**
- Average Cross-Validation Accuracy: **73,59%**

---

## Ringkasan Perbandingan Model

| Model | Testing Accuracy | Avg. CV Accuracy | Catatan |
|-------|-----------------|-----------------|---------|
| **XGBoost** | 73,20% | **73,59%** | Akurasi tertinggi; performa paling stabil |
| **Random Forest** | 72,92% | 73,44% | Performa kompetitif; *type-one error* paling rendah |
| **KNN** | 71,84% | 72,24% | Presisi cukup baik; *type-one error* lebih tinggi dari RF |
| **SVM** | -- | -- | Presisi tinggi; *type-one error* terbesar; kurang andal untuk konteks klinis |

**Kesimpulan**: XGBoost mencapai akurasi tertinggi secara keseluruhan. Namun, dalam konteks medis, *type-one error* (mengklasifikasikan pasien sakit sebagai sehat) lebih kritis daripada akurasi semata. Oleh karena itu, pemilihan model akhir harus mempertimbangkan trade-off antara akurasi dan sensitivitas.

---

## Struktur Proyek

```
CARDIOVASCULAR/
|-- app.py                                              # Aplikasi Streamlit untuk prediksi interaktif
|-- train_model.py                                      # Script pelatihan empat model ML
|-- cardio_train.csv                                    # Dataset asli (70.000 baris)
|-- cardiovascular-disease-prediction-73-59-accuracy.ipynb  # Notebook Jupyter (EDA + pelatihan + evaluasi)
|-- random_forest_cvd_model.pkl                         # Model Random Forest tersimpan
|-- svm_cvd_model.pkl                                   # Model SVM tersimpan
|-- knn_cvd_model.pkl                                   # Model KNN tersimpan
|-- xgboost_cvd_model.pkl                               # Model XGBoost tersimpan
|-- README.md                                           # Dokumentasi proyek
```

---

## Cara Penggunaan

### Prasyarat

Pastikan Python 3.8 atau versi lebih baru telah terinstal. Instal dependensi yang dibutuhkan:

```bash
pip install streamlit pandas numpy scikit-learn xgboost
```

### Melatih Model dari Awal

Jika file model `.pkl` belum tersedia atau ingin melatih ulang:

```bash
python train_model.py
```

Script ini akan:
1. Membaca dan membersihkan dataset `cardio_train.csv`.
2. Melatih empat model (XGBoost, Random Forest, KNN, SVM) menggunakan seluruh data yang telah dibersihkan.
3. Menyimpan setiap model ke dalam file `.pkl` terpisah.

### Menjalankan Aplikasi Prediksi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di *browser* dan menyediakan antarmuka untuk:
1. **Memilih model prediksi** -- pengguna dapat memilih salah satu dari empat model yang tersedia (XGBoost, Random Forest, KNN, SVM).
2. **Memasukkan data pasien** -- meliputi data personal (usia, jenis kelamin, tinggi, berat badan), data klinis (tekanan darah sistolik dan diastolik), data laboratorium (kolesterol, glukosa), dan faktor gaya hidup (merokok, alkohol, aktivitas fisik).
3. **Melihat hasil prediksi** -- sistem menampilkan klasifikasi risiko (tinggi/rendah) beserta probabilitas persentase penyakit kardiovaskular.

### Contoh Input untuk Pengujian

| Parameter | Nilai |
|---|---|
| Usia | 55 tahun |
| Jenis Kelamin | Laki-laki |
| Tinggi Badan | 170 cm |
| Berat Badan | 85 kg |
| Tekanan Darah Sistolik | 145 mmHg |
| Tekanan Darah Diastolik | 95 mmHg |
| Kolesterol | Di Atas Normal |
| Glukosa | Normal |
| Merokok | Ya |
| Alkohol | Tidak |
| Aktivitas Fisik | Tidak |

---

## Fitur Vektor Input Model

Model menerima 13 fitur dalam urutan berikut:

```
['age', 'female', 'male', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
```

Fitur `bmi` dihitung secara otomatis oleh aplikasi berdasarkan input tinggi dan berat badan. Fitur `female` dan `male` dihasilkan dari pilihan jenis kelamin pengguna.

---

## Dependensi dan Versi

| Library | Kegunaan |
|---|---|
| `pandas` | Manipulasi dan analisis data tabular |
| `numpy` | Komputasi numerik |
| `scikit-learn` | Implementasi Random Forest, SVM, KNN, dan metrik evaluasi |
| `xgboost` | Implementasi algoritma XGBoost |
| `matplotlib` | Visualisasi grafik (digunakan dalam notebook) |
| `seaborn` | Visualisasi statistik (digunakan dalam notebook) |
| `streamlit` | Framework antarmuka web interaktif |
| `pickle` | Serialisasi dan deserialisasi model |

---

## Metrik Evaluasi

Model dievaluasi menggunakan metrik berikut:

- **Accuracy Score**: Persentase prediksi yang benar dari total prediksi.
- **Cross-Validation (5-Fold)**: Rata-rata akurasi dari lima kali pembagian data yang berbeda, memberikan estimasi performa model yang lebih stabil dan tidak bias terhadap satu pembagian data tertentu.
- **Confusion Matrix**: Matriks yang menunjukkan distribusi *True Positive*, *True Negative*, *False Positive*, dan *False Negative*.
- **ROC Curve**: Kurva yang menggambarkan hubungan antara *True Positive Rate* dan *False Positive Rate* pada berbagai threshold klasifikasi.

---

## Keterbatasan

1. **Akurasi model**: Akurasi tertinggi yang dicapai adalah ~73,6% (XGBoost). Ini merupakan keterbatasan inheren dataset, bukan semata-mata keterbatasan algoritma.
2. **Data laboratorium**: Model memerlukan data kolesterol dan glukosa yang umumnya membutuhkan pemeriksaan laboratorium, sehingga tidak sepenuhnya cocok untuk *screening* mandiri tanpa uji lab.
3. **Generalisasi**: Model dilatih pada satu dataset spesifik dan belum divalidasi pada populasi atau dataset eksternal lainnya.
4. **Bukan pengganti diagnosis medis**: Hasil prediksi bersifat estimasi statistik dan tidak boleh digunakan sebagai pengganti konsultasi dengan tenaga medis profesional.

---

## Referensi

1. Ulianova, S. (2019). *Cardiovascular Disease Dataset*. Kaggle. https://www.kaggle.com/sulianova/cardiovascular-disease-dataset
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
4. Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273-297.
5. Cover, T., & Hart, P. (1967). *Nearest Neighbor Pattern Classification*. IEEE Transactions on Information Theory, 13(1), 21-27.
