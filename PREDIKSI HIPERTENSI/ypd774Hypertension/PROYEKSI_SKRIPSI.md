# Proyeksi Struktur Penulisan Skripsi
**Tema**: Integrasi Model Machine Learning pada Sistem Remote Patient Monitoring (RPM) untuk Prediksi Hipertensi

*Catatan: Struktur ini disesuaikan dengan format standar skripsi Ilmu Komputer/Teknik Informatika pada umumnya. Anda bisa menyesuaikan penamaan sub-bab dengan pedoman format dari kampus Anda.*

---

## BAB I: PENDAHULUAN

**1.1 Latar Belakang**
*   **Konteks Medis**: Hipertensi sebagai *silent killer* dan beban penderitanya.
*   **Konteks Teknologi**: Munculnya tren *Remote Patient Monitoring* (RPM) dan alat IoT (seperti Tensimeter Omron) yang memudahkan pemantauan dari rumah.
*   **Masalah (Gap)**: RPM saat ini umumnya *hanya menampilkan data mentah* (angka tensi) tanpa bisa memberikan peringatan dini atau analisis risiko jangka panjang secara otomatis.
*   **Solusi yang Ditawarkan (Skripsi Anda)**: Mengintegrasikan kecerdasan buatan (Machine Learning menggunakan algoritma *Extra Trees Classifier*) ke dalam sistem RPM untuk menganalisis data tensimeter dan riwayat pasien, sehingga bisa memberikan peringatan risiko hipertensi secara *real-time*.

**1.2 Rumusan Masalah**
1. Bagaimana mengolah dataset klinis dan melatih model *Extra Trees Classifier* untuk deteksi risiko hipertensi?
2. Bagaimana merancang dan membangun API (*Application Programming Interface*) untuk menjembatani model ML dengan sistem RPM?
3. Bagaimana tingkat akurasi dan performa model dalam melakukan klasifikasi risiko hipertensi?

**1.3 Tujuan Penelitian**
*(Menjawab rumusan masalah secara poin per poin)*

**1.4 Batasan Masalah**
*   Model dilatih menggunakan *Framingham Heart Study dataset* (bukan data rumah sakit lokal).
*   Parameter/Fitur yang digunakan dibatasi pada 9 variabel (Sistolik, Diastolik, Heart Rate, dsb).
*   Integrasi dilakukan sebatas pada *prototype* / sistem RPM berbasis web (Streamlit/Flask).

---

## BAB II: TINJAUAN PUSTAKA DAN LANDASAN TEORI

**2.1 Tinjauan Pustaka (Penelitian Terdahulu)**
*   Membahas 3-5 jurnal/paper sebelumnya yang membuat sistem RPM atau membuat ML untuk hipertensi. Jelaskan kelebihan penelitian Anda dibanding mereka (misal: Anda menggabungkan keduanya, atau menggunakan *Extra Trees* yang jarang dipakai untuk kasus ini).

**2.2 Landasan Teori**
*   **Hipertensi**: Definisi klinis, parameter sistolik/diastolik.
*   **Remote Patient Monitoring (RPM)**: Konsep dasar dan arsitekturnya (IoT, Server, Client).
*   **Machine Learning**: Konsep dasar *Supervised Learning* dan klasifikasi.
*   **Extra Trees Classifier**: Penjelasan teori *Ensemble Learning*, *Decision Tree*, dan mengapa algoritma ini tahan terhadap *overfitting*.
*   **Metrik Evaluasi**: *Confusion Matrix*, *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.

---

## BAB III: METODOLOGI PENELITIAN DAN PERANCANGAN SISTEM

**3.1 Alur Penelitian**
*   Sajikan diagram blok (Pengumpulan Data $\rightarrow$ Pre-processing $\rightarrow$ Pelatihan Model $\rightarrow$ Integrasi Sistem $\rightarrow$ Pengujian).

**3.2 Pengumpulan dan Pengolahan Data (Pre-processing)**
*   Deskripsi dataset Framingham.
*   Proses *Cleaning* (menghapus *missing values*/`dropna`).
*   Proses *Scaling* (menggunakan `StandardScaler`).

**3.3 Perancangan Model Machine Learning**
*   Skenario pembagian data (*Train/Test Split*).
*   Parameter algoritma *Extra Trees*.

**3.4 Perancangan Arsitektur Sistem (Integrasi RPM)**
*   **Sistem Arsitektur**: Gambar diagram yang menunjukkan aliran data dari input (Form/Omron) $\rightarrow$ dikirim via JSON $\rightarrow$ masuk ke Flask API $\rightarrow$ diproses Model $\rightarrow$ hasil dikembalikan ke UI (*Streamlit/Frontend*).

---

## BAB IV: HASIL DAN PEMBAHASAN (IMPLEMENTASI)

**4.1 Implementasi Sistem**
*   **Implementasi Model**: Menampilkan kode *training* dan pembuatan file `.pkl`.
*   **Implementasi Backend API**: Menjelaskan *routing* `/predict` pada Flask (`hypertension_api.py`).
*   **Implementasi Frontend (RPM Dashboard)**: Menampilkan *screenshot* antarmuka Streamlit (`frontend_app.py`) saat pengguna memasukkan data dan memunculkan hasil.

**4.2 Hasil Pengujian Model Machine Learning**
*   Menampilkan *Confusion Matrix*.
*   Menampilkan hasil *Classification Report* (Akurasi **97.3%**).
*   **Pembahasan Evaluasi**: Di sinilah Anda memasukkan argumen "Justifikasi Tingkat Akurasi Tinggi" (Korelasi klinis kuat, kelebihan Extra Trees, dan bebas *overfitting*). Bahas juga mengapa nilai **Recall (0.98)** sangat penting untuk diagnosa medis!

**4.3 Hasil Pengujian Integrasi Sistem**
*   Pengujian *Blackbox* (apakah saat tombol ditekan, API berhasil memberikan respons positif/negatif dengan benar tanpa *error*).
*   (Opsional) Pengujian waktu respons API (*latency*).

---

## BAB V: KESIMPULAN DAN SARAN

**5.1 Kesimpulan**
*   Model *Extra Trees Classifier* terbukti sangat efektif memprediksi risiko hipertensi dengan akurasi 97.33% dan sangat sensitif mengenali pasien sakit (Recall 0.98).
*   Model berhasil diintegrasikan ke dalam arsitektur RPM melalui pembangunan RESTful API menggunakan Flask, memungkinkan prediksi dilakukan secara *real-time* lewat *dashboard*.

**5.2 Saran**
*   Untuk penelitian selanjutnya, disarankan melatih ulang model menggunakan rekam medis asli populasi Indonesia (misal: data BPJS/RSUD) agar relevansinya lebih kontekstual.
*   Sistem dapat dikembangkan menjadi aplikasi *Mobile* (Android/iOS) yang terhubung langsung secara *Bluetooth* dengan alat Tensimeter pintar.

---
