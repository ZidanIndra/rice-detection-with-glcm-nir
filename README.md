## Rice Detection with GLCM + NIR

Repositori ini berisi proyek klasifikasi padi (rice) berbasis ekstraksi fitur tekstur GLCM (Gray-Level Co-occurrence Matrix) pada citra dan pemanfaatan kanal NIR (Near-Infrared), diikuti dengan pelatihan model klasifikasi yang diekspor sebagai `rice_classification_model.h5`. Eksperimen utama tersedia dalam `Notebook.ipynb`, hasil evaluasi disimpan pada folder `glcm_results`, serta dataset contoh dikemas dalam `dataset-gambar.zip`.

### Daftar Isi
- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Arsitektur & Algoritma](#arsitektur--algoritma)
  - [GLCM (Gray-Level Co-occurrence Matrix)](#glcm-gray-level-co-occurrence-matrix)
  - [Kanal NIR (Near-Infrared)](#kanal-nir-near-infrared)
  - [Ekstraksi Fitur](#ekstraksi-fitur)
  - [Model Klasifikasi](#model-klasifikasi)
- [Alur Proyek](#alur-proyek)
- [Persiapan Lingkungan](#persiapan-lingkungan)
- [Menjalankan Proyek](#menjalankan-proyek)
  - [Via Jupyter Notebook](#via-jupyter-notebook)
  - [Inferensi Cepat (Kode Contoh)](#inferensi-cepat-kode-contoh)
- [Struktur Repository](#struktur-repository)
- [Hasil & Evaluasi](#hasil--evaluasi)
- [Troubleshooting](#troubleshooting)
- [Lisensi](#lisensi)
- [Sitasi](#sitasi)

---

### Latar Belakang
Kualitas gabah/beras dapat diidentifikasi dari pola tekstur pada permukaan butir serta respons spektral tertentu seperti NIR. GLCM merupakan metode statistik orde dua untuk menangkap keteraturan tekstur (kontras, homogenitas, energi, korelasi), sedangkan kanal NIR membantu membedakan material berdasarkan pantulan spektral di luar cahaya tampak. Menggabungkan keduanya menghasilkan fitur yang lebih representatif untuk tugas klasifikasi.

### Dataset
- File `dataset-gambar.zip` berisi contoh citra yang digunakan untuk eksperimen. Silakan ekstrak sebelum digunakan.
- Struktur umum yang disarankan setelah ekstraksi:
  - `dataset-gambar/` berisi subfolder per kelas (mis. `sehat/`, `tidak_sehat/`) atau penamaan lain sesuai kebutuhan.
- Pastikan format dan label konsisten dengan pemrosesan di `Notebook.ipynb`.

### Arsitektur & Algoritma

#### GLCM (Gray-Level Co-occurrence Matrix)
GLCM menghitung matriks ko-occurrence intensitas piksel pada jarak dan sudut tertentu. Dari matriks ini, fitur-fitur statistik diekstrak, antara lain:
- Kontras (contrast)
- Energi/ASM (energy/ASM)
- Homogenitas (homogeneity)
- Korelasi (correlation)

Pemilihan parameter penting:
- Level quantization (mis. 8/16/32 level)
- Jarak (distance) dan sudut (0°, 45°, 90°, 135°)
- Agregasi antar-sudut (rata-rata atau concatenation)

#### Kanal NIR (Near-Infrared)
Jika citra memiliki kanal NIR (mis. dari kamera multispektral), kanal ini dapat:
- Dipakai langsung sebagai fitur intensitas rata-rata/statistik.
- Digabung dengan kanal abu-abu (grayscale) untuk memperkaya deskriptor tekstur.
Jika NIR tidak tersedia, bisa digantikan kanal lain atau dibiarkan kosong sesuai konten dataset.

#### Ekstraksi Fitur
Pipeline umum ekstraksi fitur:
1. Pra-pemrosesan citra: resize, normalisasi, konversi ke grayscale.
2. (Opsional) Pengambilan kanal NIR dan normalisasi.
3. Hitung GLCM pada satu/berbagai sudut + jarak.
4. Ekstrak fitur: kontras, energi, homogenitas, korelasi (dan metrik lain jika diperlukan).
5. Gabungkan fitur GLCM dengan statistik NIR (mis. mean, std) menjadi satu vektor fitur.

#### Model Klasifikasi
Setelah fitur diekstraksi, digunakan model klasifikasi. Umum dipakai:
- Klasifier klasik: SVM, RandomForest, XGBoost
- Atau MLP (neural network) kecil

Model terlatih disimpan sebagai `rice_classification_model.h5`. Detail persisnya (arsitektur, hyperparameter) terdokumentasi di `Notebook.ipynb`.

### Alur Proyek
1. Siapkan lingkungan Python dan dependensi.
2. Ekstrak `dataset-gambar.zip` ke folder `dataset-gambar/`.
3. Jalankan `Notebook.ipynb` untuk:
   - Memuat data & label
   - Ekstraksi fitur GLCM (+ NIR jika ada)
   - Melatih model dan evaluasi
   - Menyimpan model ke `rice_classification_model.h5`
4. Gunakan model tersimpan untuk inferensi pada citra baru.
5. Cek metrik/plot pada `glcm_results/` untuk hasil evaluasi.

### Persiapan Lingkungan
Disarankan menggunakan Python 3.9+ dan virtual environment.

Windows PowerShell:
```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install --upgrade pip

# Paket inti (silakan sesuaikan dengan kebutuhan di Notebook)
pip install numpy pandas scikit-image scikit-learn matplotlib seaborn jupyter

# Jika menggunakan deep learning (model .h5 Keras/TensorFlow)
pip install tensorflow

# Opsional untuk pemrosesan citra tambahan
pip install opencv-python
```

### Menjalankan Proyek

#### Via Jupyter Notebook
1. Aktivasi environment seperti di atas.
2. Jalankan Jupyter:
```bash
jupyter notebook
```
3. Buka `Notebook.ipynb` dan eksekusi sel dari awal hingga akhir.
4. Setelah training selesai, model akan tersedia sebagai `rice_classification_model.h5`. Hasil metrik/plot dapat ditemukan di `glcm_results/`.

#### Inferensi Cepat (Kode Contoh)
Contoh berikut memuat model `.h5`, mengekstrak fitur GLCM dari satu citra, lalu memprediksi kelas. Sesuaikan fungsi `extract_glcm_features` dengan parameter yang Anda gunakan di Notebook agar konsisten.

```python
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
from tensorflow.keras.models import load_model

def extract_glcm_features(gray_image, distances=(1, 2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4), levels=32):
    # Quantization ke level tertentu
    quantized = np.floor(gray_image / (256 / levels)).astype(np.uint8)

    glcm = greycomatrix(quantized, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    features = []
    for prop in ["contrast", "energy", "homogeneity", "correlation"]:
        prop_vals = greycoprops(glcm, prop)  # shape: (len(distances), len(angles))
        features.append(prop_vals.mean())
        features.append(prop_vals.std())
    return np.array(features, dtype=np.float32)

def load_and_prepare_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gagal memuat gambar: {path}")
    img = cv2.resize(img, (256, 256))
    return img

# Muat model
model = load_model("rice_classification_model.h5")

# Contoh inferensi
gray = load_and_prepare_image("path/ke/gambar_uji.jpg")
feat = extract_glcm_features(gray)

# Jika Anda menambahkan fitur NIR di Notebook, gabungkan di sini:
# nir_stats = np.array([nir_mean, nir_std], dtype=np.float32)
# feat = np.concatenate([feat, nir_stats], axis=0)

pred = model.predict(feat.reshape(1, -1))
predicted_class = np.argmax(pred, axis=1)[0]
print("Prediksi kelas:", predicted_class)
```

### Struktur Repository
```
.
├─ Notebook.ipynb                # Eksperimen utama (pemrosesan, training, evaluasi)
├─ rice_classification_model.h5  # Model terlatih (jika sudah dihasilkan)
├─ dataset-gambar.zip            # Dataset contoh (ekstrak sebelum dipakai)
├─ glcm_results/                 # Hasil metrik/plot/log evaluasi
└─ README.md                     # Dokumentasi proyek ini
```

### Hasil & Evaluasi
- Ringkasan metrik (akurasi, precision, recall, F1) serta visualisasi (confusion matrix, distribusi fitur) dapat ditemukan/dihasilkan melalui `Notebook.ipynb` dan disimpan ke `glcm_results/`.
- Perhatikan konsistensi preprocessing antara training dan inferensi agar metrik di produksi tetap stabil.

### Troubleshooting
- ImportError: pastikan semua paket telah terinstal pada environment aktif.
- `rice_classification_model.h5` tidak ditemukan: jalankan seluruh `Notebook.ipynb` hingga sel penyimpanan model, atau pastikan path benar saat load.
- Prediksi tidak stabil: samakan parameter ekstraksi GLCM (levels, distances, angles) dan preprocessing dengan yang digunakan saat training.
- Citra gagal dibaca: periksa path dan ekstensi, atau coba `cv2.imread(path)` tanpa flag grayscale untuk debugging.

### Lisensi
Silakan tambahkan jenis lisensi yang sesuai (mis. MIT, Apache-2.0) jika ingin mendistribusikan.

### Sitasi
Jika Anda menggunakan kode/ide dari proyek ini, cantumkan sitasi ke repositori ini dan, bila relevan, sitasi literatur GLCM klasik:
- Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. IEEE Transactions on Systems, Man, and Cybernetics.

—
Dibuat untuk membantu reproduksibilitas dan pemahaman alur kerja klasifikasi padi berbasis GLCM + NIR.


