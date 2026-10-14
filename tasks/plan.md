# Implementation Plan: Audit & Rekonstruksi Riset 3 Jurnal YOLOv8-UAV

## Identitas Publikasi & Penulis
- **Penulis**: Maheza Novrayuda
- **Afiliasi**: Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Putra Indonesia YPTK Padang
- **Korespondensi**: maheza.novrayuda@upiyptk.ac.id
- **Target Penerbit**: MIND Journal (SINTA 3) / Jurnal Ilmiah Terakreditasi Nasional

---

## 1. Masalah Kritis Hasil Audit Reviewer (P0 & P1)

| No | Kode | Masalah | Akar Penyebab Teknis | Solusi Rekonstruksi |
|---|---|---|---|---|
| 1 | P0.1 | $mAP_{small} \approx 0.0001$ & Bootstrap $p=0.5400$ semu | Bounding Box GT diskalakan ke $640 \times 640$, sedangkan prediksi YOLO dalam resolusi asli ($1360 \times 765$). IoU selalu $\approx 0$. | Selaraskan koordinat ke resolusi asli citra (`orig_shape`) atau normalisasi seragam `[0, 1]`. |
| 2 | P0.2 | Klaim cacat "41 False Positive dihapus" | Mengurangkan $89 - 48 = 41$ tanpa mencocokkan prediksi terhadap Ground Truth. | Evaluasi kuadran deteksi riil (TP, FP, FN pada IoU 0.50): FP turun dari 51 ke 13 (reduksi 38 FP / -74.5%), Presisi melonjak 42.7% -> 72.9%. |
| 3 | P0.3 | Salah tafsir inferensi statistik | Klaim $p = 0.5400$ "membuktikan performa setara" (fallacy of affirming the null). | Tulis interpretasi jujur: gagal menolak $H_0$, mencerminkan trade-off detektor presisi tinggi tanpa degradasi drastis. |
| 4 | P0.4 | Kebocoran citra adegan (*Scene Leakage*) | Pemotongan sekuensial VisDrone menyebabkan sekuens `0000291` dan `0000333` terbagi antar-split. | Audit dan dokumentasikan protokol partisi, sertakan skrip scene-aware splitting di `dataset.py`. |
| 5 | P0.5 | Verifikasi eksekusi Wise-IoU Loss | Belum ada rekaman eksekusi runtime fungsi loss. | Tambahkan log telemetri dan asserting gradient tracking pada `WiseIoUBboxLoss`. |
| 6 | P1.1 | Jurnal 2: XAI hanya 1 citra, normalisasi terpisah | Layer diuji dalam satu model, BGR bukan RGB, belum ada metrik kuantitatif. | Bandingkan Model 2 (P2 murni) vs Model 3 (P2+CBAM), gunakan joint normalization, hitung Foreground Activation Ratio (FAR) kuantitatif. |
| 7 | P1.2 | Jurnal 3: Benchmark hanya forward pass sintetis | Mengukur tensor dummy, tanpa preprocessing dan NMS. | Ukur End-to-End Latency riil ($T_{\text{pre}} + T_{\text{inf}} + T_{\text{post}}$) dan FPS kontinu pada GPU RTX 2050. |
| 8 | P2.1 | Batasan format MIND Journal (SINTA 3) | Jurnal 1 > 2 MB, judul Jurnal 2 memakai kata "Analisis", placeholder nama penulis. | Kompres gambar < 2 MB, ganti judul tanpa "Analisis", isi identitas Maheza Novrayuda, format margin 4 cm. |

---

## 2. Rencana Tahapan Kerja (Phased Execution)

### Fase 1: Perbaikan Modul Inti (`modules/`)
- [x] **Task 1.1**: Perbaiki `modules/metrics.py` untuk mengatasi dimensional mismatch bug pada `compute_map_small` dan `compute_per_image_accuracy_scores`.
- [x] **Task 1.2**: Tambahkan fungsi kuadran deteksi `compute_ground_truth_detection_quadrants()` di `modules/metrics.py` untuk menghitung TP, FP, FN, Presisi, Recall, dan FP Reduction riil.
- [x] **Task 1.3**: Tambahkan logging eksekusi & gradient tracking di `modules/wise_iou.py`.
- [x] **Task 1.4**: Sediakan fungsi `split_by_scene_grouping()` di `modules/dataset.py`.

### Fase 2: Eksekusi Evaluasi Ilmiah & Regenerasi Artifak (`scripts/evaluate_research.py`)
- [x] **Task 2.1**: Jalankan ulang evaluasi kuantitatif Test Set untuk 4 model (mAP@50, mAP@50:0.95, $mAP_{small}$, TP/FP/FN kuadran).
- [x] **Task 2.2**: Hitung ulang Paired Bootstrap dengan skor F1 riil (bukan 0.0) dan simpan hasil JSON.
- [x] **Task 2.3**: Jalankan Benchmark End-to-End Riil (Preprocess + Forward + NMS) pada RTX 2050 (Tabel 4 Jurnal 3).
- [x] **Task 2.4**: Ekstrak XAI Feature Map & hitung Foreground Activation Ratio (FAR) untuk Jurnal 2 (+17.95% relative gain).
- [x] **Task 2.5**: Regenerasi visualisasi Gambar 1, Gambar 2, Gambar 3 resolusi tinggi namun terkompresi optimal.

### Fase 3: Rekonstruksi Notebook & Dokumentasi Utama
- [x] **Task 3.1**: Perbarui `scripts/build_notebook.py` dengan sel data riil, narasi ilmiah yang jujur, dan eksekusi bersih.
- [x] **Task 3.2**: Bangun ulang `Optimasi_Deteksi_Objek_Kecil_UAV_YOLOv8_P2_CBAM_WiseIoU.ipynb`.
- [x] **Task 3.3**: Perbarui `README.md` dengan tabel empiris terkoreksi.

### Fase 4: Penyempurnaan 3 Manuskrip Word & RIS Mendeley
- [x] **Task 4.1**: Perbarui **Jurnal 1**: Identitas Maheza Novrayuda (UPI YPTK Padang), tabel metrik terkoreksi, analisis kuadran TP/FP/FN (-74.5% FP reduction), interpretasi bootstrap jujur, ukuran berkas DOCX terkompresi 0.26 MB (< 2.0 MB).
- [x] **Task 4.2**: Perbarui **Jurnal 2**: Identitas Maheza Novrayuda, revisi judul tanpa kata "Analisis" (`Interpretabilitas Fitur P2 Menggunakan CBAM...`), integrasi metrik XAI FAR kuantitatif (37.85% -> 44.64%, +17.95%), ukuran berkas DOCX 0.39 MB (< 2.0 MB).
- [x] **Task 4.3**: Perbarui **Jurnal 3**: Identitas Maheza Novrayuda, judul tanpa kata "Analisis" (`Komparasi Kinerja dan Efisiensi Komputasi Inferensi Real-Time...`), integrasi latensi End-to-End riil ($T_{\text{pre}}$, $T_{\text{inf}}$, $T_{\text{post}}$, 37.1 FPS), ukuran berkas DOCX 0.27 MB (< 2.0 MB).
- [x] **Task 4.4**: Verifikasi dan sinkronisasi berkas RIS di ketiga folder jurnal (25, 24, dan 22 referensi IEEE terverifikasi).
