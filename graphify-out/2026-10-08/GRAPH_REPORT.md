# Graph Report - Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss  (2026-10-08)

## Corpus Check
- 15 files · ~7,057,551 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 138 nodes · 169 edges · 15 communities (12 shown, 3 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- evaluate_research.py
- .__init__
- models.py
- WiseIoUBboxLoss
- train_research.py
- eda.py
- embed_execution_outputs.py
- build_notebook.py
- run_headless_verification.py
- 🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss
- 7. Bedah Menyeluruh Seluruh Output: Kurva, Confusion Matrix, Precision, Recall, F1, Accuracy, & MAPE
- 2. Rencana Kerja Bertahap (Phased Action Plan)
- 6. Katalog Tabel Publikasi & Data Empiris Lengkap (Tabel 1 – Tabel 5)
- Task Checklist: Transformasi Eksperimen Menuju Publikasi Ilmiah
- 2. Triad Inovasi Arsitektur & Formulasi Matematika Lengkap (LaTeX)

## God Nodes (most connected - your core abstractions)
1. `🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss` - 15 edges
2. `register_cbam_to_ultralytics()` - 8 edges
3. `setup_ablation_model()` - 7 edges
4. `main()` - 7 edges
5. `setup_uav_dataset()` - 6 edges
6. `paired_bootstrap_significance()` - 6 edges
7. `2. Triad Inovasi Arsitektur & Formulasi Matematika Lengkap (LaTeX)` - 6 edges
8. `6. Katalog Tabel Publikasi & Data Empiris Lengkap (Tabel 1 – Tabel 5)` - 6 edges
9. `7. Bedah Menyeluruh Seluruh Output: Kurva, Confusion Matrix, Precision, Recall, F1, Accuracy, & MAPE` - 6 edges
10. `ChannelAttention` - 5 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `register_cbam_to_ultralytics()`  [EXTRACTED]
  scripts/evaluate_research.py → modules/cbam.py
- `main()` --calls--> `compute_map_small()`  [EXTRACTED]
  scripts/evaluate_research.py → modules/metrics.py
- `main()` --calls--> `compute_per_image_accuracy_scores()`  [EXTRACTED]
  scripts/evaluate_research.py → modules/metrics.py
- `main()` --calls--> `paired_bootstrap_significance()`  [EXTRACTED]
  scripts/evaluate_research.py → modules/metrics.py
- `main()` --calls--> `export_latex_table()`  [EXTRACTED]
  scripts/evaluate_research.py → modules/metrics.py

## Import Cycles
- None detected.

## Communities (15 total, 3 thin omitted)

### Community 0 - "evaluate_research.py"
Cohesion: 0.16
Nodes (17): Modul Perhatian CBAM (Convolutional Block Attention Module) untuk YOLOv8.…, Mendaftarkan modul CBAM secara dinamis ke ekosistem Ultralytics sehingga model…, register_cbam_to_ultralytics(), compute_map_small(), compute_per_image_accuracy_scores(), export_latex_table(), paired_bootstrap_significance(), Any (+9 more)

### Community 1 - ".__init__"
Cohesion: 0.21
Nodes (7): CBAM, ChannelAttention, Tensor, Channel Attention Module (CAM). Menghitung bobot kepentingan antar-channel…, Spatial Attention Module (SAM). Menghitung bobot spasial untuk memusatkan…, Convolutional Block Attention Module (CBAM) lengkap. Menerapkan Channel…, SpatialAttention

### Community 2 - "models.py"
Cohesion: 0.23
Nodes (10): prepare_pretrained_checkpoint(), Path, Utility module for preparing and initializing ablation models with smart weight…, Sets up a fully configured YOLO model for ablation training using native…, Menginisialisasi arsitektur custom dari konfigurasi YAML dan melakukan Smart…, setup_ablation_model(), attach_wise_iou_to_model(), Modul Wise-IoU Loss (WIoU v1 & Dynamic Non-Monotonic WIoU v3) untuk YOLOv8.… (+2 more)

### Community 3 - "WiseIoUBboxLoss"
Cohesion: 0.20
Nodes (7): BboxLoss, Tensor, Wise-IoU Loss v3 dengan mekanisme pembobotan dinamis non-monotonik. Formula:…, Menghitung loss WIoU v3 untuk pred dan target dalam format xyxy. Args: pred:…, Adapter penurunan BboxLoss Ultralytics yang menggantikan CIoU dengan Wise-IoU…, WiseIoUBboxLoss, WiseIoULoss

### Community 4 - "train_research.py"
Cohesion: 0.23
Nodes (11): convert_visdrone_bbox_to_yolo(), download_and_extract_visdrone(), prepare_visdrone_subset(), Path, Modul Manajemen Dataset Citra UAV VisDrone2019-DET. Menyediakan fungsionalitas…, Menyiapkan dataset citra UAV VisDrone2019-DET asli dan menghasilkan path…, Mengonversi bounding box VisDrone (left, top, width, height) ke format YOLO…, Mengunduh berkas arsip VisDrone2019-DET-val.zip jika belum ada dan… (+3 more)

### Community 5 - "eda.py"
Cohesion: 0.33
Nodes (6): analyze_uav_labels(), plot_uav_eda(), Any, Modul Exploratory Data Analysis (EDA) Karakteristik Objek Citra UAV. Menghitung…, Menganalisis file anotasi YOLO (.txt) untuk mendeteksi karakteristik objek…, Menghasilkan visualisasi 4-panel berstandar publikasi (300 DPI) yang…

### Community 6 - "embed_execution_outputs.py"
Cohesion: 0.33
Nodes (5): make_html_display(), make_image_display(), DataFrame, Script to embed verified execution results, tables, and base64 figures directly…, to_base64()

### Community 9 - "🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss"
Cohesion: 0.10
Nodes (20): 10. Strategi Dekomposisi Riset: Transformasi 1 Eksperimen Notebook Menjadi 3 Artikel Jurnal SINTA 3, 11. Master Prompt AI untuk Pembuatan Artikel Jurnal (Siap Salin), 12. Panduan Langkah-demi-Langkah untuk Peneliti Pemula, 13. Glosarium Istilah Penting untuk Pemula, 1. Ringkasan Eksekutif & Karakteristik Citra Drone (UAV), 3. Jawaban Kritis atas Anomali Hasil Awal (Smoke Test vs Research Run), 4. Struktur Repositori & Higienitas Berkas, 5. Bedah Tuntas Sel-per-Sel Notebook (Sel 1 – Sel 18) & Makna Output (+12 more)

### Community 10 - "7. Bedah Menyeluruh Seluruh Output: Kurva, Confusion Matrix, Precision, Recall, F1, Accuracy, & MAPE"
Cohesion: 0.33
Nodes (6): 1. Analisis Kurva Konvergensi Pelatihan (Gambar 1), 2. Analisis Confusion Matrix (Matriks Kebingungan), 3. Analisis Precision, Recall, dan F1-Score, 4. Kedudukan Istilah "Accuracy" dalam Deteksi Objek, 5. Klarifikasi Kritis Mengenai Metrik MAPE (Mean Absolute Percentage Error), 7. Bedah Menyeluruh Seluruh Output: Kurva, Confusion Matrix, Precision, Recall, F1, Accuracy, & MAPE

### Community 11 - "2. Rencana Kerja Bertahap (Phased Action Plan)"
Cohesion: 0.25
Nodes (7): 1. Masalah yang Teridentifikasi & Analisis Akar Masalah (Root Causes), 2. Rencana Kerja Bertahap (Phased Action Plan), Fase 1: Perbaikan Modul Inti & Penyambungan Transfer Learning, Fase 2: Pengadaan Subset Dataset VisDrone Asli yang Ringan & Realistis, Fase 3: Rekonstruksi Berkas Notebook & Eksekusi Pelatihan Terarah, Fase 4: Finalisasi Repositori & Dokumentasi Publikasi, Implementation Plan: Transformasi Eksperimen Menuju Standar Publikasi Ilmiah SINTA / Scopus

### Community 12 - "6. Katalog Tabel Publikasi & Data Empiris Lengkap (Tabel 1 – Tabel 5)"
Cohesion: 0.33
Nodes (6): 6. Katalog Tabel Publikasi & Data Empiris Lengkap (Tabel 1 – Tabel 5), Tabel 1: Spesifikasi Arsitektur & Kompleksitas Model pada Studi Ablasi, Tabel 2: Evaluasi Kinerja Kuantitatif Studi Ablasi pada Test Set VisDrone, Tabel 3: Kinerja Kategori Objek Citra UAV pada Test Set VisDrone (Per-Class AP50), Tabel 4: Benchmark Efisiensi Komputasi Hardware (GPU NVIDIA RTX 2050 4GB VRAM), Tabel 5: Hasil Inferensi Statistik (Paired Bootstrap Resampling, B=1.000 Iterasi)

### Community 14 - "2. Triad Inovasi Arsitektur & Formulasi Matematika Lengkap (LaTeX)"
Cohesion: 0.33
Nodes (6): 2.1 Metrik Evaluasi Deteksi Objek Kecil, 2.2 Hierarki Fitur Resolusi Tinggi P2 (Stride 4), 2.3 Formulasi Matematis CBAM (Convolutional Block Attention Module), 2.4 Formulasi Matematis Wise-IoU Loss (WIoU v3), 2.5 Inferensi Statistik Paired Bootstrap (95% CI), 2. Triad Inovasi Arsitektur & Formulasi Matematika Lengkap (LaTeX)

## Knowledge Gaps
- **38 isolated node(s):** `📌 Daftar Isi`, `1. Ringkasan Eksekutif & Karakteristik Citra Drone (UAV)`, `2.1 Metrik Evaluasi Deteksi Objek Kecil`, `2.2 Hierarki Fitur Resolusi Tinggi P2 (Stride 4)`, `2.3 Formulasi Matematis CBAM (Convolutional Block Attention Module)` (+33 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 81 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss` connect `🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss` to `7. Bedah Menyeluruh Seluruh Output: Kurva, Confusion Matrix, Precision, Recall, F1, Accuracy, & MAPE`, `6. Katalog Tabel Publikasi & Data Empiris Lengkap (Tabel 1 – Tabel 5)`, `2. Triad Inovasi Arsitektur & Formulasi Matematika Lengkap (LaTeX)`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `register_cbam_to_ultralytics()` connect `evaluate_research.py` to `models.py`, `train_research.py`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `WiseIoULoss` connect `WiseIoUBboxLoss` to `models.py`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **What connects `📌 Daftar Isi`, `1. Ringkasan Eksekutif & Karakteristik Citra Drone (UAV)`, `2.1 Metrik Evaluasi Deteksi Objek Kecil` to the rest of the system?**
  _38 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._