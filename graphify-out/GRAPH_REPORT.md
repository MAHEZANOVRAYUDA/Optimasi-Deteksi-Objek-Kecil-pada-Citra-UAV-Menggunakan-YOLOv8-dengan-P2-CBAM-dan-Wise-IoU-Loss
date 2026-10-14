# Graph Report - Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss  (2026-10-09)

## Corpus Check
- 15 files · ~7,057,719 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 140 nodes · 171 edges · 15 communities (12 shown, 3 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- evaluate_research.py
- cbam.py
- register_cbam_to_ultralytics
- wise_iou.py
- train_research.py
- eda.py
- embed_execution_outputs.py
- build_notebook.py
- run_headless_verification.py
- 🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss
- 10. Strategi Dekomposisi Riset: Blueprint Publikasi 3 Artikel Jurnal SINTA 3 / SINTA 2
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
9. `ChannelAttention` - 5 edges
10. `SpatialAttention` - 5 edges

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
Cohesion: 0.19
Nodes (15): compute_map_small(), compute_per_image_accuracy_scores(), export_latex_table(), paired_bootstrap_significance(), Any, DataFrame, Modul Metrik Kuantitatif, Evaluasi Statistik Bootstrap & Ekspor Format…, Menghitung skor akurasi deteksi lokalisasi (F1-score @ IoU 0.50) untuk setiap… (+7 more)

### Community 1 - "cbam.py"
Cohesion: 0.19
Nodes (8): CBAM, ChannelAttention, Tensor, Modul Perhatian CBAM (Convolutional Block Attention Module) untuk YOLOv8.…, Channel Attention Module (CAM). Menghitung bobot kepentingan antar-channel…, Spatial Attention Module (SAM). Menghitung bobot spasial untuk memusatkan…, Convolutional Block Attention Module (CBAM) lengkap. Menerapkan Channel…, SpatialAttention

### Community 2 - "register_cbam_to_ultralytics"
Cohesion: 0.25
Nodes (10): Mendaftarkan modul CBAM secara dinamis ke ekosistem Ultralytics sehingga model…, register_cbam_to_ultralytics(), prepare_pretrained_checkpoint(), Path, Utility module for preparing and initializing ablation models with smart weight…, Sets up a fully configured YOLO model for ablation training using native…, Menginisialisasi arsitektur custom dari konfigurasi YAML dan melakukan Smart…, setup_ablation_model() (+2 more)

### Community 3 - "wise_iou.py"
Cohesion: 0.18
Nodes (8): BboxLoss, Tensor, Modul Wise-IoU Loss (WIoU v1 & Dynamic Non-Monotonic WIoU v3) untuk YOLOv8.…, Wise-IoU Loss v3 dengan mekanisme pembobotan dinamis non-monotonik. Formula:…, Menghitung loss WIoU v3 untuk pred dan target dalam format xyxy. Args: pred:…, Adapter penurunan BboxLoss Ultralytics yang menggantikan CIoU dengan Wise-IoU…, WiseIoUBboxLoss, WiseIoULoss

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
Cohesion: 0.08
Nodes (24): 11. Master Prompt AI untuk Pembuatan Manuskrip Jurnal 1 (Siap Salin), 12. Panduan Taktis Peneliti Menghadapi Reviewer Jurnal, 13. Glosarium Istilah Ilmiah Komprehensif, 1.1 Tantangan Domain Small Object Detection (SOD) pada Wahana Nirawak (UAV), 1.2 Audit Manifest & Integritas Partisi Dataset VisDrone, 1. Pembingkaian Ilmiah yang Tepat: High-Precision UAV Detector, 1. Ringkasan Eksekutif, Karakteristik Citra Drone, & Audit Dataset VisDrone, 2. Analisis Pergeseran Kuadran Confusion Matrix (+16 more)

### Community 10 - "10. Strategi Dekomposisi Riset: Blueprint Publikasi 3 Artikel Jurnal SINTA 3 / SINTA 2"
Cohesion: 0.50
Nodes (4): 10. Strategi Dekomposisi Riset: Blueprint Publikasi 3 Artikel Jurnal SINTA 3 / SINTA 2, 📄 ARTIKEL 1: Paper Utama (Fokus: Rekayasa Arsitektur & Studi Ablasi), 📄 ARTIKEL 2: Visual & Interpretasi Fitur (Fokus: Explainable AI & Background Clutter), 📄 ARTIKEL 3: Engineering & Sistem Tertanam (Fokus: Efisiensi Komputasi Hardware & Edge Deployment)

### Community 11 - "2. Rencana Kerja Bertahap (Phased Action Plan)"
Cohesion: 0.25
Nodes (7): 1. Masalah yang Teridentifikasi & Analisis Akar Masalah (Root Causes), 2. Rencana Kerja Bertahap (Phased Action Plan), Fase 1: Perbaikan Modul Inti & Penyambungan Transfer Learning, Fase 2: Pengadaan Subset Dataset VisDrone Asli yang Ringan & Realistis, Fase 3: Rekonstruksi Berkas Notebook & Eksekusi Pelatihan Terarah, Fase 4: Finalisasi Repositori & Dokumentasi Publikasi, Implementation Plan: Transformasi Eksperimen Menuju Standar Publikasi Ilmiah SINTA / Scopus

### Community 12 - "6. Katalog Tabel Publikasi & Data Empiris Lengkap (Tabel 1 – Tabel 5)"
Cohesion: 0.33
Nodes (6): 6. Katalog Tabel Publikasi & Data Empiris Lengkap (Tabel 1 – Tabel 5), Tabel 1: Spesifikasi Arsitektur & Kompleksitas Model pada Studi Ablasi, Tabel 2: Evaluasi Kinerja Kuantitatif Studi Ablasi pada Test Set VisDrone (98 Citra), Tabel 3: Kinerja Kategori Objek Citra UAV pada Test Set VisDrone (Per-Class AP50), Tabel 4: Benchmark Efisiensi Komputasi Hardware (GPU NVIDIA RTX 2050 4GB VRAM), Tabel 5: Hasil Inferensi Statistik (Paired Bootstrap Resampling, B=1.000 Iterasi)

### Community 14 - "2. Triad Inovasi Arsitektur & Formulasi Matematika Lengkap (LaTeX)"
Cohesion: 0.33
Nodes (6): 2.1 Metrik Evaluasi Deteksi Objek Kecil, 2.2 Hierarki Fitur Resolusi Tinggi P2 (Stride 4), 2.3 Formulasi Matematis CBAM (Convolutional Block Attention Module), 2.4 Formulasi Matematis Wise-IoU Loss (WIoU v3), 2.5 Inferensi Statistik Paired Bootstrap (95% CI), 2. Triad Inovasi Arsitektur & Formulasi Matematika Lengkap (LaTeX)

## Knowledge Gaps
- **38 isolated node(s):** `📌 Daftar Isi`, `1.1 Tantangan Domain Small Object Detection (SOD) pada Wahana Nirawak (UAV)`, `1.2 Audit Manifest & Integritas Partisi Dataset VisDrone`, `2.1 Metrik Evaluasi Deteksi Objek Kecil`, `2.2 Hierarki Fitur Resolusi Tinggi P2 (Stride 4)` (+33 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 81 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss` connect `🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss` to `10. Strategi Dekomposisi Riset: Blueprint Publikasi 3 Artikel Jurnal SINTA 3 / SINTA 2`, `6. Katalog Tabel Publikasi & Data Empiris Lengkap (Tabel 1 – Tabel 5)`, `2. Triad Inovasi Arsitektur & Formulasi Matematika Lengkap (LaTeX)`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Why does `register_cbam_to_ultralytics()` connect `register_cbam_to_ultralytics` to `evaluate_research.py`, `cbam.py`, `train_research.py`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **What connects `📌 Daftar Isi`, `1.1 Tantangan Domain Small Object Detection (SOD) pada Wahana Nirawak (UAV)`, `1.2 Audit Manifest & Integritas Partisi Dataset VisDrone` to the rest of the system?**
  _38 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `🛰️ Optimasi Deteksi Objek Kecil pada Citra UAV Menggunakan YOLOv8 dengan P2 Detection Layer, CBAM Attention, dan Wise-IoU Loss` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._