# Task Checklist: Transformasi Eksperimen Menuju Publikasi Ilmiah

## Status Terkini
- [x] Pembersihan Berkas Redundant (`yolo11n.pt` & folder sampah)
- [x] Penghapusan Berkas `PANDUAN_PENULISAN_JURNAL_AI.md` (digabungkan utuh ke `README.md`)
- [x] Pembuatan Berkas `README.md` Super Lengkap (Bedah Sel 1–18, LaTeX, Tabel Publikasi, Master Prompt AI)
- [x] **Knowledge Graph**: Inisialisasi & Query Graphify (`graphify-out/graph.json` & HTML)
- [x] **Fase 1: Rekayasa Arsitektur, Smart Transfer & Stabilisasi Modul (SELESAI)**
  - [x] Task 1.1: Perbaikan Bug Metrik $\text{mAP}_{small}$ di `modules/metrics.py` (filter prediksi area $<1024\text{ px}^2$, multi-threshold IoU 0.50:0.95)
  - [x] Task 1.2: Implementasi Smart Weight Transfer fungsional 100% di `modules/models.py` (Backbone + Top-Down Neck + Bottom-Up Neck + Heads)
  - [x] Task 1.3: Stabilisasi Wise-IoU Loss v3 di `modules/wise_iou.py` (momentum EMA 0.05, clamp dinamis aman `[0.2, 2.5]`)
  - [x] Task 1.4: Regenerasi 4 checkpoint inisialisasi baru di `outputs/research/models/init_*.pt` dengan Smart Weight Transfer terverifikasi
- [x] **Fase 2: Standarisasi Protokol Pelatihan & Evaluasi (SELESAI)**
  - [x] Task 2.1: Standarisasi `scripts/train_research.py` (`batch=2` seragam seluruh model, optimizer AdamW, `cos_lr=True`, `lr0=0.005`, `close_mosaic=10`, 50 epoch)
  - [x] Task 2.2: Perbaikan evaluasi paired bootstrap di `scripts/evaluate_research.py` menggunakan seluruh 98 citra test (`max_imgs=None`) & ekspor JSON
  - [x] Task 2.3: Perbaikan `scripts/embed_execution_outputs.py` agar membaca hasil empiris dinamis (tanpa teks statis hardcoded)
- [x] **Fase 3: Eksekusi Pelatihan Terarah (GPU RTX 2050) (SELESAI)**
  - [x] Task 3.1: Eksekusi pelatihan 4 model ablasi (M1, M2, M3, M4)
  - [x] Task 3.2: Eksekusi evaluasi komparatif & regenerasi grafik konvergensi riil
- [x] **Fase 4: Harmonisasi Manuskrip Publikasi (SELESAI)**
  - [x] Task 4.1: Sinkronisasi data empiris riil ke `README.md` (hapus klaim fiktif)
  - [x] Task 4.2: Pembaruan grafik dan tabel inline ke notebook `.ipynb`
  - [x] Task 4.3: Pembaruan Knowledge Graph (`graphify update .`)
- [x] **Fase 5: Pembersihan Repositori & Higienitas Berkas (SELESAI)**
  - [x] Menghapus `yolo11n.pt` (model tidak relevan)
  - [x] Menghapus `runs/` (folder temporary validasi)
  - [x] Menghapus `outputs/smoke/` dan `outputs/research/models/` (checkpoint usang)
  - [x] Menghapus `outputs/research/runs_legacy_20ep/` (backup legacy lama)
  - [x] Menghapus `scripts/run_phase3_evaluation.py` (skrip duplikat)
