#!/usr/bin/env python3
"""
package_jurnal_1_assets.py
Mengemas seluruh aset Gambar 1-4, Tabel 1-5 (CSV & Excel), Naskah DOCX,
dan Referensi RIS ke dalam satu folder terstruktur dan berkas ZIP submission.
"""

import os
import shutil
import zipfile
import pandas as pd

def build_package():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_dir = os.path.join(base_dir, "outputs", "paket_jurnal_1")
    os.makedirs(pkg_dir, exist_ok=True)
    
    naskah_dir = os.path.join(pkg_dir, "01_Naskah_dan_Referensi")
    gambar_dir = os.path.join(pkg_dir, "02_Gambar_Resolusi_Tinggi")
    tabel_dir = os.path.join(pkg_dir, "03_Tabel_Data_Kuantitatif")
    
    os.makedirs(naskah_dir, exist_ok=True)
    os.makedirs(gambar_dir, exist_ok=True)
    os.makedirs(tabel_dir, exist_ok=True)
    
    print("1. Menyalin Naskah DOCX dan Referensi RIS...")
    src_docx1 = os.path.join(base_dir, "Jurnal 1", "Draf_Jurnal_1_YOLOv8_P2_CBAM_WIoU_MIND_SINTA3.docx")
    src_docx2 = os.path.join(base_dir, "Jurnal 1", "Jurnal_1_Revisi_KomtekInfo.docx")
    src_ris = os.path.join(base_dir, "Jurnal 1", "Referensi_Jurnal_1_Mendeley.ris")
    
    if os.path.exists(src_docx1):
        shutil.copy2(src_docx1, os.path.join(naskah_dir, "Draf_Jurnal_1_YOLOv8_P2_CBAM_WIoU_MIND_SINTA3.docx"))
    if os.path.exists(src_docx2):
        shutil.copy2(src_docx2, os.path.join(naskah_dir, "Jurnal_1_Revisi_KomtekInfo.docx"))
    if os.path.exists(src_ris):
        shutil.copy2(src_ris, os.path.join(naskah_dir, "Referensi_Jurnal_1_Mendeley.ris"))

    print("2. Menyalin Gambar 1 - 4 Resolusi Tinggi...")
    figs_src = os.path.join(base_dir, "outputs", "research", "figures")
    
    img_mapping = [
        ("training_convergence_comparison.png", "Gambar_1_Kurva_Konvergensi_Pelatihan.png"),
        ("side_by_side_detection_comparison_opt.jpg", "Gambar_2_Visualisasi_Deteksi_Side_by_Side_Opt.jpg"),
        ("side_by_side_detection_comparison.png", "Gambar_2_Visualisasi_Deteksi_Side_by_Side_Lossless.png"),
        ("eda_uav_objects.png", "Gambar_3_Distribusi_Karakteristik_Dataset_EDA.png"),
        ("p2_cbam_feature_activation_map.png", "Gambar_4_Peta_Aktivasi_Fitur_P2_CBAM.png"),
        ("jurnal3_fig1_latency_breakdown.png", "Gambar_4_Alternatif_Dekomposisi_Latensi_Hardware.png")
    ]
    
    for src_name, dst_name in img_mapping:
        src_path = os.path.join(figs_src, src_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(gambar_dir, dst_name))
            print(f"  -> Disalin: {dst_name}")
        else:
            print(f"  [WARNING] File tidak ditemukan: {src_path}")

    print("3. Membuat Tabel 1 - 5 dalam format CSV dan Excel...")
    # Tabel 1
    t1_data = {
        "Model ID": ["M1", "M2", "M3", "M4"],
        "Konfigurasi Arsitektur": ["Baseline YOLOv8s", "YOLOv8s + P2", "YOLOv8s + P2 + CBAM", "Proposed Full (WIoU v3)"],
        "Detection Heads": ["3 Heads (P3, P4, P5)", "4 Heads (P2–P5)", "4 Heads (P2–P5)", "4 Heads (P2–P5)"],
        "Atensi (Attention)": ["Tanpa Atensi", "Tanpa Atensi", "CBAM pada P2 & P3", "CBAM pada P2 & P3"],
        "Fungsi Kerugian": ["CIoU + DFL", "CIoU + DFL", "CIoU + DFL", "WIoU v3 + DFL"],
        "Parameter (M)": [11.13, 10.63, 10.64, 10.64],
        "GFLOPs": [28.6, 36.7, 37.0, 37.0]
    }
    df_t1 = pd.DataFrame(t1_data)
    df_t1.to_csv(os.path.join(tabel_dir, "Tabel_1_Konfigurasi_Arsitektur.csv"), index=False, encoding="utf-8-sig")

    # Tabel 2
    t2_data = {
        "Model ID": ["M1", "M2", "M3", "M4"],
        "Konfigurasi Model": ["Baseline YOLOv8s", "+ P2 Layer (Stride 4)", "+ P2 + CBAM Attention", "Proposed Full (+ WIoU v3)"],
        "Precision (%)": [37.90, 48.48, 39.75, 48.68],
        "Recall (%)": [27.74, 18.86, 20.14, 19.26],
        "mAP@50 (%)": [25.86, 16.24, 17.05, 15.96],
        "mAP@50-95 (%)": [14.81, 8.15, 8.61, 7.72],
        "mAP_small (%)": [21.98, 13.75, 14.37, 14.16]
    }
    df_t2 = pd.DataFrame(t2_data)
    df_t2.to_csv(os.path.join(tabel_dir, "Tabel_2_Evaluasi_Kuantitatif_Ablasi.csv"), index=False, encoding="utf-8-sig")

    # Tabel 3
    t3_data = {
        "Model ID": ["M1 Baseline", "M2 (+ P2)", "M3 (+ CBAM)", "M4 Proposed"],
        "Pedestrian (AP50)": [27.36, 22.99, 23.76, 24.50],
        "People (AP50)": [31.51, 27.60, 27.36, 23.80],
        "Bicycle (AP50)": [9.51, 2.14, 2.52, 3.49],
        "Motor (AP50)": [33.08, 29.64, 29.04, 25.80],
        "Car (AP50)": [74.91, 71.04, 69.42, 66.20],
        "Rerata Semua Kelas (mAP50)": [25.86, 16.24, 17.05, 15.96]
    }
    df_t3 = pd.DataFrame(t3_data)
    df_t3.to_csv(os.path.join(tabel_dir, "Tabel_3_Kinerja_Deteksi_Per_Kategori.csv"), index=False, encoding="utf-8-sig")

    # Tabel 4
    t4_data = {
        "Model": ["M1 Baseline", "M4 Proposed Full"],
        "Total Prediksi": [89, 48],
        "True Positive (TP)": [38, 35],
        "False Positive (FP)": [51, 13],
        "False Negative (FN)": [29, 32],
        "Presisi Riil (%)": [42.7, 72.9],
        "Reduksi False Alarm": ["0.0% (Ref)", "-74.5% (Pangkas 38 FP)"]
    }
    df_t4 = pd.DataFrame(t4_data)
    df_t4.to_csv(os.path.join(tabel_dir, "Tabel_4_Kuadran_Deteksi_Riil.csv"), index=False, encoding="utf-8-sig")

    # Tabel 5
    t5_data = {
        "Model": ["M1 Baseline", "M2 (+ P2)", "M3 (+ CBAM)", "M4 Proposed Full"],
        "Prapemrosesan": ["1.90 ms", "1.81 ms", "1.84 ms", "1.74 ms"],
        "Inferensi Tensor": ["9.57 ms", "13.50 ms", "14.13 ms", "14.11 ms"],
        "NMS Pascaproses": ["1.74 ms", "1.51 ms", "1.59 ms", "1.53 ms"],
        "Total Latensi (p50)": ["22.91 ms", "25.76 ms", "27.11 ms", "26.99 ms"],
        "Throughput (FPS)": [43.7, 38.8, 36.9, 37.1],
        "Peak VRAM": ["248.6 MB", "268.4 MB", "269.0 MB", "269.0 MB"],
        "Status Real-Time (>30 FPS)": ["Memenuhi", "Memenuhi", "Memenuhi", "Memenuhi (+23.7% Margin)"]
    }
    df_t5 = pd.DataFrame(t5_data)
    df_t5.to_csv(os.path.join(tabel_dir, "Tabel_5_Benchmark_Hardware_RTX2050.csv"), index=False, encoding="utf-8-sig")

    excel_path = os.path.join(tabel_dir, "Tabel_1_hingga_5_Lengkap.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_t1.to_excel(writer, sheet_name="Tabel 1 Konfigurasi", index=False)
        df_t2.to_excel(writer, sheet_name="Tabel 2 Evaluasi Ablasi", index=False)
        df_t3.to_excel(writer, sheet_name="Tabel 3 Per Kategori", index=False)
        df_t4.to_excel(writer, sheet_name="Tabel 4 Kuadran Riil", index=False)
        df_t5.to_excel(writer, sheet_name="Tabel 5 Hardware RTX2050", index=False)
    print(f"  -> File Excel berhasil dibuat: {excel_path}")

    # Buat file teks markdown ringkasan tabel
    md_content = """# Ringkasan Data Tabel 1 - 5: Manuskrip Jurnal 1

## Tabel 1. Konfigurasi arsitektur dan kompleksitas model pada studi ablasi
| Model ID | Konfigurasi Arsitektur | Detection Heads | Atensi (Attention) | Fungsi Kerugian | Parameter (M) | GFLOPs |
|---|---|---|---|---|---|---|
| M1 | Baseline YOLOv8s | 3 Heads (P3, P4, P5) | Tanpa Atensi | CIoU + DFL | 11.13 M | 28.6 |
| M2 | YOLOv8s + P2 | 4 Heads (P2–P5) | Tanpa Atensi | CIoU + DFL | 10.63 M | 36.7 |
| M3 | YOLOv8s + P2 + CBAM | 4 Heads (P2–P5) | CBAM pada P2 & P3 | CIoU + DFL | 10.64 M | 37.0 |
| M4 | Proposed Full (WIoU v3) | 4 Heads (P2–P5) | CBAM pada P2 & P3 | WIoU v3 + DFL | 10.64 M | 37.0 |

---

## Tabel 2. Evaluasi kinerja kuantitatif studi ablasi pada test set VisDrone (98 citra)
| Model ID | Konfigurasi Model | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50-95 (%) | mAP_small (%) |
|---|---|---|---|---|---|---|
| M1 | Baseline YOLOv8s | 37.90% | 27.74% | 25.86% | 14.81% | 21.98% |
| M2 | + P2 Layer (Stride 4) | 48.48% | 18.86% | 16.24% | 8.15% | 13.75% |
| M3 | + P2 + CBAM Attention | 39.75% | 20.14% | 17.05% | 8.61% | 14.37% |
| M4 | Proposed Full (+ WIoU v3) | 48.68% | 19.26% | 15.96% | 7.72% | 14.16% |

---

## Tabel 3. Kinerja deteksi per-kelas objek citra UAV pada test set VisDrone (AP@50 dalam persen)
| Model ID | Pedestrian | People | Bicycle | Motor | Car | Rerata Semua Kelas |
|---|---|---|---|---|---|---|
| M1 Baseline | 27.36% | 31.51% | 9.51% | 33.08% | 74.91% | 25.86% |
| M2 (+ P2) | 22.99% | 27.60% | 2.14% | 29.64% | 71.04% | 16.24% |
| M3 (+ CBAM) | 23.76% | 27.36% | 2.52% | 29.04% | 69.42% | 17.05% |
| M4 Proposed | 24.50% | 23.80% | 3.49% | 25.80% | 66.20% | 15.96% |

---

## Tabel 4. Evaluasi kuadran deteksi riil dan eliminasi false positive pada citra uji representatif
| Model | Total Prediksi | True Positive (TP) | False Positive (FP) | False Negative (FN) | Presisi Riil (%) | Reduksi False Alarm |
|---|---|---|---|---|---|---|
| M1 Baseline | 89 | 38 | 51 | 29 | 42.7% | 0.0% (Ref) |
| M4 Proposed Full | 48 | 35 | 13 | 32 | 72.9% | -74.5% (Pangkas 38 FP) |

---

## Tabel 5. Benchmark efisiensi komputasi hardware end-to-end pada GPU NVIDIA GeForce RTX 2050
| Model | Prapemrosesan | Inferensi Tensor | NMS Pascaproses | Total Latensi (p50) | Throughput (FPS) | Peak VRAM | Status Real-Time |
|---|---|---|---|---|---|---|---|
| M1 Baseline | 1.90 ms | 9.57 ms | 1.74 ms | 22.91 ms | 43.7 FPS | 248.6 MB | Memenuhi (>30 FPS) |
| M2 (+ P2) | 1.81 ms | 13.50 ms | 1.51 ms | 25.76 ms | 38.8 FPS | 268.4 MB | Memenuhi (>30 FPS) |
| M3 (+ CBAM) | 1.84 ms | 14.13 ms | 1.59 ms | 27.11 ms | 36.9 FPS | 269.0 MB | Memenuhi (>30 FPS) |
| M4 Proposed Full | 1.74 ms | 14.11 ms | 1.53 ms | 26.99 ms | 37.1 FPS | 269.0 MB | Memenuhi (+23.7% Margin) |
"""
    with open(os.path.join(tabel_dir, "Ringkasan_Tabel_1_sampai_5.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    # 4. Buat README panduan paket
    readme_content = """PAKET SUBMISSION JURNAL ILMIAH
Judul: Rekayasa Arsitektur YOLOv8 Berbasis P2-Layer, CBAM, dan Wise-IoU Loss untuk Peningkatan Presisi Deteksi Objek Mikro pada Citra UAV
Penulis: Maheza Novrayuda
Afiliasi: Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Putra Indonesia YPTK Padang
Email: maheza.novrayuda@upiyptk.ac.id

STRUKTUR BERKAS DALAM PAKET INI:
========================================================================
1. 01_Naskah_dan_Referensi/
   - Draf_Jurnal_1_YOLOv8_P2_CBAM_WIoU_MIND_SINTA3.docx: Naskah lengkap format MIND Journal (SINTA 3), margin 4cm, Tahoma 11pt, ukuran < 2MB.
   - Jurnal_1_Revisi_KomtekInfo.docx: Naskah alternatif format Jurnal KomtekInfo.
   - Referensi_Jurnal_1_Mendeley.ris: File bibliografi digital standar Mendeley/Zotero lengkap dengan link DOI.

2. 02_Gambar_Resolusi_Tinggi/
   - Gambar_1_Kurva_Konvergensi_Pelatihan.png: Grafik perbandingan konvergensi Box Loss dan Val mAP50 sepanjang 20 epoch.
   - Gambar_2_Visualisasi_Deteksi_Side_by_Side_Opt.jpg: Visualisasi komparatif side-by-side pemangkasan alarm palsu (-74.5% FP) terkompresi optimal.
   - Gambar_2_Visualisasi_Deteksi_Side_by_Side_Lossless.png: Versi PNG lossless resolusi penuh untuk cetak publisher.
   - Gambar_3_Distribusi_Karakteristik_Dataset_EDA.png: Grafik 4-panel karakteristik objek VisDrone (84.80% mikro).
   - Gambar_4_Peta_Aktivasi_Fitur_P2_CBAM.png: Visualisasi pergeseran konsentrasi energi aktivasi fitur berkat modul CBAM (FAR +17.96%).
   - Gambar_4_Alternatif_Dekomposisi_Latensi_Hardware.png: Grafik waktu end-to-end pada GPU RTX 2050.

3. 03_Tabel_Data_Kuantitatif/
   - Tabel_1_hingga_5_Lengkap.xlsx: Seluruh Tabel 1-5 dalam 1 file Excel (multi-sheet) siap olah.
   - Tabel_1_Konfigurasi_Arsitektur.csv
   - Tabel_2_Evaluasi_Kuantitatif_Ablasi.csv
   - Tabel_3_Kinerja_Deteksi_Per_Kategori.csv
   - Tabel_4_Kuadran_Deteksi_Riil.csv
   - Tabel_5_Benchmark_Hardware_RTX2050.csv
   - Ringkasan_Tabel_1_sampai_5.md: Ringkasan teks berformat tabel markdown.
========================================================================
"""
    with open(os.path.join(pkg_dir, "README_PAKET_SUBMISSION.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("4. Mengompresi seluruh paket ke dalam ZIP...")
    zip_path = os.path.join(base_dir, "Paket_Lengkap_Jurnal_1_YOLOv8_P2_CBAM_WIoU.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(pkg_dir):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file, os.path.dirname(pkg_dir))
                zipf.write(abs_file, rel_path)
    
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n[SELESAI] Paket ZIP berhasil dibuat di: {zip_path} ({zip_size_mb:.2f} MB)")

if __name__ == '__main__':
    build_package()
