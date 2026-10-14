# Ringkasan Data Tabel 1 - 5: Manuskrip Jurnal 1

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
