#!/usr/bin/env python3
"""
generate_jurnal3_figures.py
Menghasilkan 3 visualisasi ilmiah beresolusi tinggi (300 DPI) khusus Jurnal 3
berdasarkan data empiris tervalidasi hardware_efficiency_benchmark.csv dan test_set_comparative_metrics.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('outputs/research/figures', exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

models = ['M1 (Baseline)', 'M2 (P2)', 'M3 (P2+CBAM)', 'M4 (Proposed Full)']

# Data Latency End-to-End
pre = [1.90, 1.81, 1.84, 1.74]
inf = [9.57, 13.50, 14.13, 14.11]
nms = [1.74, 1.51, 1.59, 1.53]
total_p50 = [22.91, 25.76, 27.11, 26.99]
total_p95 = [26.79, 27.45, 29.64, 29.64]
fps = [43.7, 38.8, 36.9, 37.1]
prec = [37.90, 48.48, 39.75, 48.68]
rec = [27.74, 18.86, 20.14, 19.26]
map50 = [25.86, 16.24, 17.05, 15.96]

# 1. Figure 1: Latency Breakdown & Budget 30 FPS
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
x = np.arange(len(models))
width = 0.55

p1 = ax.bar(x, pre, width, label='Prapemrosesan (Preprocess)', color='#4A90E2', edgecolor='#2A6EBB')
p2 = ax.bar(x, inf, width, bottom=pre, label='Inferensi Tensor (PyTorch)', color='#F5A623', edgecolor='#D4850C')
bottom_nms = [p + i for p, i in zip(pre, inf)]
p3 = ax.bar(x, nms, width, bottom=bottom_nms, label='Pascapemrosesan (NMS)', color='#7ED321', edgecolor='#5BA314')

# Error bar untuk p95 total
for i, (p50, p95) in enumerate(zip(total_p50, total_p95)):
    ax.plot([i, i], [p50, p95], color='#D0021B', lw=1.5)
    ax.scatter([i], [p95], color='#D0021B', s=25, zorder=5)
    ax.text(i, p95 + 0.8, f'{p50:.1f} ms\n({fps[i]:.1f} FPS)', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#111111')

# Garis ambang batas real-time 33.33 ms (30 FPS)
ax.axhline(33.33, color='#D0021B', linestyle='--', linewidth=1.5, label='Batas Anggaran Real-Time 30 FPS (33.33 ms)')
ax.text(3.4, 33.8, 'Target 30 FPS (33.33 ms)', color='#D0021B', fontweight='bold', fontsize=9, ha='right')

ax.set_ylabel('Latensi Komputasi (ms)', fontsize=11, fontweight='bold')
ax.set_title('Profil Latensi End-to-End Multi-Tahap pada GPU NVIDIA RTX 2050', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, fontweight='bold')
ax.set_ylim(0, 38)
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=9)
plt.tight_layout()
fig1_path = 'outputs/research/figures/jurnal3_fig1_latency_breakdown.png'
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f'Generated {fig1_path}')

# 2. Figure 2: Trade-off FPS vs Precision
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
colors = ['#4A90E2', '#50E3C2', '#9013FE', '#E74C3C']
markers = ['o', 's', '^', 'D']

for i in range(len(models)):
    ax.scatter(fps[i], prec[i], color=colors[i], s=120, marker=markers[i], edgecolor='#111111', lw=1.2, label=models[i], zorder=4)
    offset_x = 0.5
    offset_y = 0.6 if i != 2 else -1.2
    ax.annotate(f'{models[i]}\n({fps[i]:.1f} FPS, P: {prec[i]:.1f}%)', (fps[i] + offset_x, prec[i] + offset_y), fontsize=9, fontweight='bold')

ax.axvline(30.0, color='#D0021B', linestyle='--', linewidth=1.2, label='Standar Real-Time Minimum (30 FPS)')
ax.set_xlabel('Throughput Inferensi (FPS)', fontsize=11, fontweight='bold')
ax.set_ylabel('Presisi Deteksi (%)', fontsize=11, fontweight='bold')
ax.set_title('Matriks Trade-off Kinerja: Presisi Deteksi vs Throughput (FPS)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlim(30, 48)
ax.set_ylim(32, 54)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=9)
plt.tight_layout()
fig2_path = 'outputs/research/figures/jurnal3_fig2_fps_vs_precision.png'
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f'Generated {fig2_path}')

# 3. Figure 3: Precision vs Recall Bar Comparison
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
x = np.arange(len(models))
width = 0.35

rects1 = ax.bar(x - width/2, prec, width, label='Precision (%)', color='#2ECC71', edgecolor='#27AE60')
rects2 = ax.bar(x + width/2, rec, width, label='Recall (%)', color='#E67E22', edgecolor='#D35400')

ax.set_ylabel('Persentase Skor (%)', fontsize=11, fontweight='bold')
ax.set_title('Komparasi Metrik Precision dan Recall pada Test Set VisDrone (98 Citra)', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, fontweight='bold')
ax.set_ylim(0, 60)
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=9)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig3_path = 'outputs/research/figures/jurnal3_fig3_precision_recall_tradeoff.png'
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f'Generated {fig3_path}')
