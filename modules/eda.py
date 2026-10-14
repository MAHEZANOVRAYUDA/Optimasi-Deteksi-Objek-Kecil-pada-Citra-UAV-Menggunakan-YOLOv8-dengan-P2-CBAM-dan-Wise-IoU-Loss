"""
Modul Exploratory Data Analysis (EDA) Karakteristik Objek Citra UAV.
Menghitung statistik skala objek (kategori MS-COCO: Small, Medium, Large),
aspek rasio, dan kepadatan spasial bounding box.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def analyze_uav_labels(labels_dir: str, img_size: int = 640) -> Dict[str, Any]:
    """
    Menganalisis file anotasi YOLO (.txt) untuk mendeteksi karakteristik objek kecil UAV.
    """
    lbl_path = Path(labels_dir)
    txt_files = list(lbl_path.glob("*.txt"))

    areas_px = []
    aspect_ratios = []
    cx_list = []
    cy_list = []
    class_counts = {}

    small_count = 0   # Area < 32^2 = 1024
    medium_count = 0  # 1024 <= Area < 96^2 = 9216
    large_count = 0   # Area >= 9216

    for f in txt_files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    w_px = w * img_size
                    h_px = h * img_size
                    area_px = w_px * h_px

                    areas_px.append(area_px)
                    aspect_ratios.append(w_px / max(h_px, 1e-4))
                    cx_list.append(cx)
                    cy_list.append(cy)
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

                    if area_px < 1024:
                        small_count += 1
                    elif area_px < 9216:
                        medium_count += 1
                    else:
                        large_count += 1

    total = max(len(areas_px), 1)
    return {
        "total_objects": len(areas_px),
        "small_count": small_count,
        "medium_count": medium_count,
        "large_count": large_count,
        "small_pct": (small_count / total) * 100,
        "medium_pct": (medium_count / total) * 100,
        "large_pct": (large_count / total) * 100,
        "areas_px": np.array(areas_px),
        "aspect_ratios": np.array(aspect_ratios),
        "cx": np.array(cx_list),
        "cy": np.array(cy_list),
        "class_counts": class_counts
    }


def plot_uav_eda(stats: Dict[str, Any], save_path: str = "outputs/figures/eda_uav_objects.png") -> str:
    """
    Menghasilkan visualisasi 4-panel berstandar publikasi (300 DPI)
    yang membuktikan dominasi small objects pada citra UAV.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # Panel 1: Distribusi Skala Kategori COCO
    categories = ["Small (<32 px)", "Medium (32-96 px)", "Large (>96 px)"]
    pcts = [stats["small_pct"], stats["medium_pct"], stats["large_pct"]]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = axes[0, 0].bar(categories, pcts, color=colors, edgecolor="black", width=0.5)
    axes[0, 0].set_ylabel("Persentase Objek (%)", fontsize=11, fontweight="bold")
    axes[0, 0].set_title("(a) Distribusi Skala Objek Berdasarkan Kriteria COCO", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylim(0, 105)
    for bar in bars:
        yval = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Panel 2: Histogram Luas Bounding Box
    if len(stats["areas_px"]) > 0:
        axes[0, 1].hist(stats["areas_px"], bins=30, color="#3498db", edgecolor="black", alpha=0.85)
        axes[0, 1].axvline(1024, color="red", linestyle="--", linewidth=2, label="Batas Small (<1024 px²)")
        axes[0, 1].set_xlabel("Luas Bounding Box (Piksel²)", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Frekuensi Objek", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("(b) Sebaran Luas Objek Bounding Box", fontsize=12, fontweight="bold")
        axes[0, 1].legend()

    # Panel 3: Distribusi Aspek Rasio
    if len(stats["aspect_ratios"]) > 0:
        axes[1, 0].hist(stats["aspect_ratios"], bins=30, color="#9b59b6", edgecolor="black", alpha=0.85, range=(0, 3))
        axes[1, 0].set_xlabel("Aspek Rasio (Lebar / Tinggi)", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Frekuensi Objek", fontsize=11, fontweight="bold")
        axes[1, 0].set_title("(c) Sebaran Aspek Rasio Objek", fontsize=12, fontweight="bold")

    # Panel 4: Heatmap Titik Pusat Spasial (cx, cy)
    if len(stats["cx"]) > 0:
        h, xedges, yedges = np.histogram2d(stats["cx"], stats["cy"], bins=25, range=[[0, 1], [0, 1]])
        im = axes[1, 1].imshow(h.T, origin="lower", extent=[0, 1, 0, 1], cmap="hot_r", aspect="auto")
        fig.colorbar(im, ax=axes[1, 1], label="Kepadatan Objek")
        axes[1, 1].set_xlabel("Koordinat X Normalisasi", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Koordinat Y Normalisasi", fontsize=11, fontweight="bold")
        axes[1, 1].set_title("(d) Kepadatan Spasial Lokasi Objek", fontsize=12, fontweight="bold")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path
