"""
Skrip Eksekusi Pelatihan Riset Terkontrol (Studi Ablasi 4 Model)
Profil: 'research' (20 Epoch, Batch 4, AMP FP16, RTX 2050 4GB VRAM)
Model:
1. Model_1_Baseline: YOLOv8s (3 heads: P3, P4, P5 + CIoU)
2. Model_2_P2: YOLOv8s + P2 Layer (4 heads: P2, P3, P4, P5 + CIoU)
3. Model_3_P2_CBAM: YOLOv8s + P2 + CBAM Attention (4 heads + CIoU)
4. Model_4_Proposed_Full: YOLOv8s + P2 + CBAM + Wise-IoU v3 Loss
"""

import sys
import time
import gc
from pathlib import Path

# Enforce UTF-8 stdout/stderr on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import pandas as pd
from ultralytics import YOLO

from modules.cbam import register_cbam_to_ultralytics
from modules.models import setup_ablation_model
from modules.dataset import setup_uav_dataset

# 1. Pastikan reproduktibilitas
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_available else 0.0

print("=" * 75)
print(" EKSEKUTOR PELATIHAN RISET ABLASI KONTROL (PROFIL: RESEARCH)")
print("=" * 75)
print(f"[*] Perangkat Akselerator : {device_name}")
print(f"[*] Total VRAM Tersedia   : {vram_gb:.2f} GB")
print(f"[*] Seed Reproduktibilitas: {SEED}")
print("=" * 75)

# 2. Parameter Riset
EPOCHS = 20  # Standar evaluasi 20 epoch terkontrol (fair comparison terhadap baseline)
BATCH_SIZE = 4  # Seragam di seluruh model (VRAM 2.3GB/4GB di RTX 2050, 100% aman dan stabil)
IMGSZ = 640
WORKERS = 0
HALF = True  # AMP FP16
PATIENCE = 10

OUTPUT_DIR = Path("outputs/research")
RUNS_DIR = OUTPUT_DIR / "runs"
MODELS_DIR = OUTPUT_DIR / "models"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

for d in [RUNS_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Pastikan dataset YAML
DATASET_YAML = setup_uav_dataset(base_dir="data/uav", profile="research")
print(f"[*] Dataset YAML: {DATASET_YAML}")

ablation_experiments = [
    {
        "name": "Model_1_Baseline",
        "cfg": "configs/yolov8s-baseline.yaml",
        "use_wiou": False,
        "desc": "Baseline YOLOv8s (3 heads: P3, P4, P5 + CIoU)"
    },
    {
        "name": "Model_2_P2",
        "cfg": "configs/yolov8s-p2.yaml",
        "use_wiou": False,
        "desc": "YOLOv8s + P2 Layer (4 heads: P2, P3, P4, P5 + CIoU)"
    },
    {
        "name": "Model_3_P2_CBAM",
        "cfg": "configs/yolov8s-p2-cbam.yaml",
        "use_wiou": False,
        "desc": "YOLOv8s + P2 + CBAM Attention (4 heads + CIoU)"
    },
    {
        "name": "Model_4_Proposed_Full",
        "cfg": "configs/yolov8s-p2-cbam.yaml",
        "use_wiou": True,
        "desc": "YOLOv8s + P2 + CBAM + Wise-IoU Loss (Proposed Full)"
    }
]

total_start_time = time.time()
trained_weights = {}

import shutil

for idx, exp in enumerate(ablation_experiments, 1):
    exp_name = exp["name"]
    print(f"\n{'='*75}")
    print(f" [{idx}/4] PELATIHAN MODEL: {exp_name}")
    print(f" Deskripsi: {exp['desc']}")
    print(f"{'='*75}")

    # Bersihkan VRAM sebelum mulai model baru
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()

    exp_dir = RUNS_DIR / exp_name
    csv_check = exp_dir / "results.csv"
    best_check = exp_dir / "weights" / "best.pt"
    if csv_check.exists() and best_check.exists():
        try:
            df_chk = pd.read_csv(csv_check)
            if len(df_chk) >= EPOCHS:
                print(f"[+] Model '{exp_name}' sudah terlatih lengkap ({len(df_chk)} epoch). Melanjutkan ke model berikutnya.")
                trained_weights[exp_name] = str(best_check)
                continue
            else:
                print(f"[*] Membersihkan run lama ({len(df_chk)} epoch) agar training mulai bersih dengan Smart Transfer...")
                shutil.rmtree(exp_dir, ignore_errors=True)
        except Exception:
            pass

    t0 = time.time()
    model = setup_ablation_model(
        model_name=exp_name,
        yaml_cfg=exp["cfg"],
        base_pt_path="yolov8s.pt",
        use_wiou=exp["use_wiou"],
        models_dir=MODELS_DIR,
        force_rebuild=False
    )

    print(f"[*] Memulai training {EPOCHS} epoch (batch={BATCH_SIZE}, imgsz={IMGSZ}, AMP FP16)...")
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,
        patience=PATIENCE,
        workers=WORKERS,
        half=HALF,
        project=str(RUNS_DIR),
        name=exp_name,
        seed=SEED,
        verbose=True,
        exist_ok=True
    )

    elapsed = time.time() - t0
    best_pt = RUNS_DIR / exp_name / "weights" / "best.pt"
    trained_weights[exp_name] = str(best_pt) if best_pt.exists() else None
    print(f"\n[+] Selesai melatih {exp_name} dalam {elapsed/60:.2f} menit ({elapsed:.1f} detik).")
    print(f"[+] Bobot terbaik disimpan di: {trained_weights[exp_name]}")

    del model
    del results
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()

total_elapsed = time.time() - total_start_time
print(f"\n{'='*75}")
print(f"[+] SELURUH 4 MODEL ABLASI BERHASIL DILATIH DALAM {total_elapsed/60:.2f} MENIT!")
for m_name, w_path in trained_weights.items():
    print(f"    - {m_name:<25}: {w_path}")
print(f"{'='*75}")
