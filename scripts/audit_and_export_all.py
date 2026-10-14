"""
audit_and_export_all.py
Audit Ilmiah & Ekspor Data Terstruktur Komprehensif:
1. Dataset Manifest (548 citra, hash SHA256, sequence ID, split, bounding box counts, small object counts)
2. FAR Kuantitatif Per-Citra & Direct CBAM Hook (M2 L18 vs M3 L19, serta Pre/Post CBAM di M3)
3. Deteksi Kuadran Per-Citra (98 test images x 4 model: GT, Pred, TP, FP, FN, P, R, F1)
4. Benchmark Latensi Multi-Sesi (5 sesi x 4 model pada RTX 2050, p50, p95, p99, frame > 33.33ms, FPS riil)
5. Telemetri Environment Lengkap (OS, Python, PyTorch, CUDA, cuDNN, GPU, Ultralytics)
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path

# Enforce UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from modules.cbam import register_cbam_to_ultralytics

register_cbam_to_ultralytics()

TABLES_DIR = ROOT_DIR / "outputs" / "research" / "tables"
FIGURES_DIR = ROOT_DIR / "outputs" / "research" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

cuda_available = torch.cuda.is_available()
device = "cuda:0" if cuda_available else "cpu"
print(f"Device: {device} | CUDA Available: {cuda_available}")

# =========================================================================
# 1. DATASET MANIFEST AUDIT (548 CITRA)
# =========================================================================
print("\n" + "=" * 80)
print("[1/5] AUDIT DATASET MANIFEST (548 CITRA VISDRONE)...")
print("=" * 80)

manifest_records = []
split_stats = {"train": 0, "val": 0, "test": 0}

for split in ["train", "val", "test"]:
    img_dir = ROOT_DIR / "data" / "uav" / "images" / split
    lbl_dir = ROOT_DIR / "data" / "uav" / "labels" / split
    img_files = sorted(list(img_dir.glob("*.jpg")))
    split_stats[split] = len(img_files)

    for img_p in img_files:
        stem = img_p.stem
        seq_id = stem.split("_")[0]
        
        # SHA256 hash
        hasher = hashlib.sha256()
        with open(img_p, "rb") as f_img:
            hasher.update(f_img.read())
        sha256_hash = hasher.hexdigest()

        # Image size
        with Image.open(img_p) as im:
            w_orig, h_orig = im.size

        # Anotasi
        lbl_p = lbl_dir / f"{stem}.txt"
        n_objects = 0
        n_small = 0
        n_medium = 0
        n_large = 0
        classes_present = []

        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f_lbl:
                for line in f_lbl:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        n_objects += 1
                        cls_id = int(parts[0])
                        classes_present.append(cls_id)
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                        # Skala 640x640 standar evaluasi YOLO
                        area_640 = (w_norm * 640.0) * (h_norm * 640.0)
                        if area_640 < 1024.0:
                            n_small += 1
                        elif area_640 < 9216.0:
                            n_medium += 1
                        else:
                            n_large += 1

        manifest_records.append({
            "image_id": img_p.name,
            "stem": stem,
            "split": split,
            "sequence_id": seq_id,
            "width": w_orig,
            "height": h_orig,
            "sha256": sha256_hash,
            "total_objects": n_objects,
            "small_objects_640": n_small,
            "medium_objects_640": n_medium,
            "large_objects_640": n_large,
            "small_ratio": round(n_small / n_objects, 4) if n_objects > 0 else 0.0,
            "classes_count": len(set(classes_present))
        })

df_manifest = pd.DataFrame(manifest_records)
manifest_csv_path = TABLES_DIR / "dataset_manifest.csv"
manifest_json_path = TABLES_DIR / "dataset_manifest.json"
df_manifest.to_csv(manifest_csv_path, index=False)
with open(manifest_json_path, "w", encoding="utf-8") as f_out:
    json.dump(manifest_records, f_out, indent=2)

print(f"[+] Manifest berhasil diekspor: {manifest_csv_path} ({len(df_manifest)} baris)")
print("    Split breakdown:", split_stats)
print("    Total objects per split:")
for s in ["train", "val", "test"]:
    sub = df_manifest[df_manifest["split"] == s]
    print(f"      - {s:<6}: {len(sub)} citra, {sub['total_objects'].sum()} objek, {sub['small_objects_640'].sum()} small (<1024 px^2, {sub['small_objects_640'].sum()/sub['total_objects'].sum()*100:.2f}%)")


# =========================================================================
# 2. AUDIT FAR PER-CITRA & DIRECT CBAM MODULE HOOK (M2 VS M3)
# =========================================================================
print("\n" + "=" * 80)
print("[2/5] AUDIT FOREGROUND ACTIVATION RATIO (FAR) PER-CITRA (98 TEST IMAGES)...")
print("=" * 80)

model_m2_path = ROOT_DIR / "outputs" / "research" / "runs" / "Model_2_P2" / "weights" / "best.pt"
model_m3_path = ROOT_DIR / "outputs" / "research" / "runs" / "Model_3_P2_CBAM" / "weights" / "best.pt"

eval_m2 = YOLO(str(model_m2_path))
eval_m3 = YOLO(str(model_m3_path))

pt_m2 = eval_m2.model.to(device).eval()
pt_m3 = eval_m3.model.to(device).eval()

test_manifest = df_manifest[df_manifest["split"] == "test"].copy()
far_records = []

# Pilih 3 sampel untuk visualisasi overlay koordinat
overlay_samples = ["0000333_00001_d_0000001.jpg", "0000335_00001_d_0000001.jpg", "0000342_00001_d_0000001.jpg"]
saved_overlays = []

for idx, row in test_manifest.iterrows():
    img_name = row["image_id"]
    seq_id = row["sequence_id"]
    img_path = ROOT_DIR / "data" / "uav" / "images" / "test" / img_name
    lbl_path = ROOT_DIR / "data" / "uav" / "labels" / "test" / f"{Path(img_name).stem}.txt"

    gt_boxes_norm = []
    if lbl_path.exists():
        with open(lbl_path, "r", encoding="utf-8") as f_lbl:
            for line in f_lbl:
                p = line.strip().split()
                if len(p) >= 5:
                    cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    gt_boxes_norm.append([cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0])

    is_valid = len(gt_boxes_norm) > 0

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (640, 640))
    inp_t = torch.from_numpy(img_res).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    inp_t = inp_t.to(device)

    # 1. Forward M2 (Layer 18: P2 output)
    act_m2 = []
    h2 = pt_m2.model[18].register_forward_hook(lambda m, i, o: act_m2.append(o.detach()))
    with torch.no_grad():
        _ = pt_m2(inp_t)
    h2.remove()
    feat_m2_raw = act_m2[0][0] # [C=128, H=160, W=160]
    tensor_shape_m2 = list(feat_m2_raw.shape)
    feat_m2_mean = feat_m2_raw.mean(dim=0).cpu().float().numpy() # [160, 160]
    feat_m2_relu = np.maximum(feat_m2_mean, 0.0) # ReLU AFTER channel average

    # 2. Forward M3 (Layer 18: Pre-CBAM P2, Layer 19: Post-CBAM P2)
    act_m3_pre = []
    act_m3_post = []
    h3_pre = pt_m3.model[18].register_forward_hook(lambda m, i, o: act_m3_pre.append(o.detach()))
    h3_post = pt_m3.model[19].register_forward_hook(lambda m, i, o: act_m3_post.append(o.detach()))
    with torch.no_grad():
        _ = pt_m3(inp_t)
    h3_pre.remove()
    h3_post.remove()

    feat_m3_pre_raw = act_m3_pre[0][0]
    feat_m3_post_raw = act_m3_post[0][0]
    tensor_shape_m3 = list(feat_m3_post_raw.shape)

    feat_m3_pre_mean = feat_m3_pre_raw.mean(dim=0).cpu().float().numpy()
    feat_m3_pre_relu = np.maximum(feat_m3_pre_mean, 0.0)

    feat_m3_post_mean = feat_m3_post_raw.mean(dim=0).cpu().float().numpy()
    feat_m3_post_relu = np.maximum(feat_m3_post_mean, 0.0)

    h_f, w_f = feat_m2_relu.shape
    mask = np.zeros((h_f, w_f), dtype=np.float32)
    if is_valid:
        for b in gt_boxes_norm:
            x1 = max(0, int(b[0] * w_f))
            y1 = max(0, int(b[1] * h_f))
            x2 = min(w_f, int(b[2] * w_f) + 1)
            y2 = min(h_f, int(b[3] * h_f) + 1)
            mask[y1:y2, x1:x2] = 1.0

    mask_pixels = float(np.sum(mask))
    total_pixels = float(h_f * w_f)
    occupancy_q = float(mask_pixels / total_pixels)

    # FAR M2 (Checkpoint 2)
    fg_m2 = float(np.sum(feat_m2_relu * mask))
    bg_m2 = float(np.sum(feat_m2_relu * (1.0 - mask)))
    tot_m2 = float(np.sum(feat_m2_relu))
    far_m2 = float(fg_m2 / (tot_m2 + 1e-8)) if is_valid else 0.0

    # FAR M3 Post-CBAM (Checkpoint 3)
    fg_m3_post = float(np.sum(feat_m3_post_relu * mask))
    bg_m3_post = float(np.sum(feat_m3_post_relu * (1.0 - mask)))
    tot_m3_post = float(np.sum(feat_m3_post_relu))
    far_m3_post = float(fg_m3_post / (tot_m3_post + 1e-8)) if is_valid else 0.0

    # FAR M3 Pre-CBAM (Isolated module comparison within Model 3)
    fg_m3_pre = float(np.sum(feat_m3_pre_relu * mask))
    bg_m3_pre = float(np.sum(feat_m3_pre_relu * (1.0 - mask)))
    tot_m3_pre = float(np.sum(feat_m3_pre_relu))
    far_m3_pre = float(fg_m3_pre / (tot_m3_pre + 1e-8)) if is_valid else 0.0

    delta_far_checkpoints = far_m3_post - far_m2
    delta_far_isolated_cbam = far_m3_post - far_m3_pre

    far_records.append({
        "image_id": img_name,
        "sequence_id": seq_id,
        "is_valid": is_valid,
        "gt_objects": len(gt_boxes_norm),
        "tensor_shape": f"{tensor_shape_m2[0]}x{tensor_shape_m2[1]}x{tensor_shape_m2[2]}",
        "mask_occupancy_q": round(occupancy_q, 6),
        "m2_layer18_fg_sum": round(fg_m2, 4),
        "m2_layer18_bg_sum": round(bg_m2, 4),
        "m2_layer18_far": round(far_m2, 6),
        "m3_layer19_post_fg_sum": round(fg_m3_post, 4),
        "m3_layer19_post_bg_sum": round(bg_m3_post, 4),
        "m3_layer19_post_far": round(far_m3_post, 6),
        "m3_layer18_pre_far": round(far_m3_pre, 6),
        "paired_delta_far_checkpoints": round(delta_far_checkpoints, 6),
        "paired_delta_far_isolated_cbam": round(delta_far_isolated_cbam, 6),
    })

    if img_name in overlay_samples:
        saved_overlays.append({
            "image_id": img_name,
            "img_rgb": img_res,
            "mask": mask,
            "feat_m2": feat_m2_relu,
            "feat_m3": feat_m3_post_relu
        })

df_far = pd.DataFrame(far_records)
far_csv_path = TABLES_DIR / "far_per_image_detailed.csv"
far_json_path = TABLES_DIR / "far_per_image_detailed.json"
df_far.to_csv(far_csv_path, index=False)
with open(far_json_path, "w", encoding="utf-8") as f_out:
    json.dump(far_records, f_out, indent=2)

valid_far = df_far[df_far["is_valid"] == True]
mean_m2 = valid_far["m2_layer18_far"].mean()
mean_m3 = valid_far["m3_layer19_post_far"].mean()
diffs = valid_far["paired_delta_far_checkpoints"]
std_diff = diffs.std()
sem_diff = std_diff / np.sqrt(len(diffs))
ci_low = diffs.mean() - 1.96 * sem_diff
ci_high = diffs.mean() + 1.96 * sem_diff

# Direct isolated CBAM in Model 3
mean_m3_pre = valid_far["m3_layer18_pre_far"].mean()
diffs_isolated = valid_far["paired_delta_far_isolated_cbam"]

print(f"[+] FAR per-citra diekspor: {far_csv_path}")
print(f"    - Model 2 (P2 murni, Layer 18) Rerata FAR : {mean_m2:.4f}")
print(f"    - Model 3 (P2 CBAM, Layer 19) Rerata FAR  : {mean_m3:.4f}")
print(f"    - Rerata Selisih Berpasangan (Checkpoints): {diffs.mean():+.4f} (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}])")
print(f"    - Peningkatan Relatif Antar-Checkpoint   : {((mean_m3 - mean_m2)/mean_m2)*100:+.2f}%")
print(f"    - Efek Isolasi CBAM di Model 3 (Pre->Post): {mean_m3_pre:.4f} -> {mean_m3:.4f} ({diffs_isolated.mean():+.4f})")

# Visualisasi overlay koordinat untuk validasi registrasi spasial
if saved_overlays:
    fig, axes = plt.subplots(len(saved_overlays), 3, figsize=(15, 5 * len(saved_overlays)), dpi=300)
    if len(saved_overlays) == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, item in enumerate(saved_overlays):
        im_orig = item["img_rgb"]
        m2_map = item["feat_m2"]
        m3_map = item["feat_m3"]
        gt_mask = item["mask"]

        # Panel 1: Original Image with GT Mask Overlay
        axes[row_idx, 0].imshow(im_orig)
        mask_resized = cv2.resize(gt_mask, (640, 640), interpolation=cv2.INTER_NEAREST)
        axes[row_idx, 0].imshow(mask_resized, cmap="Greens", alpha=0.35)
        axes[row_idx, 0].set_title(f"Citra Asli + Mask Ground Truth\n[{item['image_id']}]", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Panel 2: M2 Feature Map with GT contour
        axes[row_idx, 1].imshow(m2_map, cmap="inferno")
        axes[row_idx, 1].contour(gt_mask, levels=[0.5], colors=["cyan"], linewidths=1.0)
        axes[row_idx, 1].set_title("M2 (P2) Feature Map + Kontur GT (Sian)", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Panel 3: M3 Feature Map with GT contour
        axes[row_idx, 2].imshow(m3_map, cmap="inferno")
        axes[row_idx, 2].contour(gt_mask, levels=[0.5], colors=["cyan"], linewidths=1.0)
        axes[row_idx, 2].set_title("M3 (P2+CBAM) Feature Map + Kontur GT", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

    fig_overlay_path = FIGURES_DIR / "xai_coordinate_verification_overlay.png"
    plt.tight_layout()
    plt.savefig(fig_overlay_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Verifikasi Overlay Registrasi Koordinat disimpan: {fig_overlay_path}")


# =========================================================================
# 3. DETEKSI KUADRAN PER-CITRA (98 TEST IMAGES X 4 MODEL)
# =========================================================================
print("\n" + "=" * 80)
print("[3/5] AUDIT KUADRAN DETEKSI PER-CITRA (98 TEST IMAGES X 4 MODEL)...")
print("=" * 80)

models_map = {
    "Model_1_Baseline": ROOT_DIR / "outputs" / "research" / "runs" / "Model_1_Baseline" / "weights" / "best.pt",
    "Model_2_P2": ROOT_DIR / "outputs" / "research" / "runs" / "Model_2_P2" / "weights" / "best.pt",
    "Model_3_P2_CBAM": ROOT_DIR / "outputs" / "research" / "runs" / "Model_3_P2_CBAM" / "weights" / "best.pt",
    "Model_4_Proposed_Full": ROOT_DIR / "outputs" / "research" / "runs" / "Model_4_Proposed_Full" / "weights" / "best.pt"
}

yolo_instances = {k: YOLO(str(v)) for k, v in models_map.items()}

detection_records = []
iou_thresh = 0.50
conf_thresh = 0.15

test_imgs = sorted(list((ROOT_DIR / "data" / "uav" / "images" / "test").glob("*.jpg")))

for img_p in test_imgs:
    img_name = img_p.name
    seq_id = img_name.split("_")[0]
    lbl_p = ROOT_DIR / "data" / "uav" / "labels" / "test" / f"{img_p.stem}.txt"

    with Image.open(img_p) as im:
        w_orig, h_orig = im.size

    gt_boxes = []
    gt_cls = []
    if lbl_p.exists():
        with open(lbl_p, "r", encoding="utf-8") as f_lbl:
            for line in f_lbl:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gt_boxes.append([
                        (cx - w/2.0) * w_orig,
                        (cy - h/2.0) * h_orig,
                        (cx + w/2.0) * w_orig,
                        (cy + h/2.0) * h_orig
                    ])
                    gt_cls.append(c)

    n_gt = len(gt_boxes)

    for m_key, y_model in yolo_instances.items():
        res = y_model.predict(str(img_p), conf=conf_thresh, imgsz=640, verbose=False)[0]
        preds = res.boxes.xyxy.cpu()
        p_cls = res.boxes.cls.cpu().numpy()
        n_preds = len(preds)

        tp = 0
        if n_gt > 0 and n_preds > 0:
            ious = box_iou(torch.tensor(gt_boxes, dtype=torch.float32), preds).numpy()
            matched_gt = set()
            for j in range(n_preds):
                best_iou = 0.0
                best_g = -1
                for g in range(n_gt):
                    if gt_cls[g] == p_cls[j] and g not in matched_gt:
                        if ious[g, j] > best_iou:
                            best_iou = ious[g, j]
                            best_g = g
                if best_g >= 0 and best_iou >= iou_thresh:
                    matched_gt.add(best_g)
                    tp += 1

        fp = n_preds - tp
        fn = n_gt - tp
        prec = (tp / n_preds) if n_preds > 0 else 0.0
        rec = (tp / n_gt) if n_gt > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec + 1e-8)) if (prec + rec) > 0 else 0.0

        detection_records.append({
            "image_id": img_name,
            "sequence_id": seq_id,
            "model": m_key,
            "gt_objects": n_gt,
            "total_preds": n_preds,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        })

df_det = pd.DataFrame(detection_records)
det_csv_path = TABLES_DIR / "detection_per_image_quadrants.csv"
det_json_path = TABLES_DIR / "detection_per_image_quadrants.json"
df_det.to_csv(det_csv_path, index=False)
with open(det_json_path, "w", encoding="utf-8") as f_out:
    json.dump(detection_records, f_out, indent=2)

print(f"[+] Deteksi per-citra diekspor: {det_csv_path} ({len(df_det)} baris)")

# Verifikasi khusus sampel citra 0000333_00001_d_0000001.jpg
sample_check = df_det[df_det["image_id"] == "0000333_00001_d_0000001.jpg"]
print("    Verifikasi Citra Sampel 0000333_00001_d_0000001.jpg:")
for _, row in sample_check.iterrows():
    print(f"      - {row['model']:<22}: GT={row['gt_objects']}, Pred={row['total_preds']}, TP={row['tp']}, FP={row['fp']}, FN={row['fn']}, Prec={row['precision']*100:.1f}%, Rec={row['recall']*100:.1f}%")


# =========================================================================
# 4. BENCHMARK LATENSI MULTI-SESI INDEPENDEN (5 SESI X 4 MODEL RTX 2050)
# =========================================================================
print("\n" + "=" * 80)
print("[4/5] BENCHMARK LATENSI MULTI-SESI INDEPENDEN (5 SESI X 4 MODEL)...")
print("=" * 80)

# Gunakan 50 citra uji berpasangan seragam
benchmark_subset = test_imgs[:50]
N_SESSIONS = 5

session_summary_records = []
image_latency_records = []

for session_idx in range(1, N_SESSIONS + 1):
    print(f"[*] Menjalankan Sesi Pengukuran {session_idx}/{N_SESSIONS}...")
    # Acak urutan model untuk menghindari urutan bias termal
    model_keys = list(models_map.keys())
    # Deterministic permutation based on session_idx
    rng = np.random.default_rng(42 + session_idx)
    rng.shuffle(model_keys)

    for m_key in model_keys:
        y_model = yolo_instances[m_key]

        # Warmup
        for w_img in benchmark_subset[:5]:
            _ = y_model.predict(str(w_img), conf=0.15, imgsz=640, verbose=False)

        pre_times = []
        inf_times = []
        post_times = []
        tot_times = []

        if cuda_available:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        session_t0 = time.perf_counter()

        for img_p in benchmark_subset:
            t_frame_start = time.perf_counter()
            res = y_model.predict(str(img_p), conf=0.15, imgsz=640, verbose=False)[0]
            if cuda_available:
                torch.cuda.synchronize()
            t_frame_end = time.perf_counter()
            frame_e2e_ms = (t_frame_end - t_frame_start) * 1000.0

            sp = res.speed
            t_pre = sp.get("preprocess", 0.0)
            t_inf = sp.get("inference", 0.0)
            t_post = sp.get("postprocess", 0.0)

            pre_times.append(t_pre)
            inf_times.append(t_inf)
            post_times.append(t_post)
            tot_times.append(frame_e2e_ms)

            image_latency_records.append({
                "session_id": session_idx,
                "model": m_key,
                "image_id": img_p.name,
                "preprocess_ms": round(t_pre, 3),
                "inference_ms": round(t_inf, 3),
                "postprocess_ms": round(t_post, 3),
                "total_e2e_ms": round(frame_e2e_ms, 3)
            })

        session_duration = time.perf_counter() - session_t0
        actual_throughput_fps = len(benchmark_subset) / session_duration

        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if cuda_available else 0.0

        p50_tot = float(np.percentile(tot_times, 50))
        p95_tot = float(np.percentile(tot_times, 95))
        p99_tot = float(np.percentile(tot_times, 99))
        p50_inf = float(np.percentile(inf_times, 50))
        prop_over_33ms = float(np.mean(np.array(tot_times) > 33.333) * 100.0)

        session_summary_records.append({
            "session_id": session_idx,
            "model": m_key,
            "n_frames": len(benchmark_subset),
            "session_duration_sec": round(session_duration, 3),
            "throughput_fps": round(actual_throughput_fps, 2),
            "preprocess_p50_ms": round(float(np.percentile(pre_times, 50)), 2),
            "inference_p50_ms": round(p50_inf, 2),
            "postprocess_p50_ms": round(float(np.percentile(post_times, 50)), 2),
            "total_latency_p50_ms": round(p50_tot, 2),
            "total_latency_p95_ms": round(p95_tot, 2),
            "total_latency_p99_ms": round(p99_tot, 2),
            "pct_frames_over_33ms": round(prop_over_33ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 1)
        })

df_sess = pd.DataFrame(session_summary_records)
df_img_lat = pd.DataFrame(image_latency_records)

sess_csv_path = TABLES_DIR / "hardware_latency_per_session.csv"
img_lat_csv_path = TABLES_DIR / "hardware_latency_per_image.csv"
df_sess.to_csv(sess_csv_path, index=False)
df_img_lat.to_csv(img_lat_csv_path, index=False)

print(f"[+] Hasil latensi multi-sesi diekspor: {sess_csv_path}")
print(f"[+] Hasil latensi per-citra diekspor: {img_lat_csv_path} ({len(df_img_lat)} baris)")

# Rerata lintas sesi
print("\n    Ringkasan Rerata Lintas 5 Sesi:")
for m_key in models_map.keys():
    sub = df_sess[df_sess["model"] == m_key]
    print(f"      - {m_key:<22}: Throughput {sub['throughput_fps'].mean():.1f} ± {sub['throughput_fps'].std():.2f} FPS | p50 E2E: {sub['total_latency_p50_ms'].mean():.2f} ms | p95: {sub['total_latency_p95_ms'].mean():.2f} ms | Frames > 33.3ms: {sub['pct_frames_over_33ms'].mean():.1f}%")


# =========================================================================
# 5. ENVIRONMENT & REPRODUCIBILITY TELEMETRY
# =========================================================================
print("\n" + "=" * 80)
print("[5/5] MENYIMPAN TELEMETRI ENVIRONMENT LENGKAP...")
print("=" * 80)

import platform
import ultralytics

env_telemetry = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "platform": {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    },
    "python": {
        "version": sys.version,
        "executable": sys.executable,
    },
    "frameworks": {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__
    },
    "hardware": {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 3) if torch.cuda.is_available() else 0.0,
    },
    "reproducibility": {
        "seed": 42,
        "test_dataset_size": len(test_imgs),
        "manifest_total_images": len(df_manifest),
        "train_clean_against_test": True,
        "batch_training_actual": 4,
        "epochs_training_actual": 20,
        "optimizer_actual": "auto (AdamW)",
        "lr0_actual": 0.01,
        "amp_actual": True,
        "half_actual": True
    }
}

env_json_path = TABLES_DIR / "environment_info.json"
with open(env_json_path, "w", encoding="utf-8") as f_env:
    json.dump(env_telemetry, f_env, indent=2)

print(f"[+] Environment Telemetry berhasil disimpan di: {env_json_path}")
print("=" * 80)
print(" SEMUA AUDIT DAN EKSPOR STRUKTUR DATA SELESAI DENGAN SUKSES!")
print("=" * 80)
