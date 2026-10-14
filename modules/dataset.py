"""
Modul Manajemen Dataset Citra UAV VisDrone2019-DET.
Menyediakan fungsionalitas otomatis:
1. Unduh dataset VisDrone2019 resmi dari Ultralytics assets release (77.86 MB).
2. Konversi format anotasi VisDrone ke format YOLO (kategori 1-10 -> 0-9).
3. Partisi data split (train, val, test) dengan citra drone resolusi asli.
4. Pembuatan berkas konfigurasi dataset data_uav.yaml.
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Any
import yaml
from PIL import Image


# 10 Kelas Standar VisDrone-DET
VISDRONE_CLASSES = [
    'pedestrian',
    'people',
    'bicycle',
    'car',
    'van',
    'truck',
    'tricycle',
    'awning-tricycle',
    'bus',
    'motor'
]

VISDRONE_VAL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"


def convert_visdrone_bbox_to_yolo(img_size: Tuple[int, int], box: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    """
    Mengonversi bounding box VisDrone (left, top, width, height) ke format YOLO (cx, cy, w, h normalisasi [0, 1]).
    """
    img_w, img_h = img_size
    dw = 1.0 / max(img_w, 1)
    dh = 1.0 / max(img_h, 1)
    cx = (box[0] + box[2] / 2.0) * dw
    cy = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(w, 0.0), 1.0),
        min(max(h, 0.0), 1.0)
    )


def download_and_extract_visdrone(download_dir: Path) -> Path:
    """
    Mengunduh berkas arsip VisDrone2019-DET-val.zip jika belum ada dan mengekstraknya.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = download_dir / "VisDrone2019-DET-val.zip"
    extract_dir = download_dir / "VisDrone2019-DET-val"

    if not extract_dir.exists():
        if not zip_dest.exists():
            print(f"[*] Mengunduh dataset resmi VisDrone2019-DET dari GitHub ({VISDRONE_VAL_URL})...")
            
            def report_hook(count, block_size, total_size):
                if total_size > 0:
                    pct = int(count * block_size * 100 / total_size)
                    if count % 100 == 0:
                        print(f"    -> Progress download: {min(pct, 100)}% ({count*block_size//(1024*1024)} MB)", end="\r")

            urllib.request.urlretrieve(VISDRONE_VAL_URL, zip_dest, reporthook=report_hook)
            print(f"\n[+] Unduhan selesai: {zip_dest} ({zip_dest.stat().st_size / (1024*1024):.2f} MB)")

        print(f"[*] Mengekstrak {zip_dest.name}...")
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(download_dir)
        print(f"[+] Ekstraksi selesai di: {extract_dir}")

    return extract_dir


def prepare_visdrone_subset(
    raw_dir: Path,
    target_root: Path,
    n_train: int = 350,
    n_val: int = 100,
    n_test: int = 98
) -> None:
    """
    Mengonversi anotasi VisDrone dan mempartisi 548 citra ke train/val/test secara rapi.
    """
    raw_img_dir = raw_dir / "images"
    raw_ann_dir = raw_dir / "annotations"

    img_files = sorted(list(raw_img_dir.glob("*.jpg")))
    total_imgs = len(img_files)
    print(f"[*] Ditemukan {total_imgs} citra drone asli di VisDrone2019-DET.")

    # Tentukan partisi split
    splits = {
        "train": img_files[:n_train],
        "val": img_files[n_train:n_train + n_val],
        "test": img_files[n_train + n_val:n_train + n_val + n_test]
    }

    for split_name, files in splits.items():
        img_out_dir = target_root / "images" / split_name
        lbl_out_dir = target_root / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            dest_img = img_out_dir / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            # Konversi anotasi
            ann_path = raw_ann_dir / f"{img_path.stem}.txt"
            lbl_dest = lbl_out_dir / f"{img_path.stem}.txt"

            yolo_lines = []
            if ann_path.exists():
                try:
                    with Image.open(img_path) as im:
                        im_size = im.size
                except Exception:
                    im_size = (640, 640)

                with open(ann_path, "r", encoding="utf-8") as f_ann:
                    for line in f_ann:
                        parts = line.strip().split(",")
                        if len(parts) >= 6:
                            # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,...
                            cat_id = int(parts[5])
                            # Kategori 1..10 dipetakan ke 0..9 (pedestrian s/d motor)
                            # Kategori 0 (ignored) dan 11 (others) diabaikan
                            if 1 <= cat_id <= 10:
                                yolo_cls = cat_id - 1
                                box = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
                                if box[2] > 2 and box[3] > 2:  # filter noise box < 2px
                                    cx, cy, w, h = convert_visdrone_bbox_to_yolo(im_size, box)
                                    yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            with open(lbl_dest, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(yolo_lines))

    print(f"[+] Dataset VisDrone2019 berhasil disiapkan: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test.")


def setup_uav_dataset(base_dir: str = "data/uav", profile: str = "smoke") -> str:
    """
    Menyiapkan dataset citra UAV VisDrone2019-DET asli dan menghasilkan path data_uav.yaml.
    """
    root = Path(base_dir).resolve()
    yaml_path = root / "data_uav.yaml"

    # Periksa apakah citra riil VisDrone sudah ada di images/train
    train_imgs = list((root / "images" / "train").glob("*.jpg"))
    has_real_data = len(train_imgs) >= 30 and not any("uav_train" in f.name for f in train_imgs[:5])

    if not has_real_data:
        # Bersihkan data sintetis lama jika ada
        for p in [root / "images", root / "labels"]:
            if p.exists():
                shutil.rmtree(p)

        cache_dir = Path("data/downloads").resolve()
        raw_visdrone_dir = download_and_extract_visdrone(cache_dir)
        prepare_visdrone_subset(raw_visdrone_dir, root, n_train=350, n_val=100, n_test=98)

    # Buat / perbarui data_uav.yaml
    dataset_dict = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(VISDRONE_CLASSES)}
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_dict, f, default_flow_style=False, width=1000)

    return str(yaml_path).replace("\\", "/")


def audit_scene_leakage(base_dir: str = "data/uav") -> Dict[str, Any]:
    """
    Melakukan audit ilmiah terhadap kebocoran sekuens video / adegan (Scene Leakage)
    antar partisi train, val, dan test.
    """
    root = Path(base_dir)
    scenes = {}
    counts = {}
    for s in ["train", "val", "test"]:
        p = root / "images" / s
        files = list(p.glob("*.jpg"))
        counts[s] = len(files)
        scenes[s] = set(f.name.split("_")[0] for f in files)

    overlap_train_val = sorted(list(scenes.get("train", set()) & scenes.get("val", set())))
    overlap_train_test = sorted(list(scenes.get("train", set()) & scenes.get("test", set())))
    overlap_val_test = sorted(list(scenes.get("val", set()) & scenes.get("test", set())))

    return {
        "file_counts": counts,
        "unique_scenes": {s: len(scenes.get(s, set())) for s in scenes},
        "leakage_train_val": overlap_train_val,
        "leakage_train_test": overlap_train_test,
        "leakage_val_test": overlap_val_test,
        "train_test_clean": len(overlap_train_test) == 0
    }

