"""
Runner verifikasi headless untuk membuktikan notebook dapat di-'Run All' 100% mulus tanpa error.
Mengeksekusi notebook 'Optimasi_Deteksi_Objek_Kecil_UAV_YOLOv8_P2_CBAM_WiseIoU.ipynb'
dan menyimpan notebook lengkap dengan grafik visualisasi, metrik, dan tabelnya.
"""

import io
import os
import sys
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

# Enforce UTF-8 stdout/stderr on Windows to avoid charmap UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

notebook_file = "Optimasi_Deteksi_Objek_Kecil_UAV_YOLOv8_P2_CBAM_WiseIoU.ipynb"

print("=" * 70)
print(f"[*] MEMULAI VERIFIKASI HEADLESS 'RUN ALL' PADA: {notebook_file}")
print("=" * 70)

t0 = time.time()
with open(notebook_file, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=7200, kernel_name="python3")

try:
    print(f"[*] Mengeksekusi seluruh {len(nb.cells)} sel (Markdown & Code) pada kernel python3 dalam PROFILE='smoke'...")
    ep.preprocess(nb, {"metadata": {"path": "."}})
    elapsed = time.time() - t0
    print(f"\n[+] SELURUH SEL NOTEBOOK BERHASIL DIEKSEKUSI DALAM {elapsed:.1f} DETIK!")

    # Simpan kembali notebook yang sudah terisi output lengkap
    with open(notebook_file, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[+] Output grafis, tabel, dan log berhasil disimpan ke '{notebook_file}'.")

except Exception as exc:
    safe_msg = str(exc).encode("ascii", "replace").decode("ascii")
    print(f"\n[ERROR] Terjadi kegagalan saat eksekusi: {safe_msg}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
