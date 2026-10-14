"""
Modul Perhatian CBAM (Convolutional Block Attention Module) untuk YOLOv8.
Terdiri dari:
1. Channel Attention Module (CAM)
2. Spatial Attention Module (SAM)
3. CBAM Module Gabungan
4. Registrasi Dinamis ke Ekosistem Ultralytics
"""

import inspect
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).
    Menghitung bobot kepentingan antar-channel fitur dengan mengombinasikan
    Global Average Pooling dan Global Max Pooling yang diproses melalui Shared MLP.
    
    Formula:
        Mc(F) = Sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_pool = x.mean(dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).
    Menghitung bobot spasial untuk memusatkan perhatian detektor pada lokasi
    objek kecil di antara background clutter dengan konvolusi 7x7 pada fitur ter-pool.
    
    Formula:
        Ms(F) = Sigmoid(Conv7x7([AvgPool(F); MaxPool(F)]))
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size harus 3 atau 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat_features = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(cat_features))
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) lengkap.
    Menerapkan Channel Attention diikuti Spatial Attention secara sekuensial.
    
    Args:
        c1 (int): Jumlah input channel
        reduction (int): Faktor reduksi channel pada MLP (default 16)
        kernel_size (int): Ukuran kernel konvolusi spasial (default 7)
    """
    def __init__(self, c1: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


_CBAM_REGISTERED = False

def register_cbam_to_ultralytics() -> None:
    """
    Mendaftarkan modul CBAM secara dinamis ke ekosistem Ultralytics
    sehingga model YAML kustom dapat langsung di-parse oleh YOLO().
    Idempotent: aman dipanggil berulang kali tanpa error.
    """
    global _CBAM_REGISTERED
    setattr(modules, 'CBAM', CBAM)
    setattr(tasks, 'CBAM', CBAM)
    
    if _CBAM_REGISTERED:
        return

    try:
        src = inspect.getsource(tasks.parse_model)
        if 'elif m is AIFI or m is CBAM:' not in src:
            src_modified = src.replace('elif m is AIFI:', 'elif m is AIFI or m is CBAM:')
            exec(src_modified, tasks.__dict__)
    except (OSError, TypeError):
        pass

    _CBAM_REGISTERED = True
