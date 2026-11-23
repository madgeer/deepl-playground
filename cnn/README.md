# CNN (Convolutional Neural Network) dengan PyTorch

Implementasi jaringan syaraf tiruan CNN sederhana untuk klasifikasi citra menggunakan [PyTorch](https://pytorch.org/).

## Struktur File

```
cnn/
├── model.py       # Definisi model CNN
├── train.py       # Skrip pelatihan (konfigurasi di-hardcode di atas file)
├── inference.py   # Skrip prediksi/inferensi (konfigurasi di-hardcode di atas file)
└── README.md      # Dokumentasi ini
```

## Instalasi

Pastikan PyTorch sudah terinstal, disarankan menggunakan conda environment `env-dl`:

```bash
conda activate env-dl
pip install torch
```

## Cara Pakai

Semua skrip **tidak menggunakan argumen command line** (`argparse`). Konfigurasi diubah langsung di bagian atas masing-masing file (bagian `KONFIGURASI`), lalu jalankan langsung:

```bash
conda activate env-dl
cd D:\notebook-playground\deepl\cnn
python train.py
python inference.py
```

### 1. Pelatihan model (`train.py`)

Edit bagian konfigurasi di atas file `train.py`:

```python
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
IN_CHANNELS = 1          # 1 = grayscale, 3 = RGB
IMG_SIZE = 28            # Ukuran citra (asumsi persegi, mis. 28x28)
NUM_CLASSES = 10
DROPOUT = 0.5
DEVICE = None            # None = otomatis (cuda jika tersedia, else cpu)
SAVE_PATH = "cnn_model.pth"
NUM_SAMPLES = 2000
VAL_SPLIT = 0.2
```

Jalankan:

```bash
python train.py
```

Model terbaik akan otomatis disimpan ke `SAVE_PATH` (default: `cnn_model.pth`).

### 2. Inferensi / Prediksi (`inference.py`)

Edit bagian konfigurasi di atas file `inference.py`:

```python
MODEL_PATH = "cnn_model.pth"
IN_CHANNELS = 1
IMG_SIZE = 28
NUM_CLASSES = 10
DROPOUT = 0.5
DEVICE = None            # None = otomatis (cuda jika tersedia, else cpu)
NUM_SAMPLES = 10
```

Jalankan:

```bash
python inference.py
```

## Contoh Penggunaan sebagai Modul

```python
from model import create_model
import torch

# Buat model (untuk citra grayscale 28x28, 10 kelas)
model = create_model(in_channels=1, num_classes=10, dropout=0.5)

# Forward pass
x = torch.randn(32, 1, 28, 28)  # batch 32, 1 channel, 28x28
output = model(x)               # shape: (32, 10)
print(output.shape)             # torch.Size([32, 10])
```

## Arsitektur Model

```
Input (1, 28, 28)
    │
    ▼
Conv2d(1 -> 32) -> ReLU -> MaxPool2d   (32, 14, 14)
    │
    ▼
Conv2d(32 -> 64) -> ReLU -> MaxPool2d  (64, 7, 7)
    │
    ▼
Flatten
    │
    ▼
Linear(64*7*7 -> 128) -> ReLU -> Dropout
    │
    ▼
Linear(128 -> 10)                      (Output -> logits)
    │
    ▼
Output (10 kelas)
```

Catatan: Output model adalah **logits** (belum melalui softmax). Gunakan `nn.CrossEntropyLoss` selama pelatihan dan `torch.softmax` untuk mendapatkan probabilitas saat inferensi.

## Mengubah Optimizer (Adam)

Optimizer sudah pakai `optim.Aam` di dalam fungsi `train()` di `train.py`:

```python
optimizer = optim.Adam(model.parameters(), lr=LR)
```

Kalau ingin menambahkan parameter lain (weight_decay, betas, eps), ubah baris itu di `train.py` sebelum loop training dimulai. Tidak perlu import tambahan karena `torch.optim` sudah diimpor.

## Lisensi

MIT