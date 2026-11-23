# MLP (Multi-Layer Perceptron) dengan PyTorch

Implementasi jaringan syaraf tiruan MLP sederhana menggunakan [PyTorch](https://pytorch.org/).

## Struktur File

```
mlp/
├── model.py       # Definisi model MLP
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
cd D:\notebook-playground\deepl\mlp
python train.py
python inference.py
```

### 1. Pelatihan model (`train.py`)

Edit bagian konfigurasi di atas file `train.py`:

```python
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
INPUT_SIZE = 784
NUM_CLASSES = 10
HIDDEN_SIZES = [128, 64]
DROPOUT = 0.2
DEVICE = None          # None = otomatis (cuda jika tersedia, else cpu)
SAVE_PATH = "mlp_model.pth"
NUM_SAMPLES = 2000
VAL_SPLIT = 0.2
```

Jalankan:

```bash
python train.py
```

Model terbaik akan otomatis disimpan ke `SAVE_PATH` (default: `mlp_model.pth`).

### 2. Inferensi / Prediksi (`inference.py`)

Edit bagian konfigurasi di atas file `inference.py`:

```python
MODEL_PATH = "mlp_model.pth"
INPUT_SIZE = 784
NUM_CLASSES = 10
HIDDEN_SIZES = [128, 64]
DEVICE = None          # None = otomatis (cuda jika tersedia, else cpu)
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

# Buat model
model = create_model(input_size=784, hidden_sizes=[128, 64], num_classes=10)

# Forward pass
x = torch.randn(32, 784)  # batch 32, 784 fitur
output = model(x)         # shape: (32, 10)
print(output.shape)       # torch.Size([32, 10])
```

## Arsitektur Model

```
Input (input_size)
    │
    ▼
Linear -> ReLU -> Dropout   (Hidden 1)
    │
    ▼
Linear -> ReLU -> Dropout   (Hidden 2)
    │
    ▼
Linear                      (Output -> logits)
    │
    ▼
Output (num_classes)
```

Catatan: Output model adalah **logits** (belum melalui softmax). Gunakan `nn.CrossEntropyLoss` selama pelatihan dan `torch.softmax` untuk mendapatkan probabilitas saat inferensi.

## Lisensi

MIT