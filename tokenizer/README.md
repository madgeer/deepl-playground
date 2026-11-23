# Tokenizer (Word-Level) dengan PyTorch

Implementasi tokenizer sederhana level kata untuk NLP menggunakan Python dan PyTorch.

## Struktur File

```
tokenizer/
├── tokenizer.py   # Kelas Tokenizer (vocab, encode, decode, save/load)
├── train.py       # Skrip build vocabulary dari teks
├── inference.py   # Skrip encode/decode teks
└── README.md      # Dokumentasi ini
```

## Instalasi

Pastikan Python dan PyTorch sudah terinstal, disarankan menggunakan conda environment `env-dl`:

```bash
conda activate env-dl
pip install torch
```

Tokenizer ini hanya menggunakan library standar Python (`json`, `re`, `collections`), PyTorch hanya diimpor untuk konsistensi dengan proyek lain.

## Cara Pakai

Semua skrip **tidak menggunakan argumen command line** (`argparse`). Konfigurasi diubah langsung di bagian atas masing-masing file (bagian `KONFIGURASI`), lalu jalankan langsung:

```bash
conda activate env-dl
cd D:\notebook-playground\deepl\tokenizer
python train.py
python inference.py
```

### 1. Build vocabulary (`train.py`)

Edit bagian konfigurasi di atas file `train.py`:

```python
TEXTS = [
    "halo apa kabar kamu",
    "saya belajar pemrograman python dan pytorch",
    # tambahkan teks kamu di sini
]
VOCAB_PATH = "vocab.json"
MIN_FREQ = 1
MAX_VOCAB = None
```

Jalankan:

```bash
python train.py
```

Vocabulary akan disimpan ke `VOCAB_PATH` (default: `vocab.json`).

### 2. Encode / Decode (`inference.py`)

Edit bagian konfigurasi di atas file `inference.py`:

```python
VOCAB_PATH = "vocab.json"
TEXTS = [
    "halo apa kabar kamu",
    "saya belajar pytorch",
]
MAX_LENGTH = 16
ADD_SPECIAL_TOKENS = True
PAD = True
TRUNCATION = True
SKIP_SPECIAL_TOKENS = True
```

Jalankan:

```bash
python inference.py
```

## Contoh Penggunaan sebagai Modul

```python
from tokenizer import Tokenizer, create_tokenizer

# Buat tokenizer baru
tokenizer = create_tokenizer()

# Bangun vocabulary dari teks
texts = ["halo apa kabar", "saya belajar pytorch"]
tokenizer.fit(texts)

# Encode teks -> token IDs
ids = tokenizer.encode("halo apa kabar", max_length=10, pad=True)
print(ids)  # [cls_token_id, id_halo, id_apa, id_kabar, pad_id, ...]

# Decode token IDs -> teks
text = tokenizer.decode(ids, skip_special_tokens=True)
print(text)  # "halo apa kabar"

# Simpan dan muat vocabulary
tokenizer.save("vocab.json")

tokenizer2 = Tokenizer()
tokenizer2.load("vocab.json")
```

## Special Tokens

| Nama | Token | ID |
|------|-------|-----|
| Pad | `[PAD]` | `tokenizer.pad_token_id` |
| Unknown | `[UNK]` | `tokenizer.unk_token_id` |
| Classification | `[CLS]` | `tokenizer.cls_token_id` |
| Separator | `[SEP]` | `tokenizer.sep_token_id` |
| Mask | `[MASK]` | `tokenizer.mask_token_id` |

Kata yang tidak ada dalam vocabulary akan dipetakan ke `[UNK]`.

## Lisensi

MIT