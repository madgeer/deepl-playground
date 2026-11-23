# N-gram Language Model dengan PyTorch

Implementasi model language n-gram sederhana menggunakan Python (tanpa PyTorch dependency di dalam modul sendiri, hanya menggunakan standard library dan `math`).

## Struktur File

```
ngram/
├── ngram.py   # Kelas Ngram (latih, prediksi, evaluasi)
├── train.py   # Skrip build model dari korpus teks
├── inference.py  # Skrip prediksi & evaluasi
└── README.md  # Dokumentasi ini
```

## Instalasi

Pastikan Python sudah terinstal:

```bash
conda activate env-dl
# Tidak butuh paket eksternal khusus, hanya standard library Python
```

Model ini menggunakan hanya `collections`, `json`, dan `math` dari Python standard library. Tidak ada dependensi eksternal tambahan.

## Cara Pakai

Semua skrip **tidak menggunakan argumen command line** (`argparse`). Konfigurasi diubah langsung di bagian atas masing-masing file (bagian `KONFIGURASI`), lalu jalankan langsung:

```bash
conda activate env-dl
cd D:\notebook-playground\deepl\ngram
python train.py
python inference.py
```

### 1. Build model (`train.py`)

Edit bagian konfigurasi di atas file `train.py`:

```python
CORPUS = [
    "saya suka makan apel",
    "saya suka minum jeruk",
    # tambahkan kalimat kamu di sini
]
NGRAM_SIZE = 3               # 2 = bigram, 3 = trigram
USE_SMOOTHING = True         # Gunakan Laplace smoothing
SAVE_PATH = "ngram_model.json"
```

Jalankan:

```bash
python train.py
```

Model akan disimpan ke `SAVE_PATH` (default: `ngram_model.json`). Model mencakup vocabulary, n-gram counts, dan parameter smoothing.

### 2. Prediksi & Evaluasi (`inference.py`)

Edit bagian konfigurasi di atas file `inference.py`:

```python
VOCAB_PATH = "ngram_model.json"
CONTEXT = ("saya", "suka")          # Konteks 2 kata (untuk trigram)
TEST_SENTENCES = [
    "saya suka makan apel",
    "mereka suka bermain bola",
    "saya belajar pemrograman",
]
```

Jalankan:

```bash
python inference.py
```

Output yang akan tampil:

- **Prediksi kata selanjutnya** berdasarkan konteks yang diberikan
- **Top kandidat** beserta probabilitas-nya
- **Log-probabilitas** setiap kalimat uji
- **Perplexity** sebagai metrik evaluasi seberapa "terkejut" model dengan data baru

## Contoh Penggunaan sebagai Modul

```python
from ngram import Ngram, create_ngram, load_ngram

# Buat model trigram dengan smoothing
model = create_ngram(n=3, use_smoothing=True)

# Latih dari korpus
corpus = [
    "saya suka makan apel",
    "saya suka minum jeruk",
]
model.fit(corpus)

# Prediksi kata selanjutnya
pred = model.next_word(("saya", "suka"))
print(pred)  # misal: "makan"

# Atau daftar kandidat terurut
candidates = model.next_word(("saya", "suka"), return_scores=True)
for word, prob in candidates[:3]:
    print(f"{word}: {prob:.4f}")

# Hitung log-probabilitas kalimat
lp = model.sentence_log_prob("saya suka makan apel")
print(lp)

# Simpan dan muat
model.save("my_model.json")
model2 = load_ngram("my_model.json")
```

## Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Unigram (n=1)** | Model kata saja, probabilitas berdasarkan frekuensi |
| **Bigram (n=2)** | Memprediksi kata berdasarkan 1 kata sebelumnya |
| **Trigram (n=3)** | Memprediksi kata berdasarkan 2 kata sebelumnya (default) |
| **Laplace Smoothing** | Menghindari probability nol untuk kata yang belum pernah muncul |
| **Perplexity** | Metrik evaluasi kualitas model |
| **Save/Load** | Simpan dan muat model ke/from file JSON |

## Contoh Output

```
=== Prediksi kata selanjutnya ===
Konteks: ('saya', 'suka')
Kata terkemungkinan selanjutnya: 'makan'

Top kandidat:
  makan: 0.4521
  minum: 0.3115
  bermain: 0.1324
  belajar: 0.1040

=== Evaluasi kalimat ===
'saya suka makanapel' -> log-probabilitas: -3.2145
'mereka suka bermainbola' -> log-probabilitas: -2.8731

=== Perplexity ===
Perplexity dari 3 kalimat: 3.4521
```

## Lisensi

MIT