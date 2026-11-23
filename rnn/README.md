# RNN dengan Word Embedding untuk Text Generation

Implementasi RNN (Recurrent Neural Network) yang menggunakan **Word Embedding** untuk tugas **generate teks (text generation)**.

## Struktur File

```
rnn/
├── model.py       # Kelas RNNWithEmbedding (embedding + RNN + output)
├── train.py       # Skrip pelatihan (latih model untuk memprediksi kata selanjutnya)
├── inference.py   # Skrip text generation (generate teks otomatis)
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
cd D:\notebook-playground\deepl\rnn
python train.py
python inference.py
```

### 1. Pelatihan model (`train.py`)

File ini melatih model RNN untuk memprediksi kata selanjutnya berdasarkan konteks kata-kata sebelumnya.

Edit bagian konfigurasi di atas file `train.py`:

```python
VOCABULARY = [
    "the", "and", "to", "of", "a", "in", "is", "you", "that", "it",
    # ... kata lain (lihat file train.py untuk lengkapnya)
]
VOCAB_SIZE = len(VOCABULARY)  # akan dihitung otomatis
EMBEDDING_DIM = 64       # Dimensi embedding
HIDDEN_SIZE = 128        # Ukuran hidden state
NUM_LAYERS = 2           # Jumlah layer RNN
DROPOUT = 0.3            # Dropout rate
BIDIRECTIONAL = False    # Apakah bidirectional
SEQ_LENGTH = 10          # Panjang sekuritas input
BATCH_SIZE = 16          # Ukuran batch
EPOCHS = 20              # Jumlah epoch
LR = 0.001               # Learning rate
DEVICE = None            # None = otomatis (cuda jika tersedia, else cpu)
SAVE_PATH = "rnn_embedding_model.pth"
```

Catatan: `VOCABULARY` adalah daftar kata yang digunakan untuk demo. Untuk proyek nyata, gantikan dengan korpus teks yang lebih besar (misal: Shakespeare, blog posts, dll.).

Jalankan:

```bash
python train.py
```

Model akan dilatih untuk memprediksi kata selanjutnya, lalu disimpan ke `SAVE_PATH`.

### 2. Text Generation (`inference.py`)

Edit bagian konfigurasi di atas file `inference.py`:

```python
MODEL_PATH = "rnn_embedding_model.pth"
VOCABULARY = [daftar kata sama persis seperti di train.py]
EMBEDDING_DIM = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = False
SEQ_LENGTH = 10
DEVICE = None
NUM_WORDS_TO_GENERATE = 15    # berapa kata baru yang digenerate
TEMPERATURE = 0.8           # 0.5 = determinis, 1.0 = normal, >1 = lebih acak
START_CONTEXT = [1, 2, 3]   # indeks kata awal (sekuritas awal)
```

Jalankan:

```bash
python inference.py
```

Output akan menampilkan teks yang digenerate mulai dari konteks yang diberikan.

### 3. Menjalankan dari Modul (Python code)

```python
from model import create_model, generate_text, count_parameters

# Buat model
model = create_model(
    vocab_size=10000,       # ukuran vocab korpus Anda
    embedding_dim=256,
    hidden_size=512,
    num_layers=2,
    dropout=0.3,
)

# Latih model (lihat train.py untuk skrip lengkap)
# ... model.fit(...)

# Generate teks secara rekayasa
start_context = [1, 2, 3]  # indeks kata awal, misal: "the", "and", "to"
generated = generate_text(
    model=model,
    start_context=start_context,
    num_words=20,
    temperature=0.8,
)

# Tampilkan hasil
word_to_idx = {w: i+1 for i, w in enumerate(vocab_list)}
idx_to_word = {i+1: w for i, w in enumerate(vocab_list)}
print("Generated text: ", " ".join([idx_to_word[i] for i in generated]))
```

## Arsitektur Model

```text
Input (indeks kata)                Embedding Layer
   │                                   │
   ▼                                   ▼
[batch, seq_len]  ───────►  (Embedding)  →  [batch, seq_len, embed_dim]
                                     │
                                     ▼
                              [batch, seq_len, embed_dim]
                                     │
                                     ▼
                              RNN Layer (batch_first=True)
                                     │
                                     ▼
                              [batch, seq_len, hidden_size × dirs]
                                     │
                                     ▼
                              Linear Output Layer
                                     │
                                     ▼
              [batch, seq_len, vocab_size] ← Logits prediksi kata selanjutnya
```

**Catatan penting:**
- Indeks kata **0** adalah `padding_idx` dan **tidak akan dilatih** (diabaikan oleh embedding).
- Output dari model adalah **logits** (belum softmax). Gunakan `torch.softmax` untuk mendapatkan probabilitas.
- Temperature kontrol "acakness": nilai lebih kecil = lebih deterministik/predictif, nilai lebih besar = lebih acak/kreatif.

### 4. Konfigurasi Lanjutan

Pada file `model.py` ada fungsi tambahan:

- `create_shakespeare_rnn()` — Settingan khusus untuk korpus besar
- `count_parameters()` — Hitung parameter yang bisa dilatih

```python
from model import create_shakespeare_rnn

# Untuk korpus teks yang lebih kompleks
model = create_shakespeare_rnn(
    embedding_dim=256,
    hidden_size=512,
    num_layers=2,
    dropout=0.3,
)
```

## Lisensi

MIT