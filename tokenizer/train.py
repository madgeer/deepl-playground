"""
Skrip untuk membangun vocabulary tokenizer dari teks.

Semua konfigurasi di-hardcode di bawah ini, jadi cukup jalankan:

    python train.py

Tanpa perlu melewatkan argumen command line.
"""

import torch
from tokenizer import Tokenizer, create_tokenizer


# ============================ KONFIGURASI (hardcode) ============================
TEXTS = [
    "halo apa kabar kamu",
    "saya belajar pemrograman python dan pytorch",
    "machine learning sangat menarik untuk dipelajari",
    "tokenizer mengubah teks menjadi angka token id",
    "deep learning menggunakan jaringan syaraf tiruan dalam",
    "contoh kalimat lain untuk membangun vocabulary",
    "selamat belajar dan semoga sukses",
]
VOCAB_PATH = "vocab.json"      # Path simpan vocabulary
MIN_FREQ = 1                   # Frekuensi minimum token
MAX_VOCAB = None               # Ukuran maksimal vocab (None = tanpa batas)
# ================================================================================


def main():
    # Buat tokenizer
    tokenizer = create_tokenizer()

    # Bangun vocabulary dari teks
    print("Membangun vocabulary dari", len(TEXTS), "teks...")
    tokenizer.fit(TEXTS, min_freq=MIN_FREQ, max_vocab=MAX_VOCAB)

    print(f"Ukuran vocabulary: {tokenizer.vocab_size}")
    print("Special tokens:")
    for name, token in tokenizer.special_tokens.items():
        print(f"  {name}: {token} -> id {getattr(tokenizer, name + '_token_id')}")

    # Simpan vocabulary
    tokenizer.save(VOCAB_PATH)

    # Tampilkan beberapa item vocab
    print("\nContoh item vocabulary (10 pertama):")
    for i, (token, idx) in enumerate(list(tokenizer.vocab.items())[:10]):
        print(f"  {token}: {idx}")


if __name__ == "__main__":
    main()
