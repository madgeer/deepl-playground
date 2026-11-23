"""
Skrip untuk melatih model n-gram dari korpus teks.

Semua konfigurasi di-hardcode di bawah ini, jadi cukup jalankan:

    python train.py

Tanpa perlu melewatkan argumen command line.
"""

from ngram import Ngram, create_ngram


# ============================ KONFIGURASI (hardcode) ============================
CORPUS = [
    "saya suka makan apel",
    "saya suka minum jeruk",
    "mereka suka makan apel jeruk",
    "saya suka bermain bola",
    "mereka suka bermain bola di taman",
    "saya suka belajar pemrograman",
    "belajar pemrograman sangat menarik",
    "pemrograman python dan pytorch",
    "machine learning dan deep learning",
    "sifat ngram model sangat berguna",
]
NGRAM_SIZE = 3               # Ukuran n-gram (2 = bigram, 3 = trigram)
USE_SMOOTHING = True         # Gunakan Laplace smoothing
SAVE_PATH = "ngram_model.json"  # Path simpan model
# ================================================================================


def main():
    # Buat model n-gram
    print("Membuat model n-gram...")
    model = create_ngram(n=NGRAM_SIZE, use_smoothing=USE_SMOOTHING)

    # Latih model dari corpus
    print(f"Latih dari {len(CORPUS)} kalimat...")
    model.fit(CORPUS)

    print(f"\nModel n-gram (n={model.n}) berhasil dilatih")
    print(f"Vocabulary size: {model.vocab_size}")
    print(f"Total n-grams: {model.total_ngrams}")

    # Simpan model
    model.save(SAVE_PATH)


if __name__ == "__main__":
    main()