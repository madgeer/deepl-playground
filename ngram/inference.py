"""
Skrip inferensi n-gram: prediksi kata selanjutnya dan evaluasi kalimat.

Semua konfigurasi di-hardcode di bawah ini, jadi cukup jalankan:

    python inference.py

Tanpa perlu melewatkan argumen command line.
"""

import json
import os
from ngram import Ngram, create_ngram


# ============================ KONFIGURASI (hardcode) ============================
VOCAB_PATH = "ngram_model.json"   # Path file model n-gram
CONTEXT = ("saya", "suka")          # Konteks n-1 kata untuk prediksi
TEST_SENTENCES = [
    "saya suka makan apel",
    "mereka suka bermain bola",
    "saya belajar pemrograman",
]
# ================================================================================


def main():
    # Cek apakah file model ada, kalau belum latih dulu
    model = None
    if os.path.exists(VOCAB_PATH):
        try:
            model = Ngram.load(VOCAB_PATH)
            print(f"Model dimuat dari {VOCAB_PATH}")
        except Exception as e:
            print(f"Gagal muat model: {e}")
            model = None

    if model is None:
        print("Melatih model terlebih dahulu...")
        corpus = [
            "saya suka makan apel",
            "saya suka minum jeruk",
            "mereka suka makan apel jeruk",
            "saya suka bermain bola",
            "mereka suka bermain bola di taman",
        ]
        model = create_ngram(n=3, use_smoothing=True)
        model.fit(corpus)
        model.save(VOCAB_PATH)

    # 1. Prediksi kata selanjutnya
    print(f"\n=== Prediksi kata selanjutnya ===")
    print(f"Konteks: {CONTEXT}")
    next_word = model.next_word(CONTEXT)
    print(f"Kata terkemungkinan selanjutnya: '{next_word}'")

    # Tampilkan beberapa kandidat terbaik
    scores = model.next_word(CONTEXT, return_scores=True)
    print("Top kandidat:")
    for word, prob in scores[:5]:
        print(f"  {word}: {prob:.4f}")

    # 2. Hitung log-probabilitas kalimat
    print(f"\n=== Evaluasi kalimat ===")
    for sent in TEST_SENTENCES:
        lp = model.sentence_log_prob(sent)
        print(f"'{sent}' -> log-probabilitas: {lp:.4f}")

    # 3. Hitung perplexity
    print(f"\n=== Perplexity ===")
    pp = model.perplexity(TEST_SENTENCES)
    print(f"Perplexity dari {len(TEST_SENTENCES)} kalimat: {pp:.4f}")


if __name__ == "__main__":
    main()