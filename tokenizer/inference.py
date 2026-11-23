"""
Skrip inferensi tokenizer: encode dan decode teks.

Semua konfigurasi di-hardcode di bawah ini, jadi cukup jalankan:

    python inference.py

Tanpa perlu melewatkan argumen command line.
"""

from tokenizer import Tokenizer, load_tokenizer


# ============================ KONFIGURASI (hardcode) ============================
VOCAB_PATH = "vocab.json"      # Path file vocabulary yang sudah dibuat
TEXTS = [
    "halo apa kabar kamu",
    "saya belajar pytorch",
    "machine learning dan deep learning",
    "contoh kata yang tidak ada di vocab xyzabc",
]
MAX_LENGTH = 16                # Panjang maksimal setelah padding/truncation
ADD_SPECIAL_TOKENS = True      # Tambahkan [CLS] dan [SEP]
PAD = True                     # Pad ke MAX_LENGTH
TRUNCATION = True              # Potong jika melebihi MAX_LENGTH
SKIP_SPECIAL_TOKENS = True     # Lewati special tokens saat decode
# ================================================================================


def main():
    # Muat tokenizer dari file vocabulary
    tokenizer = load_tokenizer(VOCAB_PATH)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Add special tokens: {ADD_SPECIAL_TOKENS}")
    print(f"Pad: {PAD}, Truncation: {TRUNCATION}")
    print()

    for i, text in enumerate(TEXTS):
        # Encode
        ids = tokenizer.encode(
            text,
            max_length=MAX_LENGTH,
            add_special_tokens=ADD_SPECIAL_TOKENS,
            pad=PAD,
            truncation=TRUNCATION,
        )

        # Decode kembali
        decoded = tokenizer.decode(ids, skip_special_tokens=SKIP_SPECIAL_TOKENS)

        print(f"Teks {i + 1}: {text}")
        print(f"  Token IDs: {ids}")
        print(f"  Decoded  : {decoded}")
        print(f"  Panjang  : {len(ids)}")
        print()


if __name__ == "__main__":
    main()
