"""
Skrip inferensi text generation menggunakan RNN dengan Word Embedding.

Semua konfigurasi di-hardcode di bawah ini, jadi cukup jalankan:

    python inference.py

Tanpa perlu melewatkan argumen command line.
"""

import torch
import numpy as np
from model import RNNWithEmbedding, create_model, generate_text


# ============================ KONFIGURASI (hardcode) ============================
MODEL_PATH = "rnn_embedding_model.pth"   # Path file model yang sudah dilatih
VOCABULARY = [
    "the", "and", "to", "of", "a", "in", "is", "you", "that", "it",
    "he", "was", "for", "on", "are", "as", "with", "his", "they",
    "I", "at", "be", "this", "have", "from", "or", "one", "had", "by",
    "word", "but", "not", "what", "all", "were", "we", "when", "your",
    "can", "said", "there", "use", "an", "each", "which", "she", "do",
    "how", "out", "many", "then", "them", "these", "so", "some", "her",
    "would", "make", "like", "him", "into", "time", "two", "more", "going",
    "come", "see", "number", "no", "way", "could", "people", "my", "over",
    "know", "water", "than", "call", "first", "who", "may", "down", "side",
    "been", "now", "find", "any", "new", "good", "sentence", "great", "those",
]
VOCAB_SIZE = len(VOCABULARY) + 1  # +1 untuk padding index 0 (sesuai training)
EMBEDDING_DIM = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = False
SEQ_LENGTH = 10
DEVICE = "cpu"         # Paksa CPU
NUM_WORDS_TO_GENERATE = 15     # Berapa kata tambahan digenerate
TEMPERATURE = 0.8              # Kontrol acakness (0.5 = determinis, 1.0 = normal, 1.5 = acak)
START_CONTEXT = [1, 2, 3]      # Indeks kata awal (contoh: "the and to")
# ================================================================================


def load_model(model_path: str,
               vocab_size: int,
               embedding_dim: int,
               hidden_size: int,
               num_layers: int,
               num_classes: int = None,
               dropout: float = 0.3,
               bidirectional: bool = False,
               device: str = None) -> RNNWithEmbedding:
    """
    Muat model RNN dengan embedding dari file state_dict.

    Returns
    -------
    RNNWithEmbedding
        Model yang sudah dimuat bobotnya dan siap untuk text generation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = create_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model dimuat dari {model_path} pada {device}")
    return model


def idxs_to_words(idxs: list, idx_to_word: dict) -> list:
    """
    Konversi daftar indeks kata menjadi list string.

    Parameters
    ----------
    idxs : list
        List indeks kata.
    idx_to_word : dict
        Peta indeks -> kata.

    Returns
    -------
    list
        List nama kata.
    """
    words = []
    for idx in idxs:
        word = idx_to_word.get(idx, "<UNK>")
        if word == "<PAD>":
            continue  # lewati padding
        words.append(word)
    return words


def main():
    # Tentukan perangkat
    if DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)

    # Muat model
    model = load_model(
        model_path=MODEL_PATH,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        device=device,
    )

    # Mapping indeks -> kata (reverse mapping)
    idx_to_word = {i + 1: w for i, w in enumerate(VOCABULARY)}

    # Generate teks
    print(f"\n=== Text Generation ===")
    print(f"Konteks awal indeks: {START_CONTEXT}")
    print(f"akan di-generate {NUM_WORDS_TO_GENERATE} kata baru")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Kata dalam vocab: {len(VOCABULARY)}")
    print("-" * 50)

    # Konteks awal harus sesuai panjang SEQ_LENGTH (dipad dengan 0 jika kurang)
    # START_CONTOH: [1, 2, 3] berarti kita mulai dengan "the and to"
    # Jika START_CONTEXT lebih pendek dari SEQ_LENGTH, padding akan dilakukan otomatis di dalam fungsi generate_text

    generated_indices = generate_text(
        model=model,
        start_context=START_CONTEXT,
        num_words=NUM_WORDS_TO_GENERATE,
        temperature=TEMPERATURE,
        device=device,
    )

    # Konversi indeks menjadi kata
    generated_words = idxs_to_words(generated_indices, idx_to_word)

    # Tampilkan hasil
    print("\nHasil Generate Teks:")
    print(" ".join(generated_words[:NUM_WORDS_TO_GENERATE + len(START_CONTEXT)]))

    # Tampilkan detail
    print(f"\nDetail prediksi:")
    for i, (idx, word) in enumerate(zip(generated_indices, generated_indices)):
        actual_word = idxs_to_words([idx], idx_to_word)[0]
        conf = "N/A (tidak ada confidence langsung)"
        print(f"  Langkah {i + 1:2d}: indeks={idx:3d} -> kata='{actual_word}'")


if __name__ == "__main__":
    main()