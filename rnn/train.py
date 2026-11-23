"""
Skrip pelatihan RNN dengan Word Embedding untuk text generation.

Semua konfigurasi di-hardcode di bawah ini, jadi cukup jalankan:

    python train.py

Tanpa perlu melewatkan argumen command line.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from model import RNNWithEmbedding, create_model, count_parameters


# ============================ KONFIGURASI (hardcode) ============================
VOCABULARY = [
    # Kata-kata sederhana untuk demo; dalam praktiknya gunakan korpus besar
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
VOCAB_SIZE = len(VOCABULARY) + 1  # +1 untuk padding index 0 (diasumsikan embedding punya padding_idx=0)
EMBEDDING_DIM = 64       # Dimensi embedding
HIDDEN_SIZE = 128        # Ukuran hidden state
NUM_LAYERS = 2           # Jumlah layer RNN
DROPOUT = 0.3            # Dropout rate
BIDIRECTIONAL = False    # Apakah bidirectional
SEQ_LENGTH = 10          # Panjang sekuritas input
BATCH_SIZE = 16          # Ukuran batch
EPOCHS = 20              # Jumlah epoch pelatihan
LR = 0.001               # Learning rate
DEVICE = "cpu"         # Paksa CPU untuk menghindari CU kernel error di environment ini
SAVE_PATH = "rnn_embedding_model.pth"  # Path simpan model terbaik
# ================================================================================


class TextDataset(Dataset):
    """
    Dataset untuk teks sekuerensial kata-level.
    Mengubah kalimat menjadi pasangan (input_seq, target_word).
    """

    def __init__(self, vocabulary: list, seq_length: int,
                 max_samples: int = 1000):
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.seq_length = seq_length
        self.max_samples = max_samples

        # Bangun mapping kata -> indeks dan indeks -> kata
        self.word_to_idx = {word: i + 1 for i, word in enumerate(vocabulary)}
        # Indeks 0 adalah padding (tidak digunakan)
        self.idx_to_word = {i + 1: word for i, word in enumerate(vocabulary)}

        # Buat data: setiap sampel adalah sekuritas kata -> kata target berikutnya
        self.samples = self._generate_samples(max_samples)

    def _generate_samples(self, max_samples):
        """Generate pasangan (input sekuritas, kata target) dari vocabulary."""
        samples = []
        # Buat kalimat acak menggabungkan kata dari vocabulary
        for _ in range(max_samples):
            # Buat sekuritas acak
            seq_len = random.randint(1, self.seq_length)
            # Pastikan sekuritas memiliki setidaknya 1 kata
            if seq_len < 1:
                seq_len = 1

            # Pilih kata acak untuk kata targetnya
            target_word = random.choice(self.vocabulary)

            # Buat sekuritas: pilih seq_length kata acak (dengan duplikasi)
            # Untuk simulasi sederhana, ambil kata-kata acak dari vocab
            context_words = random.choices(
                self.vocabulary, k=seq_len
            )

            # Konteks: indeks kata (dengan offset 1 karena 0 = padding)
            context_idxs = [self.word_to_idx[w] for w in context_words]
            # Pastikan panjang tetap seq_length (padding dengan 0 jika kurang)
            while len(context_idxs) < self.seq_length:
                context_idxs.append(0)  # padding

            # Potong jika terlalu panjang
            context_idxs = context_idxs[:self.seq_length]

            samples.append((context_idxs, self.word_to_idx[target_word]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        # context: tensor shape (seq_length,)
        # target: integer indeks kata
        return torch.tensor(context, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )

    def get_word(self, idx):
        """Mapatkan indeks kata kembali menjadi string."""
        if idx == 0:
            return "<PAD>"
        if idx in self.idx_to_word:
            return self.idx_to_word[idx]
        return "<UNK>"


def train(model: RNNWithEmbedding,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int = 20,
          lr: float = 0.001,
          device: str = None,
          save_path: str = "rnn_embedding_model.pth") -> dict:
    """
    Latih model RNN dengan embedding.

    Parameters
    ----------
    model : RNNWithEmbedding
        Model yang akan dilatih.
    train_loader : DataLoader
        DataLoader data pelatihan.
    val_loader : DataLoader
        DataLoader data validasi.
    epochs : int
        Jumlah epoch.
    lr : float
        Learning rate.
    device : str, optional
        Perangkat ("cuda" atau "cpu").
    save_path : str
        Path simpan model terbaik.

    Returns
    -------
    dict
        Riwayat pelatihan.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Menggunakan perangkat: {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Input shape: (batch_size, seq_length) - indeks kata
            # Model output: (batch_size, seq_length, vocab_size)
            outputs, _ = model(inputs)

            # Untuk loss: kita ambil output dari waktu langkah terakhir
            # output_last shape: (batch_size, vocab_size)
            output_last = outputs[:, -1, :]
            loss = criterion(output_last, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Akurasi: prediksi kata dengan probabilitas tertinggi
            _, predicted = torch.max(output_last, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train if total_train > 0 else 0.0

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                output_last = outputs[:, -1, :]
                loss = criterion(output_last, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(output_last, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        # Simpan model terbaik
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model terbaik disimpan ke {save_path} (acc: {val_acc * 100:.2f}%)")

    print(f"\nPelatihan selesai. Best Val Acc: {best_val_acc * 100:.2f}%")
    return history


def create_dummy_data(vocabulary: list, seq_length: int,
                      batch_size: int = 16, max_samples: int = 1000) -> tuple:
    """
    Membuat dataset dummy untuk demonstrasi (ganti dengan korpus teks nyata).

    Returns
    -------
    tuple (train_loader, val_loader)
    """
    dataset = TextDataset(vocabulary, seq_length, max_samples=max_samples)

    # Split train/val
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def generate_text(model: RNNWithEmbedding,
                  start_context: list,
                  num_words: int = 10,
                  temperature: float = 1.0,
                  device: torch.device = None) -> list:
    """
    Generate teks secara rekayasa (autoregressive).

    Parameters
    ----------
    model : RNNWithEmbedding
        Model yang sudah dilatih.
    start_context : list
        List indeks kata awal (sekuritas awal).
    num_words : int
        Jumlah kata tambahan yang akan digenerate.
    temperature : float
        Suhu untuk kontrol acakness (>1 = lebih acak, <1 = lebih deterministik).
    device : torch.device
        Device komputasi.

    Returns
    -------
    list
        List lengkap indeks kata setelah digenerate.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Konteks awal menjadi tensor
    context = torch.tensor(start_context, dtype=torch.long).unsqueeze(0)  # (1, seq_len)

    generated_indices = list(start_context)  # simpan kata yang sudah ada

    with torch.no_grad():
        for _ in range(num_words):
            # Prediksi dari konteks saat ini
            output, hidden = model(context)  # output: (1, seq_len, vocab_size)

            # Ambil output dari waktu langkah terakhir
            last_output = output[:, -1, :]  # (1, vocab_size)

            # Terapkan temperature
            if temperature > 0:
                scaled_logits = last_output / temperature
                # Softmax untuk mendapatkan probabilitas
                probs = torch.softmax(scaled_logits, dim=1)

                # Sample kata berdasarkan probabilitas (random)
                idx_dist = torch.multinomial(probs, num_samples=1)
                predicted_idx = idx_dist.item()
            else:
                # Greedy search: ambil kata dengan probabilitas tertinggi
                _, predicted_idx = torch.max(probs, dim=1) if temperature > 0 else torch.max(
                    last_output, dim=1
                )
                predicted_idx = predicted_idx.item()

            # Tambahkan ke generated indices
            generated_indices.append(predicted_idx)

            # Perbarui konteks: geser kiri dan tambah kata baru di ujung
            # context shape (1, seq_length), ambil [0,1,...,seq_len-2] dan tambah predicted_idx
            if len(generated_indices) > SEQ_LENGTH:
                # Geser context: hilangkan kata pertama, tambah kata baru di akhir
                new_context = context[:, 1:].squeeze(0)  # (seq_len-1)
                context = torch.cat([
                    new_context,
                    torch.tensor([predicted_idx], dtype=torch.long).unsqueeze(0)
                ], dim=0).unsqueeze(0)  # kembali jadi (1, seq_len)
            else:
                # Masih dalam panjang awal, hanya perlu menambah
                context = torch.cat([
                    context.squeeze(0),
                    torch.tensor([predicted_idx], dtype=torch.long).unsqueeze(0)
                ], dim=0).unsqueeze(0)

    return generated_indices


def main():
    # Tentukan perangkat
    if DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)

    # Buat model
    model = create_model(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    )

    print("Model RNN dengan Embedding:")
    print(model)
    print(f"Total parameter: {count_parameters(model):,}")

    # Buat dataset
    train_loader, val_loader = create_dummy_data(
        VOCABULARY, SEQ_LENGTH, BATCH_SIZE, max_samples=500
    )

    # Latih model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        save_path=SAVE_PATH,
    )


if __name__ == "__main__":
    main()