"""
Skrip pelatihan CNN menggunakan PyTorch.

Semua konfigurasi di-hardcode di bawah ini, jadi cukup jalankan:

    python train.py

Tanpa perlu melewatkan argumen command line.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import CNN, create_model


# ============================ KONFIGURASI (hardcode) ============================
EPOCHS = 10                # Jumlah epoch pelatihan
BATCH_SIZE = 64            # Ukuran batch
LR = 0.001                 # Learning rate
IN_CHANNELS = 1            # 1 = grayscale, 3 = RGB
IMG_SIZE = 28              # Ukuran citra (asumsi persegi, mis. 28x28)
NUM_CLASSES = 10           # Jumlah kelas output
DROPOUT = 0.5              # Dropout rate
DEVICE = None              # None = otomatis (cuda jika tersedia, else cpu)
SAVE_PATH = "cnn_model.pth"  # Path simpan model terbaik
NUM_SAMPLES = 2000         # Jumlah sampel dummy data
VAL_SPLIT = 0.2            # Proporsi data validasi
# ================================================================================


def train(model: CNN,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int = 10,
          lr: float = 0.001,
          device: str = None,
          save_path: str = "cnn_model.pth") -> dict:
    """
    Latih model CNN.

    Parameters
    ----------
    model : CNN
        Model yang akan dilatih.
    train_loader : DataLoader
        DataLoader untuk data pelatihan.
    val_loader : DataLoader
        DataLoader untuk data validasi.
    epochs : int
        Jumlah epoch pelatihan.
    lr : float
        Learning rate.
    device : str, optional
        Perangkat komputasi ("cuda" atau "cpu"). Jika None, otomatis dipilih.
    save_path : str
        Path untuk menyimpan model terbaik.

    Returns
    -------
    dict
        Riwayat pelatihan (train_loss, val_loss, val_acc per epoch).
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
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        # Simpan model terbaik
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model terbaik disimpan ke {save_path} (acc: {val_acc * 100:.2f}%)")

    print(f"\nPelatihan selesai. Best Val Acc: {best_val_acc * 100:.2f}%")
    return history


def create_dummy_data(num_samples: int = 1000,
                      in_channels: int = 1,
                      img_size: int = 28,
                      num_classes: int = 10,
                      batch_size: int = 64,
                      val_split: float = 0.2) -> tuple:
    """
    Membuat data dummy acak untuk demonstrasi (ganti dengan data asli Anda).

    Returns
    -------
    tuple
        (train_loader, val_loader)
    """
    X = torch.randn(num_samples, in_channels, img_size, img_size)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def main():
    # Tentukan perangkat
    if DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)

    # Buat model
    model = create_model(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )

    print("Model CNN:")
    print(model)
    print(f"Total parameter: {sum(p.numel() for p in model.parameters()):,}")

    # Buat data dummy (ganti dengan data asli Anda)
    train_loader, val_loader = create_dummy_data(
        num_samples=NUM_SAMPLES,
        in_channels=IN_CHANNELS,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
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
