"""
Skrip inferensi (prediksi) menggunakan model CNN yang sudah dilatih.

Semua konfigurasi di-hardcode di bawah ini, jadi cukup jalankan:

    python inference.py

Tanpa perlu melewatkan argumen command line.
"""

import torch
from model import CNN, create_model


# ============================ KONFIGURASI (hardcode) ============================
MODEL_PATH = "cnn_model.pth"   # Path file model yang sudah dilatih
IN_CHANNELS = 1                # 1 = grayscale, 3 = RGB
IMG_SIZE = 28                  # Ukuran citra (asumsi persegi, mis. 28x28)
NUM_CLASSES = 10               # Jumlah kelas output
DROPOUT = 0.5                  # Dropout rate (harus sama saat pelatihan)
DEVICE = None                  # None = otomatis (cuda jika tersedia, else cpu)
NUM_SAMPLES = 10               # Jumlah sampel dummy untuk prediksi
# ================================================================================


def load_model(model_path: str,
               in_channels: int,
               num_classes: int,
               dropout: float = 0.5,
               device: str = None) -> CNN:
    """
    Muat model CNN dari file state_dict.

    Returns
    -------
    CNN
        Model yang sudah dimuat bobotnya dan siap untuk inferensi.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = create_model(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model dimuat dari {model_path} pada {device}")
    return model


def predict(model: CNN,
            inputs: torch.Tensor,
            device: torch.device) -> tuple:
    """
    Lakukan prediksi pada tensor input citra.

    Parameters
    ----------
    model : CNN
        Model yang telah dilatih.
    inputs : torch.Tensor
        Tensor input citra dengan shape (batch_size, in_channels, H, W).
    device : torch.device
        Perangkat komputasi.

    Returns
    -------
    tuple
        (predicted_classes, probabilities)
    """
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    return predicted_classes.cpu(), probabilities.cpu()


def main():
    # Tentukan perangkat
    if DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)

    # Muat model
    model = load_model(
        model_path=MODEL_PATH,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        device=device,
    )

    # Buat data dummy citra untuk demonstrasi prediksi
    dummy_inputs = torch.randn(NUM_SAMPLES, IN_CHANNELS, IMG_SIZE, IMG_SIZE)

    # Prediksi
    predicted_classes, probabilities = predict(model, dummy_inputs, device)

    print("\nHasil Prediksi:")
    print(f"{'Sampel':<8} {'Kelas Prediksi':<15} {'Confidence':<12}")
    print("-" * 38)
    for i in range(NUM_SAMPLES):
        conf = probabilities[i, predicted_classes[i]].item() * 100
        print(f"{i + 1:<8} {predicted_classes[i].item():<15} {conf:.2f}%")


if __name__ == "__main__":
    main()
