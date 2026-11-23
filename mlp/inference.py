"""
Skrip inferensi (prediksi) menggunakan model MLP yang sudah dilatih.

Semua konfigurasi di-hardcode di bawah ini, jadi cukup jalankan:

    python inference.py

Tanpa perlu melewatkan argumen command line.
"""

import torch
from model import MLP, create_model


# ============================ KONFIGURASI (hardcode) ============================
MODEL_PATH = "mlp_model.pth"   # Path file model yang sudah dilatih
INPUT_SIZE = 784               # Jumlah fitur input
NUM_CLASSES = 10               # Jumlah kelas output
HIDDEN_SIZES = [128, 64]       # Ukuran hidden layer
DEVICE = None                  # None = otomatis (cuda jika tersedia, else cpu)
NUM_SAMPLES = 10               # Jumlah sampel dummy untuk prediksi
# ================================================================================


def load_model(model_path: str,
               input_size: int,
               num_classes: int,
               hidden_sizes: list[int] = None,
               dropout: float = 0.2,
               device: str = None) -> MLP:
    """
    Muat model MLP dari file state_dict.

    Returns
    -------
    MLP
        Model yang sudah dimuat bobotnya dan siap untuk inferensi.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = create_model(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model dimuat dari {model_path} pada {device}")
    return model


def predict(model: MLP,
            inputs: torch.Tensor,
            device: torch.device) -> tuple:
    """
    Lakukan prediksi pada tensor input.

    Parameters
    ----------
    model : MLP
        Model yang telah dilatih.
    inputs : torch.Tensor
        Tensor input dengan shape (batch_size, input_size).
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
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
        hidden_sizes=HIDDEN_SIZES,
        device=device,
    )

    # Buat data dummy untuk demonstrasi prediksi
    dummy_inputs = torch.randn(NUM_SAMPLES, INPUT_SIZE)

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
