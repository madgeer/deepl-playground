import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) sederhana untuk klasifikasi citra
    menggunakan PyTorch.

    Arsitektur (default untuk citra grayscale 1x28x28, mis. MNIST):
        Input (1, H, W)
            │
            ▼
        Conv2d(1 -> 32) -> ReLU -> MaxPool2d   (32, H/2, W/2)
            │
            ▼
        Conv2d(32 -> 64) -> ReLU -> MaxPool2d  (64, H/4, W/4)
            │
            ▼
        Flatten
            │
            ▼
        Linear -> ReLU -> Dropout
            │
            ▼
        Linear -> Output (logits)

    Parameters
    ----------
    in_channels : int
        Jumlah channel input (1 untuk grayscale, 3 untuk RGB).
    num_classes : int
        Jumlah kelas output (default: 10).
    dropout : float
        Probabilitas dropout (default: 0.5).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 dropout: float = 0.5):
        super(CNN, self).__init__()

        # Fitur ekstraktor (konvolusi)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # /2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # /4
        )

        # Klasifier (fully connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),   # asumsi input 28x28 -> setelah 2x MaxPool jadi 7x7
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor input dengan shape (batch_size, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output logits dengan shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(in_channels: int = 1,
                 num_classes: int = 10,
                 dropout: float = 0.5) -> CNN:
    """
    Fungsi pembantu untuk membuat instance model CNN.

    Returns
    -------
    CNN
        Instance model CNN.
    """
    return CNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )
