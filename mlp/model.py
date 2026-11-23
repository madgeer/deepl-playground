import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) sederhana menggunakan PyTorch.

    Arsitektur:
        Input Layer  -> Hidden Layer 1 (Linear + ReLU + Dropout)
                     -> Hidden Layer 2 (Linear + ReLU + Dropout)
                     -> Output Layer (Linear)

    Parameters
    ----------
    input_size : int
        Jumlah fitur input.
    hidden_sizes : list of int
        Ukuran setiap hidden layer (default: [128, 64]).
    num_classes : int
        Jumlah kelas output (default: 10).
    dropout : float
        Probabilitas dropout (default: 0.2).
    """

    def __init__(self, input_size: int, hidden_sizes: list[int] = None,
                 num_classes: int = 10, dropout: float = 0.2):
        super(MLP, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layers = []
        prev_size = input_size

        # Bangun hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor input dengan shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Output logits dengan shape (batch_size, num_classes).
        """
        return self.network(x)


def create_model(input_size: int = 784,
                 hidden_sizes: list[int] = None,
                 num_classes: int = 10,
                 dropout: float = 0.2) -> MLP:
    """
    Fungsi pembantu untuk membuat instance model MLP.

    Returns
    -------
    MLP
        Instance model MLP.
    """
    return MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout=dropout,
    )
