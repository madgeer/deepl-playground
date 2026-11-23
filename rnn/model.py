import torch
import torch.nn as nn


class RNNWithEmbedding(nn.Module):
    """
    RNN dengan Word Embedding untuk tugas text generation.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_classes: int = None,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super(RNNWithEmbedding, self).__init__()

        if num_classes is None:
            num_classes = vocab_size

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> tuple:
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        batch_size, seq_len, _ = output.shape
        output = output.reshape(batch_size * seq_len, -1)
        output = self.fc(output)
        output = output.reshape(batch_size, seq_len, self.num_classes)
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )


def create_model(
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_size: int = 256,
    num_layers: int = 1,
    num_classes: int = None,
    dropout: float = 0.0,
    bidirectional: bool = False,
) -> RNNWithEmbedding:
    if num_classes is None:
        num_classes = vocab_size
    return RNNWithEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    )


def count_parameters(model: RNNWithEmbedding) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_text(model: RNNWithEmbedding,
                  start_context: list,
                  num_words: int = 10,
                  temperature: float = 1.0,
                  device: torch.device = None) -> list:
    """
    Generate teks secara rekayasa (autoregressive) dengan RNN.

    Parameters
    ----------
    model : RNNWithEmbedding
        Model yang sudah dilatih.
    start_context : list
        List indeks kata awal (sekuritas awal). Panjang disesuaikan maksimal 10.
    num_words : int
        Jumlah kata tambahan yang akan digenerate.
    temperature : float
        Suhu untuk kontrol acakness (>1 = lebih acak, <1 = lebih deterministik).
    device : torch.device, optional
        Device komputasi. Jika None, otomatis dari model.

    Returns
    -------
    list
        List lengkap indeks kata setelah digenerate (termasuk konteks awal).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    max_len = 10
    context = torch.tensor(start_context[:max_len], dtype=torch.long).unsqueeze(0)
    generated_indices = list(start_context)

    with torch.no_grad():
        for _ in range(num_words):
            if context.dim() == 1:
                context = context.unsqueeze(0)

            output, _ = model(context)
            last_output = output[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(last_output / temperature, dim=1)
                idx_dist = torch.multinomial(probs, num_samples=1)
                predicted_idx = idx_dist.item()
            else:
                _, predicted_idx = torch.max(last_output, dim=1)
                predicted_idx = predicted_idx.item()

            generated_indices.append(predicted_idx)

            if context.shape[1] < max_len:
                context = torch.cat([context, torch.tensor([[predicted_idx]], dtype=torch.long)], dim=1)
            else:
                context = torch.cat([context[:, 1:], torch.tensor([[predicted_idx]], dtype=torch.long)], dim=1)

    return generated_indices


def create_shakespeare_rnn(
    embedding_dim: int = 256, hidden_size: int = 512, num_layers: int = 2,
    dropout: float = 0.3,
) -> RNNWithEmbedding:
    return RNNWithEmbedding(
        vocab_size=10000,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )