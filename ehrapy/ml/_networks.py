from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        sizes = [n_features, *[hidden_size] * num_layers]
        self.layers = nn.Sequential(
            *(
                module
                for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True)
                for module in (nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(dropout))
            )
        )

    def forward(self, values: Tensor, mask: Tensor, time_since_observed: Tensor, static: Tensor) -> tuple[Tensor, None]:
        return self.layers(static), None


class Recurrent(nn.Module):
    def __init__(
        self, cell: type[nn.GRU | nn.LSTM], n_variables: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.rnn = cell(
            3 * n_variables, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, values: Tensor, mask: Tensor, time_since_observed: Tensor, static: Tensor) -> tuple[Tensor, None]:
        output, _ = self.rnn(torch.cat([values, mask, time_since_observed], dim=-1))
        return output[:, -1], None


class GRUD(nn.Module):
    """GRU-D, which decays missing values towards the mean and the hidden state with the time since the last observation."""

    def __init__(self, n_variables: int, hidden_size: int, dropout: float):
        super().__init__()
        self.input_decay = nn.Parameter(torch.zeros(n_variables))
        self.input_decay_bias = nn.Parameter(torch.zeros(n_variables))
        self.hidden_decay = nn.Linear(n_variables, hidden_size)
        self.cell = nn.GRUCell(2 * n_variables, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor, time_since_observed: Tensor, static: Tensor) -> tuple[Tensor, None]:
        hidden = values.new_zeros(values.shape[0], self.cell.hidden_size)
        for t in range(values.shape[1]):
            delta = time_since_observed[:, t]
            input_decay = torch.exp(-torch.relu(self.input_decay * delta + self.input_decay_bias))
            # values hold the last observation, and standardized variables have mean 0
            imputed = mask[:, t] * values[:, t] + (1 - mask[:, t]) * input_decay * values[:, t]
            hidden = torch.exp(-torch.relu(self.hidden_decay(delta))) * hidden
            hidden = self.cell(torch.cat([imputed, mask[:, t]], dim=-1), hidden)
        return self.dropout(hidden), None


class TCN(nn.Module):
    """Temporal convolutional network of causal convolutions whose dilation doubles with every layer."""

    def __init__(self, n_variables: int, hidden_size: int, num_layers: int, kernel_size: int, dropout: float):
        super().__init__()
        self.kernel_size = kernel_size
        sizes = [3 * n_variables, *[hidden_size] * num_layers]
        self.convolutions = nn.ModuleList(
            nn.Conv1d(n_in, n_out, kernel_size, dilation=2**layer)
            for layer, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True))
        )
        self.residuals = nn.ModuleList(
            nn.Conv1d(n_in, n_out, 1) for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor, time_since_observed: Tensor, static: Tensor) -> tuple[Tensor, None]:
        hidden = torch.cat([values, mask, time_since_observed], dim=-1).transpose(1, 2)
        for layer, (convolution, residual) in enumerate(zip(self.convolutions, self.residuals, strict=True)):
            padded = nn.functional.pad(hidden, ((self.kernel_size - 1) * 2**layer, 0))
            hidden = torch.relu(self.dropout(convolution(padded)) + residual(hidden))
        return hidden[:, :, -1], None


class Transformer(nn.Module):
    """Transformer encoder over timepoints, pooled with attention."""

    def __init__(
        self, n_variables: int, n_timepoints: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float
    ):
        super().__init__()
        self.embedding = nn.Linear(3 * n_variables, hidden_size)
        self.position = nn.Parameter(torch.zeros(n_timepoints, hidden_size))
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, 2 * hidden_size, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.pooling = nn.Linear(hidden_size, 1)

    def forward(
        self, values: Tensor, mask: Tensor, time_since_observed: Tensor, static: Tensor
    ) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(self.embedding(torch.cat([values, mask, time_since_observed], dim=-1)) + self.position)
        attention = torch.softmax(self.pooling(hidden).squeeze(-1), dim=1)
        return (attention.unsqueeze(-1) * hidden).sum(dim=1), attention


class RETAIN(nn.Module):
    """RETAIN, which weighs timepoints and variables with attention computed in reverse time."""

    def __init__(self, n_variables: int, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Linear(3 * n_variables, hidden_size)
        self.timepoint_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.variable_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.timepoint_attention = nn.Linear(hidden_size, 1)
        self.variable_attention = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, values: Tensor, mask: Tensor, time_since_observed: Tensor, static: Tensor
    ) -> tuple[Tensor, Tensor]:
        embedded = self.dropout(self.embedding(torch.cat([values, mask, time_since_observed], dim=-1)))
        reversed_embedded = embedded.flip(1)
        timepoint_scores = self.timepoint_attention(self.timepoint_rnn(reversed_embedded)[0]).squeeze(-1).flip(1)
        attention = torch.softmax(timepoint_scores, dim=1)
        variable_weights = torch.tanh(self.variable_attention(self.variable_rnn(reversed_embedded)[0])).flip(1)
        return (attention.unsqueeze(-1) * variable_weights * embedded).sum(dim=1), attention


class Model(nn.Module):
    """A network followed by a linear layer on its embedding and the static covariates."""

    def __init__(self, network: nn.Module, n_static: int, n_outputs: int, embedding_size: int):
        super().__init__()
        self.network = network
        self.head = nn.Linear(embedding_size + n_static, n_outputs)

    def forward(
        self, values: Tensor, mask: Tensor, time_since_observed: Tensor, static: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        embedding, attention = embed(self.network, values, mask, time_since_observed, static)
        return self.head(torch.cat([embedding, static], dim=-1)), embedding, attention


def embed(network: nn.Module, *inputs: Tensor) -> tuple[Tensor, Tensor | None]:
    embedding = network(*inputs)
    return embedding if isinstance(embedding, tuple) else (embedding, None)


def cox_loss(risk: Tensor, target: Tensor) -> Tensor:
    """Negative Cox partial log-likelihood of risk scores for targets of times and whether the event occurred."""
    order = torch.argsort(target[:, 0], descending=True)
    risk, event = risk[order, 0], target[order, 1]
    return -((risk - torch.logcumsumexp(risk, dim=0)) * event).sum() / event.sum().clamp(min=1)
