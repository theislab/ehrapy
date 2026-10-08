from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn

    from ehrapy.ml._task import Kind


@dataclass(frozen=True, kw_only=True)
class Trainer:
    """How deep learning models are trained.

    Training stops once the loss on the tuning set has not improved for `patience` epochs, and the parameters of the epoch with the lowest loss are kept.

    Examples:
        >>> import ehrapy as ep
        >>> model = ep.ml.GRU(trainer=ep.ml.Trainer(max_epochs=20, device="cpu"))
    """

    #: Maximum number of passes over the training set.
    max_epochs: int = 100
    #: Number of observations per gradient step.
    batch_size: int = 128
    #: Learning rate of the AdamW optimizer.
    learning_rate: float = 1e-3
    #: Weight decay of the AdamW optimizer.
    weight_decay: float = 1e-4
    #: Number of epochs without improvement on the tuning set after which training stops.
    patience: int = 10
    #: `"balanced"` to weigh classes and labels inversely to their frequency in the training set.
    class_weight: Literal["balanced"] | None = None
    #: Device to train and predict on, such as `"cpu"` or `"cuda"`, or `"auto"` for a GPU if one is available.
    device: str = "auto"


@dataclass(frozen=True, kw_only=True)
class DeepModel:
    """A network whose embedding, together with the static covariates, feeds a linear output layer."""

    #: Whether the model reads the time series of variables instead of their summaries.
    sequential: ClassVar[bool] = True

    #: How the model is trained.
    trainer: Trainer = Trainer()

    def _network(self, n_variables: int, n_timepoints: int, n_static: int) -> nn.Module:
        raise NotImplementedError

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        kind: Kind,
        n_outputs: int,
        n_static: int,
        tuning: tuple[np.ndarray, np.ndarray],
        random_state: int,
    ) -> _FittedModel:
        torch = _torch()
        from ehrapy.ml._networks import Model, embed

        torch.manual_seed(random_state)
        fitted = _FittedModel(self, kind, n_static, X, y)
        inputs, targets = fitted._inputs(X), fitted._targets(y)
        network = self._network(inputs[0].shape[2], inputs[0].shape[1], n_static)
        with torch.no_grad():
            embedding, attention = embed(network, *(tensor[:1] for tensor in inputs))
        fitted.n_embedding = embedding.shape[1]
        fitted.n_attention = 0 if attention is None else attention.shape[1]
        fitted.module = Model(network, inputs[3].shape[1], n_outputs, fitted.n_embedding).to(fitted.device)
        fitted.n_epochs = _train(
            fitted,
            inputs,
            targets,
            fitted._inputs(tuning[0]),
            fitted._targets(tuning[1]),
            _loss(kind, y, n_outputs, self.trainer),
            random_state,
        )
        return fitted


@dataclass(frozen=True, kw_only=True)
class MLP(DeepModel):
    """Multilayer perceptron on the summaries of longitudinal variables and the static covariates.

    Examples:
        >>> import ehrapy as ep
        >>> model = ep.ml.MLP(hidden_size=32, num_layers=1)
    """

    sequential: ClassVar[bool] = False

    #: Number of units of every hidden layer.
    hidden_size: int = 64
    #: Number of hidden layers.
    num_layers: int = 2
    #: Probability of dropping a unit while training.
    dropout: float = 0.1

    def _network(self, n_variables: int, n_timepoints: int, n_static: int) -> nn.Module:
        from ehrapy.ml import _networks

        return _networks.MLP(n_static, self.hidden_size, self.num_layers, self.dropout)


@dataclass(frozen=True, kw_only=True)
class GRU(DeepModel):
    """Gated recurrent unit network on the time series of the variables, whether they were observed and the time since their last observation.

    Examples:
        >>> import ehrapy as ep
        >>> model = ep.ml.GRU(hidden_size=32)
    """

    #: Number of hidden units.
    hidden_size: int = 64
    #: Number of stacked recurrent layers.
    num_layers: int = 1
    #: Probability of dropping a unit while training.
    dropout: float = 0.1

    def _network(self, n_variables: int, n_timepoints: int, n_static: int) -> nn.Module:
        from torch import nn

        from ehrapy.ml import _networks

        return _networks.Recurrent(nn.GRU, n_variables, self.hidden_size, self.num_layers, self.dropout)


@dataclass(frozen=True, kw_only=True)
class LSTM(DeepModel):
    """Long short-term memory network on the time series of the variables, whether they were observed and the time since their last observation.

    Examples:
        >>> import ehrapy as ep
        >>> model = ep.ml.LSTM(hidden_size=32)
    """

    #: Number of hidden units.
    hidden_size: int = 64
    #: Number of stacked recurrent layers.
    num_layers: int = 1
    #: Probability of dropping a unit while training.
    dropout: float = 0.1

    def _network(self, n_variables: int, n_timepoints: int, n_static: int) -> nn.Module:
        from torch import nn

        from ehrapy.ml import _networks

        return _networks.Recurrent(nn.LSTM, n_variables, self.hidden_size, self.num_layers, self.dropout)


@dataclass(frozen=True, kw_only=True)
class GRUD(DeepModel):
    """GRU-D :cite:`che2018recurrent`, a gated recurrent unit network whose inputs decay towards the mean and whose hidden state decays with the time since the last observation.

    Examples:
        >>> import ehrapy as ep
        >>> model = ep.ml.GRUD(hidden_size=32)
    """

    #: Number of hidden units.
    hidden_size: int = 64
    #: Probability of dropping a unit of the final hidden state while training.
    dropout: float = 0.1

    def _network(self, n_variables: int, n_timepoints: int, n_static: int) -> nn.Module:
        from ehrapy.ml import _networks

        return _networks.GRUD(n_variables, self.hidden_size, self.dropout)


@dataclass(frozen=True, kw_only=True)
class TCN(DeepModel):
    """Temporal convolutional network :cite:`bai2018empirical` of causal convolutions on the time series of the variables, whether they were observed and the time since their last observation.

    Examples:
        >>> import ehrapy as ep
        >>> model = ep.ml.TCN(hidden_size=32)
    """

    #: Number of channels of every convolution.
    hidden_size: int = 64
    #: Number of convolutions, whose dilation doubles with every layer.
    num_layers: int = 3
    #: Number of timepoints every convolution spans.
    kernel_size: int = 3
    #: Probability of dropping a unit while training.
    dropout: float = 0.1

    def _network(self, n_variables: int, n_timepoints: int, n_static: int) -> nn.Module:
        from ehrapy.ml import _networks

        return _networks.TCN(n_variables, self.hidden_size, self.num_layers, self.kernel_size, self.dropout)


@dataclass(frozen=True, kw_only=True)
class Transformer(DeepModel):
    """Transformer encoder :cite:`vaswani2017attention` over timepoints, pooled with attention that is stored by :func:`~ehrapy.ml.predict`.

    Examples:
        >>> import ehrapy as ep
        >>> model = ep.ml.Transformer(hidden_size=32, num_heads=2)
    """

    #: Number of features of every timepoint.
    hidden_size: int = 64
    #: Number of encoder layers.
    num_layers: int = 2
    #: Number of attention heads, which must divide `hidden_size`.
    num_heads: int = 4
    #: Probability of dropping a unit while training.
    dropout: float = 0.1

    def _network(self, n_variables: int, n_timepoints: int, n_static: int) -> nn.Module:
        from ehrapy.ml import _networks

        return _networks.Transformer(
            n_variables, n_timepoints, self.hidden_size, self.num_layers, self.num_heads, self.dropout
        )


@dataclass(frozen=True, kw_only=True)
class RETAIN(DeepModel):
    """RETAIN :cite:`choi2016retain`, which weighs timepoints with attention that is stored by :func:`~ehrapy.ml.predict`.

    Examples:
        >>> import ehrapy as ep
        >>> model = ep.ml.RETAIN(hidden_size=32)
    """

    #: Number of features of every timepoint.
    hidden_size: int = 64
    #: Probability of dropping a unit while training.
    dropout: float = 0.1

    def _network(self, n_variables: int, n_timepoints: int, n_static: int) -> nn.Module:
        from ehrapy.ml import _networks

        return _networks.RETAIN(n_variables, self.hidden_size, self.dropout)


@dataclass(frozen=True, kw_only=True)
class _Module(DeepModel):
    network: Any

    def _network(self, n_variables: int, n_timepoints: int, n_static: int) -> nn.Module:
        return copy.deepcopy(self.network)


DEEP_MODELS: dict[str, type[DeepModel]] = {
    "mlp": MLP,
    "gru": GRU,
    "lstm": LSTM,
    "grud": GRUD,
    "tcn": TCN,
    "transformer": Transformer,
    "retain": RETAIN,
}


def _deep_model(model: object) -> DeepModel | None:
    """The deep learning model a name or torch module stands for, if any."""
    if isinstance(model, DeepModel):
        return model
    if isinstance(model, str):
        return DEEP_MODELS[model]() if model in DEEP_MODELS else None
    torch = sys.modules.get("torch")
    return _Module(network=model) if torch is not None and isinstance(model, torch.nn.Module) else None


class _FittedModel:
    """A trained network with the statistics that standardize its inputs and targets."""

    module: nn.Module
    n_embedding: int
    n_attention: int
    n_epochs: int
    target_mean: np.ndarray | float
    target_std: np.ndarray | float

    def __init__(self, model: DeepModel, kind: Kind, n_static: int, X: np.ndarray, y: np.ndarray):
        self.model, self.kind, self.n_static = model, kind, n_static
        self.device = _device(model.trainer.device)
        if model.sequential:
            self.mean, self.std = _moments(X[:, : X.shape[1] - n_static], axis=(0, 2))
            self.static_mean, self.static_std = _moments(X[:, X.shape[1] - n_static :, 0], axis=0)
        self.target_mean, self.target_std = _moments(y, axis=0) if kind == "regression" else (0.0, 1.0)

    def outputs(self, X: np.ndarray) -> np.ndarray:
        """Predictions, embedding and attention of every observation, side by side."""
        torch = _torch()

        outputs, embedding, attention = _forward(self, self._inputs(X))
        match self.kind:
            case "binary" | "multilabel":
                outputs = torch.sigmoid(outputs)
            case "multiclass":
                outputs = torch.softmax(outputs, dim=1)
            case "regression":
                outputs = outputs * self.target_std + self.target_mean
        return torch.cat([outputs, embedding, *([] if attention is None else [attention])], dim=1).double().numpy()

    def _inputs(self, X: np.ndarray) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Values, whether they were observed and the time since their last observation of shape `(observations, timepoints, variables)`, and static covariates."""
        torch = _torch()

        if not self.model.sequential:
            empty = np.empty((len(X), 0, 0))
            arrays = [empty, empty, empty, X]
        else:
            n_variables = X.shape[1] - self.n_static
            values = (X[:, :n_variables] - self.mean[:, None]) / self.std[:, None]
            observed = ~np.isnan(values)
            steps = np.arange(X.shape[2])
            last = np.maximum.accumulate(np.where(observed, steps, -1), axis=2)
            filled = np.take_along_axis(np.nan_to_num(values), np.maximum(last, 0), axis=2) * (last >= 0)
            since = np.where(last >= 0, steps - last, steps + 1) / X.shape[2]
            static = np.nan_to_num((X[:, n_variables:, 0] - self.static_mean) / self.static_std)
            arrays = [*(np.moveaxis(array, 1, 2) for array in (filled, observed, since)), static]
        return tuple(torch.as_tensor(array, dtype=torch.float32) for array in arrays)

    def _targets(self, y: np.ndarray) -> Tensor:
        torch = _torch()

        if self.kind == "multiclass":
            return torch.as_tensor(y, dtype=torch.long)
        return torch.as_tensor(((y - self.target_mean) / self.target_std).reshape(len(y), -1), dtype=torch.float32)


def _train(
    fitted: _FittedModel,
    inputs: tuple[Tensor, ...],
    targets: Tensor,
    tuning_inputs: tuple[Tensor, ...],
    tuning_targets: Tensor,
    loss: Callable[[Tensor, Tensor], Tensor],
    random_state: int,
) -> int:
    """Train with early stopping on the tuning loss and return the number of epochs."""
    torch = _torch()

    trainer, module = fitted.model.trainer, fitted.module
    optimizer = torch.optim.AdamW(module.parameters(), lr=trainer.learning_rate, weight_decay=trainer.weight_decay)
    rng = np.random.default_rng(random_state)
    best_loss, best_state, waiting, epochs = math.inf, copy.deepcopy(module.state_dict()), 0, 0
    while epochs < trainer.max_epochs and waiting < trainer.patience:
        epochs += 1
        module.train()
        for batch in np.array_split(rng.permutation(len(targets)), max(1, len(targets) // trainer.batch_size)):
            optimizer.zero_grad()
            outputs = module(*(tensor[batch].to(fitted.device) for tensor in inputs))[0]
            loss(outputs, targets[batch].to(fitted.device)).backward()
            optimizer.step()
        if len(tuning_targets):
            tuning_outputs = _forward(fitted, tuning_inputs)[0].to(fitted.device)
            tuning_loss = loss(tuning_outputs, tuning_targets.to(fitted.device)).item()
            waiting += 1
            if tuning_loss < best_loss:
                best_loss, best_state, waiting = tuning_loss, copy.deepcopy(module.state_dict()), 0
    if len(tuning_targets):
        module.load_state_dict(best_state)
    return epochs


def _forward(fitted: _FittedModel, inputs: tuple[Tensor, ...]) -> tuple[Tensor, Tensor, Tensor | None]:
    """Outputs, embedding and attention of the network in batches, on the CPU."""
    torch = _torch()

    fitted.module.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(inputs[0]), fitted.model.trainer.batch_size):
            batch = (tensor[start : start + fitted.model.trainer.batch_size].to(fitted.device) for tensor in inputs)
            results.append([None if result is None else result.cpu() for result in fitted.module(*batch)])
    outputs, embedding, attention = zip(*results, strict=True)
    return torch.cat(outputs), torch.cat(embedding), None if attention[0] is None else torch.cat(attention)


def _loss(kind: Kind, y: np.ndarray, n_outputs: int, trainer: Trainer) -> Callable[[Tensor, Tensor], Tensor]:
    torch = _torch()
    from ehrapy.ml._networks import cox_loss

    balanced = trainer.class_weight == "balanced"
    match kind:
        case "binary" | "multilabel":
            positives = y.reshape(len(y), -1).sum(axis=0)
            weight = torch.as_tensor((len(y) - positives) / np.maximum(positives, 1), dtype=torch.float32)
            return torch.nn.BCEWithLogitsLoss(pos_weight=weight.to(_device(trainer.device)) if balanced else None)
        case "multiclass":
            counts = np.bincount(y.astype(int), minlength=n_outputs)
            weight = torch.as_tensor(len(y) / (n_outputs * np.maximum(counts, 1)), dtype=torch.float32)
            return torch.nn.CrossEntropyLoss(weight=weight.to(_device(trainer.device)) if balanced else None)
        case "regression":
            return torch.nn.MSELoss()
    return cox_loss


def _moments(values: np.ndarray, *, axis: int | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation ignoring missing values, 0 and 1 where they are undefined or the deviation is 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean, std = np.nanmean(values, axis=axis), np.nanstd(values, axis=axis)
    return np.nan_to_num(mean), np.where(np.isnan(std) | (std == 0), 1.0, std)


def _device(device: str) -> Any:
    torch = _torch()

    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise ImportError("Deep learning models need PyTorch. Install it with `pip install 'ehrapy[ml]'`.") from e
    return torch
