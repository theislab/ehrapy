# Machine learning

Prediction models for patient outcomes, from linear and gradient boosting models to recurrent and transformer networks for time series.
A {class}`~ehrapy.ml.Task` names the label and the timepoints the features come from, {func}`~ehrapy.ml.split` assigns patients to a training, a tuning and a held-out set, and {func}`~ehrapy.ml.fit` learns a model on the training set only.

```{eval-rst}
.. module:: ehrapy
    :no-index:
```

## Tasks and splits

```{eval-rst}
.. autosummary::
    :toctree: ml
    :nosignatures:

    ml.Task
    ml.split
```

## Training and prediction

```{eval-rst}
.. autosummary::
    :toctree: ml
    :nosignatures:

    ml.fit
    ml.predict
    ml.Predictor
```

## Deep learning models

Models that need PyTorch, which `pip install 'ehrapy[ml]'` installs.
Except for the multilayer perceptron, they read the time series of the variables instead of their summaries.

```{eval-rst}
.. autosummary::
    :toctree: ml
    :nosignatures:

    ml.MLP
    ml.GRU
    ml.LSTM
    ml.GRUD
    ml.TCN
    ml.Transformer
    ml.RETAIN
    ml.DeepModel
    ml.Trainer
```

## Evaluation

```{eval-rst}
.. autosummary::
    :toctree: ml
    :nosignatures:

    ml.evaluate
```

## Calibration and uncertainty

```{eval-rst}
.. autosummary::
    :toctree: ml
    :nosignatures:

    ml.calibrate
    ml.conformalize
```

## Interpretation

Plot the importances with {func}`~ehrapy.plot.rank_features_supervised`.

```{eval-rst}
.. autosummary::
    :toctree: ml
    :nosignatures:

    ml.permutation_importance
```
