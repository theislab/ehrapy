# Machine learning

Prediction models for patient outcomes, from linear and gradient boosting models to recurrent and transformer networks for time series.
Models are trained, calibrated and evaluated on patient-level splits of static and longitudinal data.

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

These models need the `ml` extra, `pip install 'ehrapy[ml]'`.

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

```{eval-rst}
.. autosummary::
    :toctree: ml
    :nosignatures:

    ml.permutation_importance
```
