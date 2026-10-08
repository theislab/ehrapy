# Machine learning

Prediction models for patient outcomes, built on scikit-learn.
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

## Evaluation

```{eval-rst}
.. autosummary::
    :toctree: ml
    :nosignatures:

    ml.evaluate
```
