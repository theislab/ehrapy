import sys
from dataclasses import dataclass
from typing import List, Mapping, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from rich import print


def _detect_categorical_columns(
    data: np.ndarray, col_names: Union[List[str], pd.Index]
) -> Mapping[str, List[Optional[str]]]:
    """Autodetect all categorical columns in a DataFrame

    ehrapy makes educated guesses on which columns of the data might be of categorical type.
    These need to be encoded into numbers to allow for downstream analysis.
    For details see: :func:`~ehrapy.api.data._categoricals._is_categorical_column`

    Args:
        data: Numpy array of the data to inspect. Usually AnnData's X
        col_names: The names of the column of the data

    Returns:
        A dictionary containing all categorical column names with a hint on whether they need to be encoded or not
    """
    categoricals: Mapping[str, List[Optional[str]]] = {
        "categorical_encoded": [],
        "categorical_not_encoded": [],
        "not_categorical": [],
    }
    for i in range(data.shape[1]):
        # call ravel on each categorical column to get a flattened 1D array rather than a 2D array with one column
        categorical_column = _is_categorical_column(data[::, i : i + 1 :].ravel(), col_names[i])
        categoricals[categorical_column.categorical_type].append(col_names[i])

    return categoricals


@dataclass
class CategoricalColumnType:
    is_categorical: bool
    categorical_type: str


def _is_categorical_column(col: np.ndarray, col_name: str) -> CategoricalColumnType:
    """Check for a single column, whether it's categorical or not.

    For string columns, a column will be counted as categorical, if there are at least two duplicate elements.
    For numerical values, a column will be counted as categorical, if there are at most 50% unique elements.
    Boolean columns or numerical columns with only one or two categories won't be counted as categorical since they
    won't require any encoding.

    Args:
        col: The column to inspect
        col_name: The name of the column to inspect

    Returns:
        Whether a column is categorical or not and the appropriate key indicating whether this columns needs encoding or not
    """
    c_dtype = infer_dtype(col)
    if c_dtype == "categorical":
        return CategoricalColumnType(True, "categorical_encoded")
    try:
        categorical = pd.Categorical(col)
        # when we only have unary/binary numerical categories
        if 1 <= len(categorical.categories) <= 2 or (
            (c_dtype == "floating" or c_dtype == "integer") and 1 <= len(categorical.categories) <= 2
        ):
            if c_dtype != "floating" and c_dtype != "integer":
                return CategoricalColumnType(True, "categorical_encoded")
            else:
                return CategoricalColumnType(True, "categorical_not_encoded")
    except ValueError:
        print(
            f"[bold red] Could not cast column {col_name} to Categorical type.\n"
            f"Please file an issue at https://github.com/ehrapy!"
        )
        sys.exit(1)
    if c_dtype == "string":
        # As discussed: will currently leave it as it is; freetext -> medcat first!
        # if len(categorical.categories) >= len(categorical):
        # return CategoricalColumnType(False, "not_categorical")
        return CategoricalColumnType(True, "categorical_encoded")
    elif c_dtype == "floating" or c_dtype == "integer" or c_dtype == "mixed-integer-float":
        # TODO: Find a good threshold (need to apply to real data to find this; can not fix now)
        if len(categorical.categories) > len(categorical) * 0.5:
            return CategoricalColumnType(False, "not_categorical")
        return CategoricalColumnType(True, "categorical_not_encoded")
    # free text, non categorical numerical columns, datetime
    return CategoricalColumnType(False, "not_categorical")
