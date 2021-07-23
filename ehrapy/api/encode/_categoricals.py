import sys
from typing import List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from rich import print


def _detect_categorical_columns(arr: np.ndarray, names: List[str]) -> Mapping[str, List[Optional[str]]]:
    """Autodetect all categorical columns in a DataFrame

    Args:
        df:
           The original dataframe

    Returns:
        A dictionary containing all categorical column names with a hint on whether they need to be encoded or not

    """
    categoricals: Mapping[str, List[Optional[str]]] = {
        "categorical_encoded": [],
        "categorical_not_encoded": [],
        "not_categorical": [],
    }
    for i in range(arr.shape[1]):
        is_cat, key = _is_categorical_column(arr[::, i : i + 1 :].ravel(), names[i])
        categoricals[key].append(names[i])
    return categoricals


def _is_categorical_column(col: np.ndarray, name: str) -> Tuple[bool, str]:
    """Check for a single column, whether it's categorical or not.

    For string columns, a column will be counted as categorical, if there are at least two duplicate elements.
    For numerical values, a column will be counted as categorical, if there are at most TODO non-unique elements.
    Boolean columns or numerical columns with only one or two categories won't be counted as categorical, since they
    won't require any encoding.

    Args:
        col: The column

    Returns:
        Whether a column is categorical or not and the appropriate key indicating whether this columns needs encoding or not
    """
    c_dtype = infer_dtype(col)
    if c_dtype == "categorical":
        return True, "categorical_encoded"
    try:
        c = pd.Categorical(col)
        # when we only have unary/binary numerical categories
        if 1 <= len(c.categories) <= 2 or (
            (c_dtype == "floating" or c_dtype == "integer") and 1 <= len(c.categories) <= 2
        ):
            if c_dtype != "floating" and c_dtype != "integer":
                return True, "categorical_encoded"
            else:
                return True, "categorical_not_encoded"
    # TODO: Which type of exception?
    except Exception:
        print(
            f"[bold red] Could not cast column {name} to Categorical type. Please file an issue"
            f"at https://github.com/ehrapy!"
        )
        sys.exit(1)
    if c_dtype == "string":
        if len(c.categories) >= len(c):
            return False, "not_categorical"
        return True, "categorical_encoded"
    elif c_dtype == "floating" or c_dtype == "integer" or c_dtype == "mixed-integer-float":
        # TODO: Find a good threshold
        if len(c.categories) > len(c) * 0.5:
            return False, "not_categorical"
        return True, "categorical_not_encoded"
    # free text, non categorical numerical columns, datetime
    return False, "not_categorical"
