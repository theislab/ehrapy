import sys
from typing import List, Mapping, Optional, Tuple

import pandas as pd
from pandas.api.types import infer_dtype
from rich import print


def _detect_categorical_columns(df: pd.DataFrame) -> Mapping[str, List[Optional[str]]]:
    """Autodetect all categorical columns in a DataFrame

    Args:
        df:
           The original dataframe

    Returns:
        A dictionary containing all categorical column names with a hint on whether they need to be encoded or not

    """
    categoricals: Mapping[str, List[Optional[str]]] = {"categorical_encode": [], "categorical_no_encode": []}
    for col_name in df.columns:
        is_cat, key = _is_categorical_column(df[col_name])
        if is_cat:
            categoricals[key].append(col_name)
    return categoricals


def _is_categorical_column(col) -> Tuple[bool, str]:
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
        return True, "categorical_encode"
    try:
        c = pd.Categorical(col)
        # when we only have unary/binary numerical categories
        if 1 <= len(c.categories) <= 2 or (
            (c_dtype == "floating" or c_dtype == "integer") and 1 <= len(c.categories) <= 2
        ):
            if c_dtype != "floating" and c_dtype != "integer":
                return True, "categorical_encode"
            else:
                return True, "categorical_no_encode"
    # TODO: Which type of exception?
    except Exception:
        print(
            f"[bold red] Could not cast column {col.name} to Categorical type. Please file an issue"
            f"at https://github.com/ehrapy!"
        )
        sys.exit(1)
    if c_dtype == "string":
        if len(c.categories) >= len(c):
            return False, ""
        return True, "categorical_encode"
    elif c_dtype == "floating" or c_dtype == "integer":
        # TODO: Find a good threshold
        # at max 80% should be exact same values to be counted as categorical
        if len(c.categories) > len(c) * 0.8:
            return False, ""
        return True, "categorical_no_encode"
    # mixed datatype columns, free text, non categorical numerical columns, datetime
    return False, ""
