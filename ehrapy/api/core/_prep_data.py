import pandas as pd
import sys
from rich import print
from pandas.api.types import infer_dtype
from typing import List, Optional


def _detect_categorical_columns(df: pd.DataFrame) -> List[Optional[str]]:
    """Autodetect all categorical columns in a DataFrame

        Args:
            df:
               The original dataframe

        Returns:
            A list of column names, which where detected to be categorical
        """
    categoricals = []
    for col_name in df.columns:
        if _is_categorical_column(df[col_name]):
            categoricals.append(col_name)
    return categoricals


def _is_categorical_column(col) -> bool:
    """Check for a single column, whether it's categorical or not.

        For string columns, a column will be counted as categorical, if there are at least two duplicate elements.
        For numerical values, a column will be counted as categorical, if there are at most TODO non-unique elements.
        Boolean columns or numerical columns with only one or two categories won't be counted as categorical, since they
        won't require any encoding.

        Args:
            col: The column

        Returns:
            Whether a column is categorical or not
        """
    c_dtype = infer_dtype(col)
    if c_dtype == 'categorical':
        return True
    try:
        c = pd.Categorical(col)
        # when we only have unary/binary numerical categories we don't need any encoding
        if c_dtype == 'boolean' or (1 <= len(c.categories) <= 2 and (c_dtype == 'floating' or c_dtype == 'integer')):
            return False
    # TODO: Which type of exception?
    except Exception:
        print(f'[bold red] Could not cast column {col.name} to Categorical type. Please file an issue'
              f'at https://github.com/ehrapy!')
        sys.exit(1)
    if c_dtype == 'string':
        if len(c.categories) >= len(c):
            return False
        return True
    elif c_dtype == 'floating' or c_dtype == 'integer':
        # TODO: Find a good threshold
        # at max 80% should be exact same values to be counted as categorical
        if len(c.categories) > len(c) * 0.8:
            return False
        return True
    # mixed datatype columns, free text, non categorical numerical columns, datetime
    return False



