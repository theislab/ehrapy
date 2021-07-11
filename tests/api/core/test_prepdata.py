import pandas as pd
import numpy as np
from typing import List
import random, string

from ehrapy.api.core._prep_data import _detect_categorical_columns


class TestPrepData:
    def test_detect_categorical_columns(self):
        letters = string.ascii_lowercase
        df_test = pd.DataFrame({'A': np.arange(100), 'B': np.arange(100.0), 'C':np.random.randint(10, size=100),
                                'D': [''.join(random.choice(letters) for _ in range(8)) for _ in range(100)], 'E': np.random.randint(2, size=100),
                               'F': [random.choice(['foo', 'bar', 'baz']) for _ in range(100)]})
        cats = _detect_categorical_columns(df_test)
        assert isinstance(cats, List)
        assert cats == ['C', 'F']
