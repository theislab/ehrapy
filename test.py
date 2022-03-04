import numpy as np

test = np.array([-5, 5, 6, -7])
import warnings

with warnings.catch_warnings():
     warnings.simplefilter("error")
    
     try:
         normalized = np.log1p(test)
     except RuntimeWarning:
         print("now")

