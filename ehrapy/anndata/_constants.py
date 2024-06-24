# Typing Column
# -----------------------
# The column name and used values in adata.var for column types.

FEATURE_TYPE_KEY = "feature_type"
NUMERIC_TAG = "numeric"
CATEGORICAL_TAG = "categorical"
DATE_TAG = "date"

NORM_NAMES = {
    "StandardScaler" : "scale",
    "MinMaxScaler" : "minmax", 
    "MaxAbsScaler" : "maxabs",
    "PowerTransformer" : "quantile",
    "QuantileTransformer" : "power",
    "RobustScaler" : "robust_scale",
    "log_norm" : "log"
}