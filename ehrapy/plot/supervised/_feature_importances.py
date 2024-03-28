import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData


def feature_importances(adata: AnnData, key: str = "feature_importances", n_features: int = 10):
    """
    Plot features with greates absolute importances as a barplot.

    Args:
        adata: :class:`~anndata.AnnData` object storing the data. A key in adata.var should contain the feature
            importances, calculated beforehand.
        key: The key in adata.var to use for feature importances. Defaults to 'feature_importances'.
        n_features: The number of features to plot. Defaults to 10.

    Returns:
        None
    """
    if key not in adata.var.keys():
        raise ValueError(f"Key {key} not found in adata.var.")

    df = pd.DataFrame({"importance": adata.var[key]}, index=adata.var_names)
    df["absolute_importance"] = df["importance"].abs()
    df = df.sort_values("absolute_importance", ascending=False)
    sns.barplot(x=df["importance"][:n_features], y=df.index[:n_features], orient="h")
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
