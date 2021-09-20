from ehrapy.api.data.dataloader import Dataloader
from ehrapy.api.data.datasets import Datasets


def mimic_2(encode: bool = False):
    """Downloads and returns a prepared AnnData object of the clinical data from the MIMIC-II database (https://physionet.org/content/mimic2-iaccd/1.0/)

    Args:
        encode: Whether to return an already encoded AnnData object

    Returns:
        An :class:`~anndata.AnnData` object with the (optionally encoded) values in X

    Example:
        .. code-block:: python

                import ehrapy.api as ep
                adata = eh.data.mimic_2(encode=True)
    """
    return Datasets.mimic_2(encode=encode)
