from ehrapy.api.data.dataloader import Dataloader
from ehrapy.api.data.datasets import Datasets


def mimic_2_demo(encode: bool = False):
    """Downloads and returns a prepared AnnData object of the clinical data from the MIMIC-II database (https://physionet.org/content/mimic2-iaccd/1.0/)

    Args:
        encode: Whether to return an already encoded AnnData object

    Returns:
        An :class:`~anndata.AnnData` object with the (optionally encoded) values in X

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = eh.data.mimic_2_demo(encode=True)
    """
    return Datasets.mimic_2_demo(encode=encode)


def mimic_3_demo(encode: bool = False, mudata: bool = False):
    """Downloads and returns a prepared MuData object of the clinical data from the MIMIC-III database (https://physionet.org/content/mimic2-iaccd/1.0/)

    Args:
        encode: Whether to return an already encoded MuData object
        mudata: Whether to return a MuData object. Returns a Dictionary of file names to AnnData objects if False

    Returns:
        An :class:`~mudata.MuData` object with the (optionally encoded) values in X

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            mudata = eh.data.mimic_3_demo(encode=True, return_mudata=True)
    """
    return Datasets.mimic_3_demo(encode=encode, mudata=mudata)
