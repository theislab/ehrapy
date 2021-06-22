import tempfile
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
from anndata import AnnData
from rich import print

from ehrapy.api.data.dataloader import Dataloader


class Empty(Enum):
    token = 0


_empty = Empty.token


def read(
    filename: Union[Path, str],
    backed: Optional[Literal["r", "r+"]] = None,
    sheet: Optional[str] = None,
    extension: Optional[str] = None,
    delimiter: Optional[str] = None,
    patient_id: Optional[str] = None,
    first_column_names: bool = False,
    backup_url: Optional[str] = None,
    cache: bool = False,
    cache_compression: Union[Literal["gzip", "lzf"], None, Empty] = _empty,
    **kwargs,
) -> AnnData:
    """Read file and return :class:`~anndata.AnnData` object.

    To speed up reading, consider passing ``cache=True``, which creates an hdf5 cache file.

    Args:
        filename: Name of the input file to read
        backed: If ``'r'``, load :class:`~anndata.AnnData` in ``backed`` mode instead of fully loading it into memory (`memory` mode).
                If you want to modify backed attributes of the AnnData object, you need to choose ``'r+'``.
        sheet: Name of sheet/table in hdf5 or Excel file.
        extension: Extension that indicates the file type. If ``None``, uses extension of filename.
        delimiter: Delimiter that separates data within text file. If ``None``, will split at arbitrary number of white spaces,
                   which is different from enforcing splitting at any single white space ``' '``.
        patient_id: Patient id to merge the csv files on (if several) and to use as index
        first_column_names: Assume the first column stores row names. This is only necessary if these are not strings:
                            strings in the first column are automatically assumed to be row names.
        backup_url: Retrieve the file from an URL if not present on disk.
        cache: If `False`, read from source, if `True`, read from fast 'h5ad' cache.
        cache_compression: See the h5py :ref:`dataset_compression`. (Default: `settings.cache_compression`)

    Returns:
        An :class:`~anndata.AnnData` object
    """
    output_file_name: str = ""
    file = Path(filename)
    if not file.exists():
        print("[bold yellow]Path or dataset does not yet exist. Attempting to download...")
        output_file_name = backup_url.split("/")[-1]
        is_zip: bool = output_file_name.endswith(".zip")
        Dataloader.download(backup_url, output_file_name=output_file_name, is_zip=is_zip)
    else:
        # Depending on the file extension of the input file or primary files inside the directory call the appropriate read function

        merged_df: pd.DataFrame = Dataloader.read_csvs(
            csvs=f"{tempfile.gettempdir()}/{output_file_name}", on=patient_id
        )
        merged_df
    # 5. Get it into an AnnData format
    # 6. Mark

    # Dataloader.download(
    #     url="https://physionet.org/content/mimic2-iaccd/1.0/full_cohort_data.csv",
    #     output_file_name="mimic_2.csv",
    #     is_zip=False,
    # )

    pass
    # filename = Path(filename)  # allow passing strings
    # if is_valid_filename(filename):
    #     return _read(
    #         filename,
    #         backed=backed,
    #         sheet=sheet,
    #         ext=ext,
    #         delimiter=delimiter,
    #         first_column_names=first_column_names,
    #         backup_url=backup_url,
    #         cache=cache,
    #         cache_compression=cache_compression,
    #         **kwargs,
    #     )
    # # generate filename and read to dict
    # filekey = str(filename)
    # filename = settings.writedir / (filekey + '.' + settings.file_format_data)
    # if not filename.exists():
    #     raise ValueError(
    #         f'Reading with filekey {filekey!r} failed, '
    #         f'the inferred filename {filename!r} does not exist. '
    #         'If you intended to provide a filename, either use a filename '
    #         f'ending on one of the available extensions {avail_exts} '
    #         'or pass the parameter `ext`.'
    #     )
    # return read_h5ad(filename, backed=backed)
