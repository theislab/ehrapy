from ehrapy.api.io.read import DataReader
from ehrapy.api.io.write import DataWriter


def read(
    filename,
    extension=None,
    delimiter=None,
    index_column=None,
    columns_obs_only=None,
    cache: bool = False,
    backup_url=None,
):
    return DataReader.read(filename, extension, delimiter, index_column, columns_obs_only, cache, backup_url)
