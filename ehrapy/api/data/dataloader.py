import glob
import pathlib
import tempfile
from typing import List, Union
from zipfile import ZipFile

import pandas as pd
import requests
from anndata import AnnData
from rich import print
from rich.progress import Progress


class Dataloader:
    """Responsible for downloading and extracting input files"""

    @staticmethod
    def download(
        url: str,
        output_file_name: str,
        output_path: str,
        block_size: int = 1024,
        overwrite: bool = False,
        is_zip: bool = False,
    ) -> None:
        """Downloads a dataset irrespective of the format.

        Args:
            url: URL to download
            output_file_name: Name of the downloaded file
            output_path: Path to download/extract the files to (default: OS tmpdir)
            block_size: Block size for downloads in bytes (default: 1024)
            overwrite: Whether to overwrite existing files (default: False)
            is_zip: Whether the downloaded file needs to be unzipped (default: False)
        """
        download_to_path = f"{output_path}/{output_file_name}"
        if pathlib.Path(download_to_path).exists():
            print(f"[bold red]File {download_to_path} already exists!")
            if not overwrite:
                return

        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))

        with Progress() as progress:
            task = progress.add_task("[red]Downloading...", total=total)
            with open(download_to_path, "wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress.update(task, advance=block_size)

        if is_zip:
            output_path = download_to_path or tempfile.gettempdir()
            with ZipFile(download_to_path, "r") as zip_obj:
                zip_obj.extractall(path=output_path)

    @staticmethod
    def read_csvs(csvs: Union[str, List], sep: str = ",") -> pd.DataFrame:
        """Reads one or several csv files and returns a (merged) Pandas DataFrame

        Args:
            csvs: One or multiple paths to a folder of csv files or csv files directly
            sep: Separator of the csv file

        Returns:
            A single Pandas DataFrame of all csv files merged
        """
        if not isinstance(csvs, List):
            # Not a single csv directly, but a folder containing multiple csv files
            if not (csvs.endswith("csv") or csvs.endswith("tsv")):
                csvs = glob.glob(f'*.{"csv"}') + glob.glob(f'*.{"tsv"}')
            # path to single csv file directly
            else:
                csvs = [csvs]

        combined_csvs_df = pd.concat([pd.read_csv(f, sep=sep) for f in csvs])

        return combined_csvs_df

    @staticmethod
    def df_to_anndata(dataframe: pd.DataFrame) -> AnnData:
        """Transforms a single (concatenated) Pandas DataFrame into an AnnData object.

        This transformation does not yet perform any encodings on the AnnData object.

        Args:
            dataframe:

        Returns:
            AnnData object where X contains the raw data
        """
        pass
