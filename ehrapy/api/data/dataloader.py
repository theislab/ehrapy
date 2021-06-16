import glob
import pathlib
import tempfile
from typing import Union, List
from zipfile import ZipFile

import pandas as pd
import requests
from rich import print
from rich.progress import Progress


class Dataloader:
    """Responsible for downloading, extracting and transforming input files into AnnData objects"""

    def __init__(self, output_file_name: str, path: str = None):
        """

        Args:
            output_file_name: File name to output the
            path: Path to save the file to
            is_zip: Whether the file to download is a zip file and needs unzipping (default: False)
        """
        self.output_file_name = output_file_name
        self.output_path = path or tempfile.gettempdir()
        self.download_to_path = f"{path}/{output_file_name}"

    def download(self, url: str, block_size: int = 1024, overwrite: bool = False) -> None:
        """Downloads a dataset irrespective of the format.

        Args:
            url: URL to download
            block_size: Block size for downloads in bytes
            overwrite: Whether to overwrite existing files (default: False)
        """
        if pathlib.Path(self.download_to_path).exists():
            print(f"[bold red]File {self.download_to_path} already exists!")
            if not overwrite:
                return

        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))

        with Progress() as progress:
            task = progress.add_task("[red]Downloading...", total=total)
            with open(self.download_to_path, "wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress.update(task, advance=block_size)

    def unzip(self) -> None:
        """Unzips a zip file and saves it."""
        with ZipFile(self.download_to_path, "r") as zip_obj:
            zip_obj.extractall(path=self.output_path)

    def read_csvs(self, csvs: Union[str, List], sep: str = ",") -> pd.DataFrame:
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
