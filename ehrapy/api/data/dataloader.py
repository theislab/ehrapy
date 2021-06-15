import pathlib
import tempfile
from zipfile import ZipFile

import requests
from rich import print
from rich.progress import Progress


class Dataloader:
    """Responsible for downloading, extracting and transforming input files into AnnData objects"""

    def __init__(self, url: str, output_file_name: str, path: str = None, is_zip: bool = False):
        """

        Args:
            url: URL to download
            output_file_name: File name to output the
            path: Path to save the file to
            is_zip: Whether the file to download is a zip file and needs unzipping (default: False)
        """
        self.url = url
        self.output_file_name = output_file_name
        self.path = path or tempfile.gettempdir()
        self.tmp_output_path = f"{path}/{output_file_name}"
        self.zip = is_zip

    def download(self, block_size: int = 1024, overwrite: bool = False) -> None:
        """Downloads a dataset irrespective of the format.

        Args:
            block_size: Block size for downloads in bytes
            overwrite: Whether to overwrite existing files (default: False)
        """
        if pathlib.Path(self.tmp_output_path).exists():
            print(f"[bold red]File {self.tmp_output_path} already exists!")
            if not overwrite:
                return

        response = requests.get(self.url, stream=True)
        total = int(response.headers.get("content-length", 0))

        with Progress() as progress:
            task = progress.add_task("[red]Downloading...", total=total)
            with open(self.tmp_output_path, "wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress.update(task, advance=block_size)

    def unzip(self) -> None:
        """Unzips a zip file and saves it."""
        with ZipFile(self.tmp_output_path, "r") as zip_obj:
            zip_obj.extractall(path=self.path)
