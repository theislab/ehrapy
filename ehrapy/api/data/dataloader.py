import tempfile
from pathlib import Path
from random import choice
from string import ascii_lowercase
from typing import Union
from zipfile import ZipFile

import requests
from rich import print
from rich.progress import Progress


class Dataloader:
    """Responsible for downloading and extracting input files"""

    @staticmethod
    def download(  # pragma: no cover
        url: str,
        output_file_name: str = None,
        output_path: Union[str, Path] = None,
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
        if output_file_name is None:
            letters = ascii_lowercase
            output_file_name = f"ehrapy_tmp_{''.join(choice(letters) for _ in range(10))}"

        if output_path is None:
            output_path = tempfile.gettempdir()

        download_to_path = (
            f"{output_path}{output_file_name}"
            if str(output_path).endswith("/")
            else f"{output_path}/{output_file_name}"
        )
        if Path(download_to_path).exists():
            warning = f"[bold red]File {download_to_path} already exists!"
            if not overwrite:
                print(warning)
                return
            else:
                print(f"{warning} Overwriting...")

        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))

        with Progress() as progress:
            task = progress.add_task("[red]Downloading...", total=total)
            Path(output_path).mkdir(parents=True, exist_ok=True)
            with open(download_to_path, "wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress.update(task, advance=block_size)

        if is_zip:
            output_path = output_path or tempfile.gettempdir()
            with ZipFile(download_to_path, "r") as zip_obj:
                zip_obj.extractall(path=output_path)
                extracted = zip_obj.namelist()
                print(extracted)
