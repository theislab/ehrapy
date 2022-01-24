from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from random import choice
from string import ascii_lowercase

import requests
from rich import print
from rich.progress import Progress


def download(
    url: str,
    output_file_name: str = None,
    output_path: str | Path = None,
    block_size: int = 1024,
    overwrite: bool = False,
    is_archived: bool = False,
) -> None:  # pragma: no cover
    """Downloads a dataset irrespective of the format.

    Args:
        url: URL to download
        output_file_name: Name of the downloaded file
        output_path: Path to download/extract the files to (default: OS tmpdir)
        block_size: Block size for downloads in bytes (default: 1024)
        overwrite: Whether to overwrite existing files (default: False)
        is_archived: Whether the downloaded file needs to be unarchived (default: False)
    """
    if output_file_name is None:
        letters = ascii_lowercase
        output_file_name = f"ehrapy_tmp_{''.join(choice(letters) for _ in range(10))}"

    if output_path is None:
        output_path = tempfile.gettempdir()

    download_to_path = (
        f"{output_path}{output_file_name}" if str(output_path).endswith("/") else f"{output_path}/{output_file_name}"
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

    with Progress(refresh_per_second=1500) as progress:
        task = progress.add_task("[red]Downloading...", total=total)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(download_to_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                progress.update(task, advance=block_size)

    if is_archived:
        output_path = output_path or tempfile.gettempdir()
        shutil.unpack_archive(download_to_path, output_path)
