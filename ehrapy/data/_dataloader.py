from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from random import choice
from string import ascii_lowercase

import requests
from rich import print
from rich.progress import Progress

from ehrapy import logging as logg


def download(
    url: str,
    output_file_name: str = None,
    output_path: str | Path = None,
    block_size: int = 1024,
    overwrite: bool = False,
    is_archived: bool = False,
) -> None:  # pragma: no cover
    """Downloads a file irrespective of format.

    Args:
        url: URL to download.
        output_file_name: Name of the downloaded file.
        output_path: Path to download/extract the files to. Defaults to 'OS tmpdir'.
        block_size: Block size for downloads in bytes.Defaults to 1024.
        overwrite: Whether to overwrite existing files. Defaults to False.
        is_archived: Whether the downloaded file needs to be unarchived. Defaults to False.
    """
    if output_file_name is None:
        letters = ascii_lowercase
        output_file_name = f"ehrapy_tmp_{''.join(choice(letters) for _ in range(10))}"

    if output_path is None:
        output_path = tempfile.gettempdir()

    def _sanitize_file_name(file_name):
        # Remove forbidden characters for Windows
        if os.name == "nt":
            windows_forbidden = '<>:"/\\|?*'
            file_name = "".join(c for c in file_name if c not in windows_forbidden)

        # Remove trailing periods and whitespace (valid for all platforms)
        file_name = file_name.rstrip(". ").strip()

        return file_name

    download_to_path = Path(
        _sanitize_file_name(
            f"{output_path}{output_file_name}"
            if str(output_path).endswith("/")
            else f"{output_path}/{output_file_name}"
        )
    )

    if download_to_path.exists():
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
        with open(download_to_path.resolve(), "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                progress.update(task, advance=block_size)

        # force the progress bar to 100% at the end
        progress.update(task, completed=total, refresh=True)

    if is_archived:
        output_path = output_path or tempfile.gettempdir()
        shutil.unpack_archive(download_to_path, output_path)

    logg.debug(f"Downloaded `{output_file_name}` to `{output_path}`.")
