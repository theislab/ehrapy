from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from random import choice
from string import ascii_lowercase
from typing import Literal

import requests
from filelock import FileLock
from rich import print
from rich.progress import Progress


def download(
    url: str,
    archive_format: Literal["zip", "tar", "tar.gz", "tgz"] = None,
    output_file_name: str = None,
    output_path: str | Path = None,
    block_size: int = 1024,
    overwrite: bool = False,
) -> None:  # pragma: no cover
    """Downloads a file irrespective of format.

    Args:
        url: URL to download.
        archive_format: The format if an archive file.
        output_file_name: Name of the downloaded file.
        output_path: Path to download/extract the files to. Defaults to 'OS tmpdir' if not specified.
        block_size: Block size for downloads in bytes.
        overwrite: Whether to overwrite existing files.
    """
    if output_file_name is None:
        letters = ascii_lowercase
        output_file_name = f"ehrapy_tmp_{''.join(choice(letters) for _ in range(10))}"

    if output_path is None:
        output_path = tempfile.gettempdir()

    def _sanitize_file_name(file_name):
        if os.name == "nt":
            file_name = file_name.replace("?", "_").replace("*", "_")
        return file_name

    download_to_path = Path(
        _sanitize_file_name(
            f"{output_path}{output_file_name}"
            if str(output_path).endswith("/")
            else f"{output_path}/{output_file_name}"
        )
    )

    Path(output_path).mkdir(parents=True, exist_ok=True)
    lock_path = f"{download_to_path}.lock"
    with FileLock(lock_path):
        if download_to_path.exists():
            warning = f"[bold red]File {download_to_path} already exists!"
            if not overwrite:
                print(warning)
                return
            else:
                print(f"{warning} Overwriting...")

        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))

        temp_file_name = f"{download_to_path}.part"

        with Progress(refresh_per_second=5) as progress:
            task = progress.add_task("[red]Downloading...", total=total)
            with Path(temp_file_name).open("wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress.update(task, advance=block_size)

            # force the progress bar to 100% at the end
            progress.update(task, completed=total, refresh=True)

            Path(temp_file_name).replace(download_to_path)

        if archive_format:
            output_path = output_path or tempfile.gettempdir()
            shutil.unpack_archive(download_to_path, output_path, format=archive_format)
            download_to_path.unlink()
            list_of_paths = [path for path in Path(output_path).resolve().glob("*/") if not path.name.startswith(".")]
            latest_path = max(list_of_paths, key=lambda path: path.stat().st_ctime)
            shutil.move(latest_path, latest_path.parent / remove_archive_extension(output_file_name))  # type: ignore

    Path(lock_path).unlink(missing_ok=True)


def remove_archive_extension(file_path):
    return (
        str(Path(file_path).with_suffix(""))
        if any(
            Path(file_path).suffix.endswith(ext)
            for ext in [
                ".zip",
                ".tar",
                ".tar.gz",
                ".tgz",
                ".tar.bz2",
                ".tbz2",
                ".tar.xz",
                ".txz",
            ]
        )
        else file_path
    )
