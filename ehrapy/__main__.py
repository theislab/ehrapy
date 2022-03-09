#!/usr/bin/env python
"""Command-line interface."""
import click
from rich import traceback


@click.command()
@click.version_option(version="0.2.0", message=click.style("ehrapy Version: 0.2.0"))
def main() -> None:
    """ehrapy."""


if __name__ == "__main__":
    traceback.install()
    main(prog_name="ehrapy")  # pragma: no cover
