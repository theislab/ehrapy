import json
import logging
import sys
import urllib.request
from subprocess import PIPE, Popen, check_call
from urllib.error import HTTPError, URLError

from pkg_resources import parse_version
from rich import print

import ehrapy
from ehrapy.cli.custom_cli.questionary import ehrapy_questionary

log = logging.getLogger(__name__)


class UpgradeCommand:
    """Responsible for checking for newer versions ehrapy and upgrading it if required."""

    @staticmethod
    def check_upgrade_ehrapy() -> None:
        """Checks whether the locally installed version of ehrapy is the latest.

        If not it prompts whether to upgrade and runs the upgrade command if desired.
        """
        if not UpgradeCommand.check_ehrapy_latest():
            if ehrapy_questionary(function="confirm", question="Do you want to upgrade?", default="y"):
                UpgradeCommand.upgrade_ehrapy()

    @classmethod
    def check_ehrapy_latest(cls) -> bool:
        """Checks whether the locally installed version of ehrapy is the latest available on PyPi.

        Returns:
            True if locally version is the latest or PyPI is inaccessible, False otherwise
        """
        latest_local_version = ehrapy.__version__
        sliced_local_version = (
            latest_local_version[:-9] if latest_local_version.endswith("-SNAPSHOT") else latest_local_version
        )
        log.debug(f"Latest local ehrapy version is: {latest_local_version}.")
        log.debug("Checking whether a new ehrapy version exists on PyPI.")
        try:
            # Retrieve info on latest version
            # Adding nosec (bandit) here, since we have a hardcoded https request
            # It is impossible to access file:// or ftp://
            # See: https://stackoverflow.com/questions/48779202/audit-url-open-for-permitted-schemes-allowing-use-of-file-or-custom-schemes
            req = urllib.request.Request("https://pypi.org/pypi/ehrapy/json")  # nosec
            with urllib.request.urlopen(req, timeout=1) as response:  # nosec
                contents = response.read()
                data = json.loads(contents)
                latest_pypi_version = data["info"]["version"]
        except (HTTPError, TimeoutError, URLError):
            print(
                "[bold red]Unable to contact PyPI to check for the latest ehrapy version."
                " Do you have an internet connection?"
            )
            # Returning true by default, since this is not a serious issue
            return True

        if parse_version(sliced_local_version) > parse_version(latest_pypi_version):
            print(
                f"[bold yellow]Installed version {latest_local_version} of ehrapy is newer than the latest release {latest_pypi_version}!"
                f" You are running a nightly version and features may break!"
            )
        elif parse_version(sliced_local_version) == parse_version(latest_pypi_version):
            return True
        else:
            print(
                f"[bold red]Installed version {latest_local_version} of ehrapy is outdated. Newest version is {latest_pypi_version}!"
            )
            return False

        return False

    @classmethod
    def upgrade_ehrapy(cls) -> None:
        """Calls pip as a subprocess with the --upgrade flag to upgrade ehrapy to the latest version."""
        log.debug("Attempting to upgrade ehrapy via   pip install --upgrade ehrapy   .")
        if not UpgradeCommand.is_pip_accessible():
            sys.exit(1)
        try:
            check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ehrapy"])
        except Exception as e:
            print("[bold red]Unable to upgrade ehrapy")
            print(f"[bold red]Exception: {e}")

    @classmethod
    def is_pip_accessible(cls) -> bool:
        """Verifies that pip is accessible and in the PATH.

        Returns:
            True if accessible, False if not
        """
        log.debug("Verifying that pip is accessible.")
        pip_installed = Popen(["pip", "--version"], stdout=PIPE, stderr=PIPE, universal_newlines=True)
        (git_installed_stdout, git_installed_stderr) = pip_installed.communicate()
        if pip_installed.returncode != 0:
            log.debug("Pip was not accessible! Attempted to test via   pip --version   .")
            print("[bold red]Unable to find 'pip' in the PATH. Is it installed?")
            print("[bold red]Run command was [green]'pip --version '")
            return False

        return True
