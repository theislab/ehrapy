from __future__ import annotations

from io import StringIO

from ehrapy import print_versions


def test_print_versions_to_stdout(capsys):
    print_versions()

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert "Python" in captured.out or "dependencies" in captured.out.lower()


def test_print_versions_to_file():
    buffer = StringIO()

    print_versions(file=buffer)

    output = buffer.getvalue()
    assert len(output) > 0
