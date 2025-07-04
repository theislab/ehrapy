from ehrapy._compat import _shell_command_accessible


def test_shell_command_accessible_true(monkeypatch):
    def mock_popen(*args, **kwargs):
        class MockProcess:
            def __init__(self):
                self.returncode = 0

            def communicate(self):
                return ("output", "")

        return MockProcess()

    monkeypatch.setattr("subprocess.Popen", mock_popen)
    assert _shell_command_accessible(["echo", "hello"]) is True


def test_shell_command_accessible_false(monkeypatch):
    def mock_popen(*args, **kwargs):
        class MockProcess:
            def __init__(self):
                self.returncode = 1

            def communicate(self):
                return ("", "error")

        return MockProcess()

    monkeypatch.setattr("subprocess.Popen", mock_popen)
    assert _shell_command_accessible(["nonexistentcommand"]) is False
