from ehrapy._utils_available import _check_module_importable, _shell_command_accessible


def test_check_module_importable_true():
    assert _check_module_importable("math") is True


def test_check_module_importable_false():
    assert _check_module_importable("nonexistentmodule12345") is False


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


