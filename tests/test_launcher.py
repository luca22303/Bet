"""The bootstrap launchers.

`pip` is frequently not on PATH -- macOS ships only `pip3`, and a Windows
install often leaves it unexported even when `python` works. Every install step
therefore goes through `python -m pip`, which works wherever Python does. These
tests hold that line, because a launcher that assumes `pip` fails on the first
machine that does not have it, which is the failure that made it necessary.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
START_SH = ROOT / "start.sh"
START_BAT = ROOT / "start.bat"


def test_both_launchers_exist():
    assert START_SH.exists()
    assert START_BAT.exists()


def test_posix_launcher_is_executable():
    assert START_SH.stat().st_mode & 0o111, "start.sh needs the executable bit"


def test_posix_launcher_is_valid_shell():
    """Checked with `sh`, not bash: it has to run on a stock macOS."""
    result = subprocess.run(["sh", "-n", str(START_SH)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", [START_SH, START_BAT], ids=["sh", "bat"])
def test_launchers_never_call_bare_pip(script):
    """The bug these scripts exist to route around."""
    text = script.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("#", "rem ", "REM ")):
            continue
        # `pip install ...` as a command is the failure; `python -m pip` is not.
        assert not re.search(r"(?<!-m )\bpip3?\s+install\b", stripped), \
            f"calls pip directly: {stripped}"


@pytest.mark.parametrize("script", [START_SH, START_BAT], ids=["sh", "bat"])
def test_launchers_require_a_supported_python(script):
    """3.10 is the floor; the codebase uses `X | None` annotations."""
    text = script.read_text(encoding="utf-8", errors="replace")
    assert "version_info >= (3, 10)" in text


@pytest.mark.parametrize("script", [START_SH, START_BAT], ids=["sh", "bat"])
def test_launchers_name_the_fix_when_python_is_missing(script):
    """A launcher that fails must say what to install, not just that it failed."""
    text = script.read_text(encoding="utf-8", errors="replace").lower()
    assert "python.org/downloads" in text


def test_posix_launcher_names_the_unix_install_commands():
    """Each script names the platforms it actually runs on.

    The Windows one has no business mentioning Homebrew.
    """
    text = START_SH.read_text().lower()
    assert "brew install python" in text
    assert "apt install python3" in text


def test_windows_launcher_has_crlf_line_endings():
    """LF-only breaks block and label parsing in cmd."""
    raw = START_BAT.read_bytes()
    assert b"\r\n" in raw
    assert not re.search(rb"(?<!\r)\n", raw), "found a bare LF"


def test_posix_launcher_handles_a_missing_venv_module():
    """Debian and Ubuntu package venv separately.

    The bare failure is cryptic enough to send people looking in the wrong
    place, so the script names the package.
    """
    assert "python3-venv" in START_SH.read_text()


def test_launchers_pass_arguments_through():
    """`./start.sh --port 9000` has to reach the server."""
    assert '"$@"' in START_SH.read_text()
    assert "%*" in START_BAT.read_text(encoding="utf-8", errors="replace")


def test_posix_launcher_avoids_external_commands_for_its_own_path():
    """Parameter expansion rather than `dirname`, so a stripped PATH still works."""
    text = START_SH.read_text()
    assert "${0%/*}" in text
    assert "$(dirname" not in text
