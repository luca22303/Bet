#!/usr/bin/env sh
# Start RueBet: find Python, set up a virtual environment, launch the app.
#
# Written for `sh` rather than bash so it runs on a stock macOS, a minimal
# Linux container and anything else with a POSIX shell.
#
# It does not assume `pip` is on PATH. On macOS there is often no `pip` at all,
# only `pip3`, and inside a fresh virtual environment the bare name may not be
# exported yet -- `python -m pip` works wherever Python itself does, so that is
# what this uses throughout.

set -eu

# Parameter expansion rather than `dirname`, so this works even on a stripped
# PATH where coreutils is not reachable.
case "$0" in
    */*) cd "${0%/*}" || exit 1 ;;
esac

say() { printf '%s\n' "$*"; }
fail() { printf '\n%s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------- guard: paste

# zsh does not treat `#` as a comment in an interactive shell unless
# interactive_comments is set, and it is off by default -- which is every macOS
# user since Catalina. A line copied as `./start.sh --refresh  # macOS` therefore
# arrives here with the comment as arguments, and the failure surfaces much
# later as an unrecognised-argument error from a command they did not type.
for arg in "$@"; do
    case "$arg" in
        '#'*)
            say ""
            say "It looks like a trailing comment was pasted along with the command."
            say "zsh passes '#' through as an argument rather than starting a comment."
            say ""
            say "Run just this, with nothing after it:"
            say "  ./start.sh --refresh"
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------- find python

PYTHON=""
for candidate in python3 python python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        # 3.10 is the floor: the codebase uses `X | None` annotations.
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

# Windows shells that reach this script (Git Bash, WSL) usually have the `py`
# launcher rather than a `python3` on PATH.
if [ -z "$PYTHON" ] && command -v py >/dev/null 2>&1; then
    if py -3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON="py -3"
    fi
fi

if [ -z "$PYTHON" ]; then
    fail "No Python 3.10 or newer found.

  macOS:    brew install python
  Ubuntu:   sudo apt install python3 python3-venv
  Windows:  https://www.python.org/downloads/  (tick 'Add python.exe to PATH')

Then run this script again."
fi

say "Python: $($PYTHON --version 2>&1) ($(command -v ${PYTHON%% *}))"

# ------------------------------------------------------------- set up the env

VENV=".venv"
if [ ! -d "$VENV" ]; then
    say "Creating a virtual environment in $VENV ..."
    # Debian and Ubuntu ship venv separately, and the failure is otherwise
    # cryptic enough to send people looking in the wrong place.
    $PYTHON -m venv "$VENV" 2>/dev/null || fail "Could not create a virtual environment.

On Debian or Ubuntu this usually means the venv module is packaged separately:
  sudo apt install python3-venv"
fi

if [ -x "$VENV/bin/python" ]; then
    VPY="$VENV/bin/python"
elif [ -x "$VENV/Scripts/python.exe" ]; then
    VPY="$VENV/Scripts/python.exe"          # Git Bash on Windows
else
    fail "The virtual environment in $VENV looks broken. Delete it and re-run."
fi

# Install only when something is missing or the project changed, so the usual
# start is immediate rather than a dependency resolution every time.
NEEDS_INSTALL=0
"$VPY" -c 'import bet' 2>/dev/null || NEEDS_INSTALL=1
if [ -f "$VENV/.installed" ] && [ pyproject.toml -nt "$VENV/.installed" ]; then
    NEEDS_INSTALL=1
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
    say "Installing RueBet and its dependencies (first run takes a minute) ..."
    "$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "$VPY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
    "$VPY" -m pip install --quiet -e . || fail "Install failed. Re-run without --quiet to see why:
  $VPY -m pip install -e ."
    : > "$VENV/.installed"
fi

# ------------------------------------------------------------------- run it

say ""
exec "$VPY" -m bet.cli serve "$@"
