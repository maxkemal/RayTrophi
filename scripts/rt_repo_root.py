#!/usr/bin/env python3
"""Locate the repository root from either copy of the scripts directory.

★★★★ Why this module exists. CLAUDE.md rule 4 says test scripts live in TWO
places -- `scripts/` and `x64/Release/scripts/` -- and the application runs the
second one. Every source-auditing script derived its root by counting directory
levels upward from its own file (two levels). That is correct for the repo copy
and WRONG for the Release copy, where two levels up is `x64/Release`; every path
then became `x64/Release/RayTrophiStudio/source/...` and the script died with
FileNotFoundError on its first read.

★★★ The failure mode is worth naming: those scripts worked perfectly when a
human ran them from the repo, and failed only when the APPLICATION ran them --
which is the one way that matters, because the app is how an agent drives them.
A check that only passes in the environment nobody tests in is not a check.

The fix is to stop counting levels and SEARCH for a landmark only the repository
root has. That answer is the same from both copies, and from any future third.
"""

from pathlib import Path
import sys

# The landmark: the source tree itself. Cheap, unambiguous, and it cannot be
# satisfied accidentally by a build output directory.
_MARKER = Path("RayTrophiStudio") / "source"


def repo_root(start=None) -> Path:
    """Return the repository root, searching upward from `start`.

    `start` defaults to this module's own directory; a caller exec'd as a string
    (no usable `__file__`) can pass `Path.cwd()` instead.
    """
    if start is None:
        start = Path(__file__).resolve().parent
    start = Path(start).resolve()

    for candidate in [start, *start.parents]:
        if (candidate / _MARKER).is_dir():
            return candidate

    # ★ Fail loudly and SAY WHERE WE LOOKED. A silent fallback to cwd would let
    #   the script read some other tree's files and report confident nonsense --
    #   worse than not running at all.
    raise RuntimeError(
        "repository root not found: no ancestor of {} contains {}".format(start, _MARKER)
    )
