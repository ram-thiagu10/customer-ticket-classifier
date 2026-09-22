"""
_bootstrap.py — One-line setup to import from course/_shared/.

Usage at the top of any demo, solution, or project file:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_shared"))

    from llm_provider import get_chat_model  # works from anywhere in the repo

The exact ``parents[N]`` index varies with file depth — the boilerplate below
auto-resolves by walking up the directory tree until it finds ``_shared/``.

Usage (recommended):

    from _bootstrap import setup_path
    setup_path()  # adds course/_shared/ to sys.path
    from llm_provider import get_chat_model

``setup_path()`` is idempotent — calling it twice is harmless.
"""

from __future__ import annotations

import sys
from pathlib import Path


def setup_path() -> Path:
    """Add ``course/_shared/`` to ``sys.path``.

    Walks up from this file's location until it finds the directory that
    contains ``course/_shared/``, then prepends that directory to ``sys.path``.

    Returns the absolute path to ``course/_shared/`` so callers can use it
    directly if they prefer.

    Idempotent — safe to call multiple times.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "_shared"
        if candidate.is_dir() and (candidate / "llm_provider.py").is_file():
            target = str(candidate)
            if target not in sys.path:
                sys.path.insert(0, target)
            return candidate
    raise RuntimeError(
        "Could not locate course/_shared/ by walking up from "
        f"{here}. Check that the _shared directory exists."
    )


# Auto-run when imported: makes the boilerplate a single line per file.
setup_path()