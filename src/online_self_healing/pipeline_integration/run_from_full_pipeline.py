from __future__ import annotations

import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from online_self_healing.pipeline_integration.runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
