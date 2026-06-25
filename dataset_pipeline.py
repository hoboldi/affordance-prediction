"""
Backward-compatible entry point for SAM3D batch reconstruction.

Prefer: ``python scripts/generate_sam3d.py`` or ``python dataset_pipeline.py`` (this file).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for path in (_ROOT, _ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.generate_sam3d import main

if __name__ == "__main__":
    main()
