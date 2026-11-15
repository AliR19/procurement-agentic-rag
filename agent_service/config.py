"""Runtime configuration (all overridable with environment variables)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = Path(os.getenv("POLICY_DIR", ROOT / "data" / "policies"))
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / ".chroma"))
