"""Refuse external fitting or tuning while the competition bundle is reachable."""

import os
from pathlib import Path


def require_external_only():
    """Training, tuning and confirmation must run with MTAH_DATA_DIR pointing nowhere."""
    location = os.environ.get("MTAH_DATA_DIR")
    if location is None:
        raise SystemExit("Set MTAH_DATA_DIR to a nonexistent directory for external-only fitting")
    if Path(location).exists():
        raise SystemExit("External-only fitting requires the competition bundle to be unavailable")
