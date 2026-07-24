#!/usr/bin/env python3
"""Run Geant4, then finalize ROOT outputs in a fresh Python process."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def main() -> None:
    arguments = sys.argv[1:]
    subprocess.run(
        [sys.executable, str(HERE / "run_angle.py"), *arguments],
        check=True,
    )
    if "--build-only" not in arguments:
        subprocess.run(
            [sys.executable, str(HERE / "finalize_angle.py"), *arguments],
            check=True,
        )


if __name__ == "__main__":
    main()
