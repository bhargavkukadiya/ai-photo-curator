"""Install the built wheel into the calling CI environment and smoke-test it."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    dist = Path(__file__).resolve().parents[1] / "dist"
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"Expected exactly one wheel in '{dist}', found {len(wheels)}. "
            "Clean dist/ and rebuild the wheel before retrying."
        )
    wheel = wheels[0]
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--reinstall",
            str(wheel),
        ],
        check=True,
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import photo_curator; assert Path(photo_curator.__file__).parent.parent.name == 'site-packages'",
            ],
            cwd=directory,
            env=env,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "photo_curator", "--help"],
            cwd=directory,
            env=env,
            check=True,
        )
        for name in ("ai-photo-curator", "photo-curator"):
            executable = Path(sys.executable).parent / (
                name + (".exe" if os.name == "nt" else "")
            )
            subprocess.run(
                [str(executable), "--help"], cwd=directory, env=env, check=True
            )


if __name__ == "__main__":
    main()
