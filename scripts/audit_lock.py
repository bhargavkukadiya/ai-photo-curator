"""Audit every distinct registry version in uv.lock, across all branches/extras."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path


def main() -> int:
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    versions: dict[str, set[str]] = defaultdict(set)
    for package in tomllib.loads(lock.read_text())["package"]:
        if "registry" in package["source"]:
            versions[package["name"]].add(package["version"])
    result = 0
    with tempfile.TemporaryDirectory(prefix="curator_audit_") as directory:
        for index in range(max(map(len, versions.values()), default=0)):
            pins = [
                f"{name}=={sorted(values)[index]}"
                for name, values in sorted(versions.items())
                if len(values) > index
            ]
            requirements = Path(directory) / f"requirements-{index}.txt"
            report = Path(directory) / f"report-{index}.json"
            requirements.write_text("\n".join(pins) + "\n")
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--disable-pip",
                    "--no-deps",
                    "--progress-spinner",
                    "off",
                    "--format",
                    "json",
                    "-r",
                    str(requirements),
                    "-o",
                    str(report),
                ],
                check=False,
            )
            result = max(result, run.returncode)
            if not report.exists():
                result = max(result, 1)
                continue
            audit = json.loads(report.read_text())
            for dependency in audit["dependencies"]:
                if dependency.get("skip_reason") or dependency.get("vulns"):
                    print(json.dumps(dependency))
                    result = max(result, 1)
            print(f"Audited {len(pins)} locked package versions in bucket {index}.")
    return result


if __name__ == "__main__":
    sys.exit(main())
