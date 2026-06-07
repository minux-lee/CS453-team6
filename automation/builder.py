from __future__ import annotations

import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from . import config

_FITNESS_CLASS_ENTRY = (
    "org/evosuite/coverage/mutation/StrongMutationSuiteFitness.class"
)
_DIVERSITY_MARKER = b"MutantTypeEntropy"


@dataclass
class BuildResult:
    rebuilt: bool
    jar_path: Path
    verified: bool
    message: str


def _mvn(args: List[str]) -> subprocess.CompletedProcess:
    cmd = ["mvn", *args, "-DskipTests", "-Dossindex.skip=true", "-q"]
    return subprocess.run(
        cmd, cwd=str(config.EVOSUITE_DIR), capture_output=True, text=True, check=False
    )


def jar_contains_diversity(jar_path: Path) -> bool:
    if not jar_path.exists():
        return False
    try:
        with zipfile.ZipFile(jar_path) as zf:
            with zf.open(_FITNESS_CLASS_ENTRY) as fh:
                return _DIVERSITY_MARKER in fh.read()
    except (KeyError, zipfile.BadZipFile):
        return False


def build(clean: bool = True) -> BuildResult:
    goal = ["clean", "install"] if clean else ["install"]
    proc = _mvn(["-pl", "client,master", *goal])
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        return BuildResult(
            rebuilt=True,
            jar_path=config.EVOSUITE_JAR,
            verified=False,
            message=f"Maven build failed (exit {proc.returncode}):\n{tail}",
        )

    verified = jar_contains_diversity(config.EVOSUITE_JAR)
    return BuildResult(
        rebuilt=True,
        jar_path=config.EVOSUITE_JAR,
        verified=verified,
        message="Build succeeded and jar verified."
        if verified
        else "Build succeeded but jar is MISSING the diversity bytecode.",
    )


def ensure_built() -> BuildResult:
    if jar_contains_diversity(config.EVOSUITE_JAR):
        return BuildResult(
            rebuilt=False,
            jar_path=config.EVOSUITE_JAR,
            verified=True,
            message="Existing jar already contains diversity bytecode.",
        )
    return build(clean=True)
