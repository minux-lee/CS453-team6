from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import config


class EnvironmentError_(RuntimeError):
    pass


@dataclass
class GitState:
    branch: str
    modified_files: List[str]
    untracked_files: List[str]
    fitness_modified: bool
    raw_status: str


def _run(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False
    )


def which_or_raise(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise EnvironmentError_(f"Required tool '{tool}' not found on PATH.")
    return path


def java_version() -> str:
    out = _run(["java", "-version"])
    return (out.stderr or out.stdout).splitlines()[0].strip() if (out.stderr or out.stdout) else "unknown"


def parse_git_state() -> GitState:
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], config.REPO_ROOT).stdout.strip()
    status = _run(["git", "status", "--porcelain"], config.REPO_ROOT).stdout

    modified, untracked = [], []
    for line in status.splitlines():
        if not line.strip():
            continue
        code, _, path = line[:2], line[2], line[3:]
        if code.strip() == "??":
            untracked.append(path)
        else:
            modified.append(path)

    fitness_rel = str(config.FITNESS_SOURCE.relative_to(config.REPO_ROOT))
    diff = _run(["git", "diff", "--", fitness_rel], config.REPO_ROOT).stdout
    fitness_modified = bool(diff.strip()) or any(
        fitness_rel in m for m in modified
    )

    return GitState(
        branch=branch,
        modified_files=modified,
        untracked_files=untracked,
        fitness_modified=fitness_modified,
        raw_status=status,
    )


def diversity_logic_present() -> bool:
    if not config.FITNESS_SOURCE.exists():
        return False
    text = config.FITNESS_SOURCE.read_text(encoding="utf-8", errors="ignore")
    markers = ("computeDiversityDeficiency", "DIVERSITY_COUPLING", "MutantTypeEntropy")
    return all(m in text for m in markers)


def benchmark_deps_present(cfg: config.SweepConfig) -> List[str]:
    missing: List[str] = []
    for bench in cfg.benchmarks:
        for cp in bench.classpath:
            if not Path(cp).exists() and cp not in missing:
                missing.append(cp)
    return missing


def verify(cfg: config.SweepConfig, require_diversity_source: bool = True) -> GitState:
    which_or_raise("java")
    which_or_raise("mvn")

    git = parse_git_state()

    if require_diversity_source and not diversity_logic_present():
        raise EnvironmentError_(
            "Diversity-aware fitness logic not found in "
            f"{config.FITNESS_SOURCE}. Refusing to benchmark stock EvoSuite."
        )

    missing = benchmark_deps_present(cfg)
    if missing:
        raise EnvironmentError_(
            "Missing benchmark classpath entries:\n  " + "\n  ".join(missing)
        )

    return git


def summary(cfg: config.SweepConfig) -> str:
    git = parse_git_state()
    lines = [
        "Environment summary",
        "-------------------",
        f"  java            : {java_version()}",
        f"  git branch      : {git.branch}",
        f"  fitness modified: {git.fitness_modified}",
        f"  diversity logic : {diversity_logic_present()}",
        f"  evosuite jar    : {config.EVOSUITE_JAR} "
        f"({'present' if config.EVOSUITE_JAR.exists() else 'MISSING'})",
        f"  benchmarks      : {len(cfg.benchmarks)}",
    ]
    return "\n".join(lines)
