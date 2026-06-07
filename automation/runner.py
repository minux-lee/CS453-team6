from __future__ import annotations

import csv
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .config import RunSpec


@dataclass
class RunResult:
    config_id: str
    target_class: str
    coupling: str
    k: float
    seed: int
    search_budget_s: int
    status: str
    wall_time_s: float
    stats: Dict[str, str] = field(default_factory=dict)
    error: str = ""

    def merged_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "config_id": self.config_id,
            "target_class": self.target_class,
            "coupling": self.coupling,
            "k": self.k,
            "seed": self.seed,
            "search_budget_s": self.search_budget_s,
            "status": self.status,
            "wall_time_s": round(self.wall_time_s, 2),
            "error": self.error,
        }
        for key, value in self.stats.items():
            row[f"es_{key}"] = value
        return row


def build_command(spec: RunSpec, jar: Path, work_dir: Path) -> List[str]:
    cp = ":".join(spec.benchmark.classpath)
    cmd = [
        "java",
        f"-Xmx{spec.client_mem_mb + 1000}m",
        "-jar",
        str(jar),
        "-generateSuite",
        "-class",
        spec.benchmark.target_class,
        "-projectCP",
        cp,
        "-criterion",
        spec.criterion,
        "-seed",
        str(spec.seed),
        "-mem",
        str(spec.client_mem_mb),
        f"-Dsearch_budget={spec.search_budget_s}",
        "-Dstopping_condition=MaxTime",
        f"-Ddiversity_coupling={spec.coupling}",
        f"-Ddiversity_k={spec.k}",
        f"-Ddiversity_cap={spec.diversity_cap}",
        f"-Dpopulation={spec.population}",
        "-Dtest_archive=false",
        "-Dassertions=false",
        "-Djunit_check=false",
        "-Dminimize=false",
        "-Dshow_progress=false",
        "-Dtest_dir=" + str(work_dir / "tests"),
        "-Doutput_variables=" + ",".join(config.OUTPUT_VARIABLES),
        "-Dreport_dir=" + str(work_dir / "evosuite-report"),
    ]
    return cmd


def _parse_statistics(report_dir: Path) -> Optional[Dict[str, str]]:
    stats_file = report_dir / "statistics.csv"
    if not stats_file.exists():
        return None
    with stats_file.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    return rows[-1]


def run_one(spec: RunSpec, jar: Path, keep_workdir: bool = False,
            results_base: Optional[Path] = None) -> RunResult:
    base = results_base or config.RAW_STATS_DIR
    work_dir = base / spec.config_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir = work_dir / "evosuite-report"

    cmd = build_command(spec, jar, work_dir)
    log_path = config.RUN_LOG_DIR / f"{spec.config_id}.log"
    start = time.time()
    status, error = "ok", ""

    try:
        with log_path.open("w") as log_fh:
            log_fh.write("$ " + " ".join(cmd) + "\n\n")
            log_fh.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=spec.run_timeout_s,
                check=False,
            )
        if proc.returncode != 0:
            status, error = "error", f"exit code {proc.returncode}"
    except subprocess.TimeoutExpired:
        status, error = "timeout", f"exceeded {spec.run_timeout_s}s"

    wall = time.time() - start
    stats = _parse_statistics(report_dir)
    if stats is None:
        if status == "ok":
            status, error = "no_stats", "statistics.csv missing or empty"
        stats = {}

    if not keep_workdir:
        tests_dir = work_dir / "tests"
        if tests_dir.exists():
            shutil.rmtree(tests_dir, ignore_errors=True)

    return RunResult(
        config_id=spec.config_id,
        target_class=spec.benchmark.target_class,
        coupling=spec.coupling,
        k=spec.k,
        seed=spec.seed,
        search_budget_s=spec.search_budget_s,
        status=status,
        wall_time_s=wall,
        stats=stats,
        error=error,
    )
