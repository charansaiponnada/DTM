"""
src/logger.py
──────────────
Centralized logging system for the DTM Drainage AI pipeline.

Features:
  - Rotating file logs (daily + size-based rotation)
  - Separate log files per pipeline stage
  - Color-coded console output via loguru + rich
  - JSON structured logs for machine-readable audit trail
  - Timing decorators to profile each stage
  - Summary report at end of run
"""

from __future__ import annotations
import sys
import json
import time
import functools
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

# Global rich console
console = Console()

# Track per-stage timings and events for the summary report
_stage_log: list[dict] = []
_run_start: float = 0.0


# ══════════════════════════════════════════════════════════════════════════
#  Log Directory & File Naming
# ══════════════════════════════════════════════════════════════════════════

def _make_log_dir(base: str | Path = "logs") -> Path:
    log_dir = Path(base)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _run_id() -> str:
    """Unique ID for this pipeline run: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ══════════════════════════════════════════════════════════════════════════
#  Setup — Call Once at Pipeline Start
# ══════════════════════════════════════════════════════════════════════════

def setup_logging(
    log_dir: str | Path = "logs",
    run_id: Optional[str] = None,
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "14 days",
    console_level: str = "INFO",
) -> str:
    """
    Configure loguru sinks:
      1. Console (stdout)  — INFO and above, colored
      2. Main log file     — DEBUG and above, all stages combined
      3. Error log file    — WARNING and above only
      4. JSON log file     — machine-readable structured log

    Parameters
    ----------
    log_dir       : directory to write log files
    run_id        : unique run identifier (auto-generated if None)
    level         : minimum level for file logs (DEBUG recommended)
    rotation      : rotate file after this size
    retention     : delete files older than this
    console_level : minimum level shown in terminal

    Returns
    -------
    run_id string
    """
    global _run_start
    _run_start = time.time()

    if run_id is None:
        run_id = _run_id()

    log_dir = _make_log_dir(log_dir)

    # ── Remove default loguru sink ───────────────────────────────────────
    logger.remove()

    # ── Sink 1: Console (colored, clean format) ──────────────────────────
    logger.add(
        sys.stdout,
        level=console_level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )

    # ── Sink 2: Main combined log (DEBUG+) ───────────────────────────────
    main_log = log_dir / f"run_{run_id}.log"
    logger.add(
        str(main_log),
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        backtrace=True,
        diagnose=True,
    )

    # ── Sink 3: Errors only ──────────────────────────────────────────────
    error_log = log_dir / f"errors_{run_id}.log"
    logger.add(
        str(error_log),
        level="WARNING",
        rotation="5 MB",
        retention=retention,
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line}\n"
            "  → {message}\n"
        ),
        backtrace=True,
        diagnose=True,
    )

    # ── Sink 4: JSON structured log ──────────────────────────────────────
    json_log = log_dir / f"structured_{run_id}.jsonl"

    def json_sink(message):
        record = message.record
        entry = {
            "timestamp": record["time"].isoformat(),
            "level":     record["level"].name,
            "logger":    record["name"],
            "function":  record["function"],
            "line":      record["line"],
            "message":   record["message"],
            "run_id":    run_id,
        }
        with open(json_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    logger.add(json_sink, level="DEBUG", format="{message}")

    # ── Announce ─────────────────────────────────────────────────────────
    console.print(Panel(
        Text.assemble(
            ("DTM Drainage AI Pipeline\n", "bold cyan"),
            ("MoPR Geospatial Intelligence Hackathon\n\n", "dim"),
            ("Run ID  : ", "dim"), (run_id, "bold white"), ("\n", ""),
            ("Log dir : ", "dim"), (str(log_dir), "bold white"), ("\n", ""),
            ("Level   : ", "dim"), (level, "bold yellow"),
        ),
        title="[bold green]Pipeline Starting[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    logger.info(f"Logging initialised | run_id={run_id} | log_dir={log_dir}")
    return run_id


# ══════════════════════════════════════════════════════════════════════════
#  Stage-Level Logging Context
# ══════════════════════════════════════════════════════════════════════════

class StageLogger:
    """
    Context manager that wraps a pipeline stage with:
      - Rich panel header / footer
      - Elapsed time measurement
      - Automatic error capture
      - Stage result appended to run summary

    Usage
    -----
    with StageLogger("Ground Classification", stage_num=2, total_stages=6) as sl:
        result = classify_ground(...)
        sl.set_result({"points_classified": 64_000_000})
    """

    def __init__(
        self,
        stage_name: str,
        stage_num: int,
        total_stages: int = 6,
        log_dir: str | Path = "logs",
    ):
        self.stage_name   = stage_name
        self.stage_num    = stage_num
        self.total_stages = total_stages
        self.log_dir      = Path(log_dir)
        self._result      = {}
        self._start       = 0.0
        self._stage_file  = None

    def __enter__(self) -> "StageLogger":
        self._start = time.time()

        # Per-stage log file
        stage_slug = self.stage_name.lower().replace(" ", "_")
        self._stage_file = self.log_dir / f"stage_{self.stage_num:02d}_{stage_slug}.log"
        logger.add(
            str(self._stage_file),
            level="DEBUG",
            format="{time:HH:mm:ss} | {level: <8} | {message}",
            encoding="utf-8",
            filter=lambda r: True,   # capture everything during this stage
        )

        console.print(f"\n[bold cyan]{'═'*64}[/bold cyan]")
        console.print(
            f"[bold white]  STAGE {self.stage_num}/{self.total_stages} — {self.stage_name.upper()}[/bold white]"
        )
        console.print(f"[bold cyan]{'═'*64}[/bold cyan]")
        logger.info(f"▶ Stage {self.stage_num}/{self.total_stages}: {self.stage_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start

        if exc_type is not None:
            logger.error(
                f"✗ Stage {self.stage_num} FAILED after {elapsed:.1f}s: {exc_val}"
            )
            console.print(
                f"[bold red]  ✗ {self.stage_name} FAILED ({elapsed:.1f}s)[/bold red]\n"
            )
            _stage_log.append({
                "stage":   self.stage_num,
                "name":    self.stage_name,
                "status":  "FAILED",
                "elapsed": round(elapsed, 2),
                "error":   str(exc_val),
                "result":  {},
            })
            return False   # re-raise

        logger.success(
            f"✓ Stage {self.stage_num}: {self.stage_name} "
            f"completed in {elapsed:.1f}s"
        )
        console.print(
            f"[bold green]  ✓ {self.stage_name} — {elapsed:.1f}s[/bold green]\n"
        )
        _stage_log.append({
            "stage":   self.stage_num,
            "name":    self.stage_name,
            "status":  "OK",
            "elapsed": round(elapsed, 2),
            "error":   None,
            "result":  self._result,
        })
        return False

    def set_result(self, result: dict):
        """Attach key metrics to this stage's log entry."""
        self._result = result
        for k, v in result.items():
            logger.info(f"  {k}: {v}")

    def info(self, msg: str):
        logger.info(f"  [{self.stage_name}] {msg}")

    def warning(self, msg: str):
        logger.warning(f"  [{self.stage_name}] {msg}")

    def debug(self, msg: str):
        logger.debug(f"  [{self.stage_name}] {msg}")


# ══════════════════════════════════════════════════════════════════════════
#  Timing Decorator
# ══════════════════════════════════════════════════════════════════════════

def timed(func: Optional[Callable] = None, *, label: Optional[str] = None):
    """
    Decorator that logs function execution time.

    Usage
    -----
    @timed
    def my_function(): ...

    @timed(label="IDW Interpolation")
    def interpolate(): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = label or fn.__qualname__
            t0   = time.perf_counter()
            logger.debug(f"→ {name} started")
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                logger.debug(f"← {name} finished in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.error(f"← {name} raised {type(e).__name__} after {elapsed:.3f}s: {e}")
                raise
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ══════════════════════════════════════════════════════════════════════════
#  Progress Logger (for loops without tqdm)
# ══════════════════════════════════════════════════════════════════════════

class ProgressLogger:
    """
    Logs progress of a known-length loop at regular intervals.

    Usage
    -----
    pl = ProgressLogger("Processing tiles", total=100)
    for i, tile in enumerate(tiles):
        process(tile)
        pl.update(i + 1)
    pl.done()
    """

    def __init__(self, label: str, total: int, log_every_pct: float = 10.0):
        self.label        = label
        self.total        = total
        self.log_every    = max(1, int(total * log_every_pct / 100))
        self._last_logged = 0
        self._start       = time.time()
        logger.info(f"{label}: 0/{total} (0%)")

    def update(self, current: int):
        if current - self._last_logged >= self.log_every or current == self.total:
            elapsed  = time.time() - self._start
            pct      = 100 * current / self.total
            eta_s    = (elapsed / current * (self.total - current)) if current > 0 else 0
            logger.info(
                f"{self.label}: {current}/{self.total} "
                f"({pct:.0f}%) | elapsed {elapsed:.1f}s | ETA {eta_s:.1f}s"
            )
            self._last_logged = current

    def done(self):
        elapsed = time.time() - self._start
        logger.success(f"{self.label}: complete in {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
#  Run Summary Report
# ══════════════════════════════════════════════════════════════════════════

def print_summary(output_dir: Optional[str | Path] = None, save_json: bool = True):
    """
    Print a rich summary table of all completed stages and save a
    JSON report to the log directory.

    Call this at the very end of the pipeline run.
    """
    total_elapsed = time.time() - _run_start

    # ── Rich Table ───────────────────────────────────────────────────────
    table = Table(
        title="Pipeline Run Summary",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Stage",    style="bold white",  width=4)
    table.add_column("Name",     style="cyan",         width=28)
    table.add_column("Status",   width=10)
    table.add_column("Time (s)", justify="right",      width=10)
    table.add_column("Key Results",                    width=38)

    all_ok = True
    for entry in _stage_log:
        status_str  = "[bold green]✓ OK[/bold green]" if entry["status"] == "OK" \
                      else "[bold red]✗ FAILED[/bold red]"
        result_str  = "  ".join(f"{k}={v}" for k, v in entry["result"].items()) \
                      if entry["result"] else (entry["error"] or "—")
        if entry["status"] != "OK":
            all_ok = False
        table.add_row(
            str(entry["stage"]),
            entry["name"],
            status_str,
            f"{entry['elapsed']:.1f}",
            result_str[:60],
        )

    table.add_section()
    overall = "[bold green]ALL STAGES PASSED ✓[/bold green]" if all_ok \
              else "[bold red]SOME STAGES FAILED ✗[/bold red]"
    table.add_row("—", "TOTAL", overall, f"{total_elapsed:.1f}", "")

    console.print()
    console.print(table)

    # ── Output files list ────────────────────────────────────────────────
    if output_dir:
        output_dir = Path(output_dir)
        files_table = Table(title="Output Files", box=box.SIMPLE, border_style="dim")
        files_table.add_column("Format", style="yellow", width=14)
        files_table.add_column("Filename",               width=38)
        files_table.add_column("Size",    justify="right", width=10)

        format_map = {".tif": "COG (raster)", ".gpkg": "GPKG (vector)", ".las": "LAS (cloud)", ".joblib": "Model (.joblib)"}
        for f in sorted(output_dir.rglob("*.*")):
            if f.is_file() and not f.name.startswith("_"):
                size_kb = f.stat().st_size / 1024
                size_str = f"{size_kb/1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
                fmt = format_map.get(f.suffix, f.suffix)
                files_table.add_row(fmt, f.name, size_str)

        console.print(files_table)

    # ── Save JSON report ─────────────────────────────────────────────────
    if save_json:
        report = {
            "run_completed":  datetime.now().isoformat(),
            "total_elapsed_s": round(total_elapsed, 2),
            "all_stages_ok":  all_ok,
            "stages":         _stage_log,
        }
        report_path = Path("logs") / f"summary_{_run_id()}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Summary report saved → {report_path}")

    logger.success(f"Pipeline finished in {total_elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
#  Convenience: log_exception context manager
# ══════════════════════════════════════════════════════════════════════════

class log_exception:
    """
    Suppress and log an exception with context, rather than crashing.

    Usage
    -----
    with log_exception("Optional Open3D visualization", reraise=False):
        import open3d
        open3d.visualization.draw(...)
    """

    def __init__(self, context: str = "", reraise: bool = True):
        self.context = context
        self.reraise = reraise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.warning(
                f"{'[' + self.context + '] ' if self.context else ''}"
                f"{exc_type.__name__}: {exc_val}"
            )
            return not self.reraise   # True = suppress
        return False
