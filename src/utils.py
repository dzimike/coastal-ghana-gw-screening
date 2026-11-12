"""
src/utils.py
============
Shared utility functions used across all modules in the Ghana
groundwater-subsidence project.

Covers:
  - Logging configuration
  - File and directory management
  - Download helpers (progress bar, checksum, retry)
  - Raster and vector I/O shortcuts
  - Unit conversion helpers
  - Date and time utilities
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Logging setup
# ---------------------------------------------------------------------------
def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure root logger with console (and optional file) handlers.

    Call once at the top of each script or notebook:

        from src.utils import setup_logging
        log = setup_logging()

    Parameters
    ----------
    level : int
        Logging level (default logging.INFO).
    log_file : str, Path, or None
        Optional path for a rotating file log.
    fmt, datefmt : str
        Log format strings.

    Returns
    -------
    logging.Logger
        Root logger (use ``logging.getLogger(__name__)`` in sub-modules).
    """
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        root.handlers.clear()

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler (optional)
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        log.info("Logging to file: %s", log_file)

    return root


# ---------------------------------------------------------------------------
# 2. Directory utilities
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_files(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = False,
) -> list[Path]:
    """
    List files in a directory matching a glob pattern.

    Parameters
    ----------
    directory : str or Path
    pattern : str
        Glob pattern (e.g. ``"*.nc"``, ``"GLDAS*.nc4"``).
    recursive : bool
        If True, search subdirectories as well.

    Returns
    -------
    list of Path, sorted.
    """
    p = Path(directory)
    glob_fn = p.rglob if recursive else p.glob
    return sorted(f for f in glob_fn(pattern) if f.is_file())


def file_size_mb(path: str | Path) -> float:
    """Return file size in megabytes."""
    return Path(path).stat().st_size / (1024 ** 2)


def safe_remove(path: str | Path) -> None:
    """Remove a file if it exists; silently ignore if absent."""
    p = Path(path)
    if p.is_file():
        p.unlink()
        log.debug("Removed: %s", p)


# ---------------------------------------------------------------------------
# 3. Download helpers
# ---------------------------------------------------------------------------
def download_file(
    url: str,
    dest: str | Path,
    session: requests.Session | None = None,
    chunk_size: int = 8192,
    timeout: int = 120,
    retries: int = 3,
    retry_delay: float = 5.0,
    show_progress: bool = True,
    expected_md5: str | None = None,
) -> Path:
    """
    Download a file from ``url`` to ``dest`` with progress bar and retry logic.

    Parameters
    ----------
    url : str
        Source URL.
    dest : str or Path
        Destination file path. Parent directory is created if needed.
    session : requests.Session or None
        Existing session (e.g. with authentication headers). If None,
        a new session is created without authentication.
    chunk_size : int
        Streaming chunk size in bytes (default 8 192).
    timeout : int
        Connection and read timeout in seconds (default 120).
    retries : int
        Number of retry attempts on failure (default 3).
    retry_delay : float
        Seconds to wait between retries (default 5.0).
    show_progress : bool
        Display a tqdm progress bar (default True).
    expected_md5 : str or None
        If provided, verify the MD5 checksum of the downloaded file.

    Returns
    -------
    Path
        Path to the downloaded file.

    Raises
    ------
    RuntimeError
        If all retry attempts fail or MD5 checksum mismatches.
    """
    dest = Path(dest)
    ensure_dir(dest.parent)

    _session = session or requests.Session()

    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading (%d/%d): %s", attempt, retries, url)
            resp = _session.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            tmp = dest.with_suffix(dest.suffix + ".tmp")

            with open(tmp, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                disable=not show_progress,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    bar.update(len(chunk))

            # Verify checksum
            if expected_md5:
                actual = _md5(tmp)
                if actual != expected_md5:
                    safe_remove(tmp)
                    raise RuntimeError(
                        f"MD5 mismatch for {dest.name}: "
                        f"expected {expected_md5}, got {actual}"
                    )

            tmp.rename(dest)
            log.info("Saved: %s  (%.1f MB)", dest, file_size_mb(dest))
            return dest

        except (requests.RequestException, RuntimeError) as exc:
            log.warning("Attempt %d failed: %s", attempt, exc)
            safe_remove(dest.with_suffix(dest.suffix + ".tmp"))
            if attempt < retries:
                log.info("Retrying in %.0f seconds ...", retry_delay)
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {retries} attempts."
                ) from exc

    raise RuntimeError("Unreachable")   # pragma: no cover


def _md5(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def already_downloaded(path: str | Path, min_size_mb: float = 0.01) -> bool:
    """
    Return True if a file already exists and is larger than ``min_size_mb``.

    Use this to skip re-downloading files that are already on disk.
    """
    p = Path(path)
    if p.is_file() and file_size_mb(p) >= min_size_mb:
        log.debug("Already downloaded: %s  (%.1f MB)", p, file_size_mb(p))
        return True
    return False


# ---------------------------------------------------------------------------
# 4. Unit conversion
# ---------------------------------------------------------------------------
def cm_ewh_to_mm(value):
    """Convert cm equivalent water height to mm EWH (multiply by 10)."""
    return value * 10.0


def kg_m2_to_mm_ewh(value):
    """
    Convert GLDAS soil moisture (kg/m²) to mm equivalent water height.

    For liquid water: 1 kg/m² = 1 mm EWH (density = 1000 kg/m³).
    """
    return value  # 1:1 for liquid water


def mm_to_cm_ewh(value):
    """Convert mm EWH to cm EWH (divide by 10)."""
    return value / 10.0


# ---------------------------------------------------------------------------
# 5. Date / time utilities
# ---------------------------------------------------------------------------
def month_range(start: str, end: str) -> list[str]:
    """
    Generate a list of monthly date strings (YYYY-MM) from start to end inclusive.

    Parameters
    ----------
    start, end : str
        ISO date strings (e.g. "2002-04" or "2002-04-01").

    Returns
    -------
    list of str in "YYYY-MM" format.

    Example
    -------
    >>> month_range("2002-04", "2002-07")
    ['2002-04', '2002-05', '2002-06', '2002-07']
    """
    import pandas as pd
    idx = pd.date_range(start, end, freq="MS")
    return [d.strftime("%Y-%m") for d in idx]


def parse_year_month(s: str) -> tuple[int, int]:
    """Parse a "YYYY-MM" string into (year, month) tuple."""
    parts = s.split("-")
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# 6. Progress and reporting helpers
# ---------------------------------------------------------------------------
def log_download_summary(
    downloaded: list[Path],
    skipped: list[Path],
    failed: list[str],
) -> None:
    """Log a structured download summary."""
    total_mb = sum(file_size_mb(f) for f in downloaded)
    log.info("=" * 60)
    log.info("Download summary")
    log.info("  Downloaded : %d files  (%.1f MB)", len(downloaded), total_mb)
    log.info("  Skipped    : %d files (already on disk)", len(skipped))
    log.info("  Failed     : %d files", len(failed))
    if failed:
        log.warning("Failed URLs:")
        for url in failed:
            log.warning("  %s", url)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# 7. .netrc helper
# ---------------------------------------------------------------------------
def check_netrc(machine: str = "urs.earthdata.nasa.gov") -> bool:
    """
    Check whether ``~/.netrc`` contains an entry for ``machine``.

    Returns True if credentials are found, False otherwise.
    """
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False
    content = netrc_path.read_text()
    return machine in content


def write_netrc(username: str, password: str,
                machine: str = "urs.earthdata.nasa.gov") -> Path:
    """
    Append (or create) a ~/.netrc entry for ``machine``.

    WARNING: Sets file permissions to 0600 (user read/write only).

    Parameters
    ----------
    username, password : str
        NASA Earthdata credentials.
    machine : str
        Hostname for the netrc entry.

    Returns
    -------
    Path to ~/.netrc
    """
    netrc_path = Path.home() / ".netrc"

    # Remove existing entry for this machine if present
    if netrc_path.exists():
        lines = netrc_path.read_text().splitlines()
        cleaned = []
        skip = False
        for line in lines:
            if f"machine {machine}" in line:
                skip = True
            elif skip and line.strip().startswith("machine"):
                skip = False
            if not skip:
                cleaned.append(line)
        netrc_path.write_text("\n".join(cleaned) + "\n")

    # Append new entry
    with open(netrc_path, "a") as fh:
        fh.write(
            f"\nmachine {machine}\n"
            f"login {username}\n"
            f"password {password}\n"
        )

    # Restrict permissions
    netrc_path.chmod(0o600)
    log.info("~/.netrc updated for machine: %s", machine)
    return netrc_path
