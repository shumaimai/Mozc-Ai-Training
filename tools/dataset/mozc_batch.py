"""Cross-boundary Mozc N-best batch helper (WSL ↔ Windows).

Mozc must run as a **Windows** process so `mozc.data` is mmap'd on NTFS
(not via WSL `/mnt/c`, which fails mmap). Two launch modes:

- **Mode A** (manual, preferred when interop is flaky): write `keys.txt`,
  print a pasteable PowerShell command; resume by joining `candidates.tsv`.
- **Mode B** (WSL interop): launch `/mnt/c/.../mozc_batch.exe` with all
  file args converted via `wslpath -w`.

Keys are unique readings, hiragana + NFKC normalized. Mozc gets **no context**.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from tools.dataset.normalize import normalize_reading

DEFAULT_ENV_FILE = Path("config/mozc_batch.env")
ENV_EXE = "MOZC_BATCH_EXE"
ENV_DATA = "MOZC_ENGINE_DATA_PATH"
ENV_MAX = "MOZC_MAX_CANDIDATES"

ModeName = Literal["auto", "a", "b", "join"]


class MozcBatchModeARequired(RuntimeError):
    """Raised when Mode B fails / is skipped and the caller must run Mode A."""

    def __init__(self, message: str, *, powershell: str, keys_path: Path, candidates_path: Path):
        super().__init__(message)
        self.powershell = powershell
        self.keys_path = keys_path
        self.candidates_path = candidates_path


def normalize_mozc_key(reading: str) -> str:
    """Hiragana + NFKC (+ strip spaces/middots) — Mozc input form."""
    return normalize_reading(reading or "")


def unique_keys(readings: Iterable[str], *, normalize: bool = True) -> list[str]:
    """Unique conversion keys in first-seen order."""
    seen: dict[str, None] = {}
    for raw in readings:
        key = normalize_mozc_key(raw) if normalize else (raw or "")
        if key and key not in seen:
            seen[key] = None
    return list(seen.keys())


def readings_from_records(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Unique conversion keys (readings) in first-seen order."""
    return unique_keys(
        (row.get("record", row).get("reading", "") for row in rows),
        normalize=True,
    )


def write_keys_txt(keys: Iterable[str], path: Path, *, normalize: bool = True) -> list[str]:
    """Write one reading per line; return the keys written."""
    ordered = unique_keys(keys, normalize=normalize)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")
    return ordered


def parse_candidates_tsv(lines: Iterable[str]) -> dict[str, list[str]]:
    """Parse mozc_batch output ("key\\tcand1\\tcand2..." per line)."""
    result: dict[str, list[str]] = {}
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        key, *candidates = line.split("\t")
        # Keys in TSV should already match normalized form; normalize on read
        # so join still works if a caller wrote raw readings.
        result[normalize_mozc_key(key) or key] = candidates
    return result


def load_candidates_map(candidates_path: Path) -> dict[str, list[str]]:
    return parse_candidates_tsv(Path(candidates_path).read_text(encoding="utf-8").splitlines())


def _context_of(record: dict[str, Any]) -> list[str]:
    context = record.get("metadata", {}).get("context")
    if not context:
        return []
    return context if isinstance(context, list) else [context]


def merge(
    rows: Iterable[dict[str, Any]],
    key_to_candidates: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Join Mozc candidates onto records, producing classify-ready rows."""
    merged: list[dict[str, Any]] = []
    for row in rows:
        record = row.get("record", row)
        reading = normalize_mozc_key(record.get("reading", ""))
        merged.append(
            {
                "record": record,
                "candidates": key_to_candidates.get(reading, []),
                "context": row.get("context") or _context_of(record),
            }
        )
    return merged


def join_reading_candidates(
    rows: Iterable[dict[str, Any]],
    key_to_candidates: dict[str, list[str]],
    *,
    reading_field: str = "reading",
) -> list[list[str]]:
    """Return candidates list per row (reading → N-best join)."""
    out: list[list[str]] = []
    for row in rows:
        record = row.get("record", row) if isinstance(row, dict) else {}
        raw = row.get(reading_field) if isinstance(row, dict) and reading_field in row else None
        if raw is None:
            raw = record.get(reading_field, "") if isinstance(record, dict) else ""
        key = normalize_mozc_key(str(raw or ""))
        out.append(list(key_to_candidates.get(key, [])))
    return out


def load_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines (no secrets required; local path config)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def running_under_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower().find("microsoft") >= 0
    except OSError:
        return False


def windows_path_to_wsl(path: str | Path) -> Path:
    """C:\\foo\\bar → /mnt/c/foo/bar (no-op if already POSIX / existing)."""
    p = Path(path)
    s = str(path)
    if p.is_file() or p.is_dir():
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return p


def path_exists_cross_platform(path: str | Path) -> bool:
    p = Path(path)
    if p.is_file():
        return True
    if running_under_wsl():
        return windows_path_to_wsl(path).is_file()
    return False


def _wslpath_w(path: Path | str) -> str:
    """Convert a path to Windows form via `wslpath -w` (required for Mode B args)."""
    p = Path(path)
    s = str(p)
    # Already Windows?
    if re.match(r"^[A-Za-z]:[\\/]", s):
        return s.replace("/", "\\")
    # Prefer wslpath when available (handles /home/... and /mnt/c/...).
    wslpath = shutil.which("wslpath")
    if wslpath:
        completed = subprocess.run(
            [wslpath, "-w", s],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            out = (completed.stdout or "").strip()
            if out:
                return out
    # Fallback: /mnt/c/foo → C:\foo
    m = re.match(r"^/mnt/([a-z])/(.*)$", s.replace("\\", "/"))
    if m:
        return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\")
    raise RuntimeError(f"cannot convert to Windows path (install wslpath?): {path}")


def format_mode_a_powershell(
    *,
    exe_win: str,
    engine_win: str,
    keys_win: str,
    candidates_win: str,
    max_candidates: int,
) -> str:
    """Pasteable PowerShell one-liner (Mode A)."""
    def q(p: str) -> str:
        return "'" + p.replace("'", "''") + "'"

    return (
        f"& {q(exe_win)} "
        f"--engine_data_path={q(engine_win)} "
        f"--input={q(keys_win)} "
        f"--output={q(candidates_win)} "
        f"--max_candidates={max_candidates}"
    )


def resolve_batch_config(
    env_file: Path | None = None,
    exe: str | None = None,
    engine_data: str | None = None,
    max_candidates: int | None = None,
) -> tuple[Path, Path, int]:
    """Resolve mozc_batch paths from CLI overrides, process env, then env file.

    Returns (exe_path, engine_data_path, max_candidates) using the configured
    path strings as-is (typically Windows paths in config/mozc_batch.env).
    Existence is checked via WSL /mnt/<drive> remapping when needed.
    """
    file_values = load_env_file(env_file or DEFAULT_ENV_FILE)
    resolved_exe = exe or os.environ.get(ENV_EXE) or file_values.get(ENV_EXE, "")
    resolved_data = (
        engine_data or os.environ.get(ENV_DATA) or file_values.get(ENV_DATA, "")
    )
    raw_max = (
        str(max_candidates)
        if max_candidates is not None
        else os.environ.get(ENV_MAX) or file_values.get(ENV_MAX, "50")
    )
    if not resolved_exe:
        raise ValueError(
            f"{ENV_EXE} is unset; set it or create {DEFAULT_ENV_FILE} "
            f"(see config/mozc_batch.env.example)"
        )
    if not resolved_data:
        raise ValueError(
            f"{ENV_DATA} is unset; set it or create {DEFAULT_ENV_FILE} "
            f"(see config/mozc_batch.env.example)"
        )
    exe_path = Path(resolved_exe)
    data_path = Path(resolved_data)
    if not path_exists_cross_platform(exe_path):
        raise FileNotFoundError(f"mozc_batch binary not found: {exe_path}")
    if not path_exists_cross_platform(data_path):
        raise FileNotFoundError(f"engine data not found: {data_path}")
    return exe_path, data_path, int(raw_max)


def mode_a_instructions(
    *,
    exe: Path | str,
    engine_data: Path | str,
    keys_path: Path,
    candidates_path: Path,
    max_candidates: int,
) -> tuple[str, str]:
    """Return (powershell_command, human_message) for Mode A."""
    keys_path = Path(keys_path).resolve()
    candidates_path = Path(candidates_path).resolve()
    if running_under_wsl():
        exe_win = str(exe) if re.match(r"^[A-Za-z]:[\\/]", str(exe)) else _wslpath_w(windows_path_to_wsl(exe))
        engine_win = (
            str(engine_data)
            if re.match(r"^[A-Za-z]:[\\/]", str(engine_data))
            else _wslpath_w(windows_path_to_wsl(engine_data))
        )
        keys_win = _wslpath_w(keys_path)
        cands_win = _wslpath_w(candidates_path)
    else:
        exe_win = str(exe)
        engine_win = str(engine_data)
        keys_win = str(keys_path)
        cands_win = str(candidates_path)

    ps = format_mode_a_powershell(
        exe_win=exe_win,
        engine_win=engine_win,
        keys_win=keys_win,
        candidates_win=cands_win,
        max_candidates=max_candidates,
    )
    msg = (
        "MODE A (Windows PowerShell): keys written; run mozc_batch on Windows, then resume join.\n"
        f"  keys:        {keys_path}\n"
        f"  candidates:  {candidates_path}\n"
        f"  PowerShell:\n{ps}\n"
        "After candidates.tsv exists, re-run with mode=join (or auto)."
    )
    return ps, msg


def run_mozc_batch_mode_b(
    exe: Path | str,
    engine_data: Path | str,
    keys_path: Path,
    candidates_path: Path,
    max_candidates: int,
) -> None:
    """Mode B: WSL interop — launch Linux-visible PE path; Windows paths for file args."""
    if not running_under_wsl():
        raise RuntimeError("Mode B requires WSL")

    exe_launch = windows_path_to_wsl(exe)
    if not exe_launch.is_file():
        raise FileNotFoundError(f"mozc_batch exe not visible under WSL: {exe} → {exe_launch}")

    keys_path = Path(keys_path).resolve()
    candidates_path = Path(candidates_path).resolve()
    candidates_path.parent.mkdir(parents=True, exist_ok=True)

    # Engine path: keep configured Windows path when present; else wslpath -w.
    engine_s = str(engine_data)
    if re.match(r"^[A-Za-z]:[\\/]", engine_s):
        engine_arg = engine_s.replace("/", "\\")
    else:
        engine_arg = _wslpath_w(windows_path_to_wsl(engine_data))

    keys_arg = _wslpath_w(keys_path)
    out_arg = _wslpath_w(candidates_path)

    command = [
        str(exe_launch),
        f"--engine_data_path={engine_arg}",
        f"--input={keys_arg}",
        f"--output={out_arg}",
        f"--max_candidates={max_candidates}",
    ]
    print(
        f"mozc_batch Mode B keys_in={keys_path} exe={exe_launch} "
        f"engine={engine_arg} out={out_arg} max={max_candidates}",
        flush=True,
    )
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"mozc_batch Mode B failed rc={completed.returncode}: {' '.join(command)}"
        )
    if not candidates_path.is_file():
        raise RuntimeError(f"mozc_batch Mode B produced no output: {candidates_path}")


def run_mozc_batch_native_windows(
    exe: Path | str,
    engine_data: Path | str,
    keys_path: Path,
    candidates_path: Path,
    max_candidates: int,
) -> None:
    """Native Windows (or already-Windows host): all paths as given."""
    candidates_path = Path(candidates_path)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(exe),
        f"--engine_data_path={engine_data}",
        f"--input={keys_path}",
        f"--output={candidates_path}",
        f"--max_candidates={max_candidates}",
    ]
    print(f"mozc_batch native: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"mozc_batch failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def run_mozc_batch_boundary(
    exe: Path | str,
    engine_data: Path | str,
    keys_path: Path,
    candidates_path: Path,
    max_candidates: int,
    *,
    mode: ModeName = "auto",
    allow_existing_candidates: bool = True,
) -> str:
    """Run mozc_batch across the WSL/Windows boundary.

    Returns the mode used: \"a\" (join-only after existing TSV), \"b\", or \"native\".

    Mode A does not execute the exe; it writes instructions and raises
    MozcBatchModeARequired unless candidates.tsv already exists (resume/join).
    """
    keys_path = Path(keys_path)
    candidates_path = Path(candidates_path)
    if not keys_path.is_file():
        raise FileNotFoundError(f"keys.txt missing: {keys_path}")

    if mode == "join" or (
        allow_existing_candidates
        and mode in ("auto", "a")
        and candidates_path.is_file()
        and candidates_path.stat().st_size > 0
        and candidates_path.stat().st_mtime >= keys_path.stat().st_mtime
    ):
        if not candidates_path.is_file():
            raise FileNotFoundError(f"candidates.tsv missing for join: {candidates_path}")
        print(f"mozc_batch join: using existing {candidates_path}", flush=True)
        return "a"

    if mode == "a":
        ps, msg = mode_a_instructions(
            exe=exe,
            engine_data=engine_data,
            keys_path=keys_path,
            candidates_path=candidates_path,
            max_candidates=max_candidates,
        )
        print(msg, flush=True)
        raise MozcBatchModeARequired(
            msg, powershell=ps, keys_path=keys_path, candidates_path=candidates_path
        )

    if not running_under_wsl():
        run_mozc_batch_native_windows(exe, engine_data, keys_path, candidates_path, max_candidates)
        return "native"

    # Mode B (or auto under WSL)
    if mode in ("auto", "b"):
        try:
            run_mozc_batch_mode_b(exe, engine_data, keys_path, candidates_path, max_candidates)
            return "b"
        except Exception as exc:  # noqa: BLE001 — fall back to Mode A instructions
            if mode == "b":
                raise
            ps, msg = mode_a_instructions(
                exe=exe,
                engine_data=engine_data,
                keys_path=keys_path,
                candidates_path=candidates_path,
                max_candidates=max_candidates,
            )
            print(f"Mode B failed ({exc}); falling back to Mode A.", flush=True)
            print(msg, flush=True)
            raise MozcBatchModeARequired(
                msg, powershell=ps, keys_path=keys_path, candidates_path=candidates_path
            ) from exc

    raise ValueError(f"unknown mode: {mode}")


def run_mozc_batch(
    exe: Path,
    engine_data: Path,
    keys_path: Path,
    candidates_path: Path,
    max_candidates: int,
) -> None:
    """Backward-compatible entry: auto Mode B under WSL, native on Windows."""
    run_mozc_batch_boundary(
        exe,
        engine_data,
        keys_path,
        candidates_path,
        max_candidates,
        mode="auto",
        allow_existing_candidates=False,
    )


def emit_mode_a_and_exit_if_needed(
    *,
    exe: Path | str,
    engine_data: Path | str,
    keys_path: Path,
    candidates_path: Path,
    max_candidates: int,
) -> None:
    """Helper for CLIs that want Mode A printed then sys.exit(2)."""
    ps, msg = mode_a_instructions(
        exe=exe,
        engine_data=engine_data,
        keys_path=keys_path,
        candidates_path=candidates_path,
        max_candidates=max_candidates,
    )
    print(msg, file=sys.stderr)
    print(ps)
    raise SystemExit(2)
