"""TensorBoard launch helpers used by the training entrypoint."""

from __future__ import annotations

import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def launch_tensorboard(logdir: Path, host: str, port: int) -> str:
    """Start TensorBoard for the requested logdir, reusing or avoiding occupied ports when needed."""
    logdir = logdir.expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    existing_port = _find_running_tensorboard_port(logdir)
    if existing_port is not None and _is_port_open(host, existing_port):
        url = _build_url(host, existing_port)
        _print_tensorboard_banner(
            url=url,
            status="already running",
            note=f"Reusing existing TensorBoard for logdir: {logdir}",
        )
        return url

    selected_port = port
    note: str | None = None
    if _is_port_open(host, port):
        selected_port = _find_available_port(host, port + 1)
        note = (
            f"Configured port {port} is already occupied by another process. "
            f"Started this run on port {selected_port}."
        )

    url = _build_url(host, selected_port)
    command = _resolve_tensorboard_command(logdir=logdir, host=host, port=selected_port)
    try:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _print_tensorboard_banner(url=url, status="started", note=note)
    except FileNotFoundError:
        print(
            f"[tensorboard] launch skipped: executable not found. "
            f"Expected TensorBoard for logdir {logdir}"
        )
    return url


def _resolve_tensorboard_command(logdir: Path, host: str, port: int) -> list[str]:
    tensorboard_bin = shutil.which("tensorboard")
    if tensorboard_bin is not None:
        return [
            tensorboard_bin,
            "--logdir",
            str(logdir),
            "--host",
            host,
            "--port",
            str(port),
        ]
    return [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        host,
        "--port",
        str(port),
    ]


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((_connect_host(host), port)) == 0


def _connect_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _build_url(host: str, port: int) -> str:
    display_host = "localhost" if host in {"0.0.0.0", "::"} else host
    return f"http://{display_host}:{port}/"


def _find_available_port(host: str, start_port: int, max_tries: int = 32) -> int:
    for candidate in range(start_port, start_port + max_tries):
        if not _is_port_open(host, candidate):
            return candidate
    raise RuntimeError(f"Could not find a free TensorBoard port starting from {start_port}.")


def _find_running_tensorboard_port(logdir: Path) -> int | None:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "args="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    target_logdir = str(logdir)
    for line in proc.stdout.splitlines():
        if "tensorboard" not in line:
            continue
        try:
            args = shlex.split(line)
        except ValueError:
            continue
        logdir_value = _extract_cli_flag(args, "--logdir")
        port_value = _extract_cli_flag(args, "--port")
        if logdir_value is None or port_value is None:
            continue
        try:
            resolved_logdir = str(Path(logdir_value).expanduser().resolve())
            resolved_port = int(port_value)
        except (OSError, ValueError):
            continue
        if resolved_logdir == target_logdir:
            return resolved_port
    return None


def _extract_cli_flag(args: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for idx, value in enumerate(args):
        if value == flag and idx + 1 < len(args):
            return args[idx + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def _print_tensorboard_banner(url: str, status: str, note: str | None = None) -> None:
    blue = "\033[94m"
    green = "\033[92m"
    yellow = "\033[93m"
    reset = "\033[0m"
    line = f"{blue}{'=' * 72}{reset}"
    status_color = green if status == "started" else yellow
    print(line)
    print(f"{status_color}[TensorBoard {status.upper()}]{reset}")
    if note is not None:
        print(f"{yellow}{note}{reset}")
    print(f"{blue}Open in browser:{reset} {green}{url}{reset}")
    print(line)
