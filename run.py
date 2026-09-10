"""Start FastAPI and Streamlit together with one command: python run.py"""

from __future__ import annotations

import argparse
import http.client
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _spawn(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(args, cwd=ROOT)


def _pids_on_port(port: int) -> set[int]:
    if sys.platform != "win32":
        return set()
    result = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: set[int] = set()
    marker = f":{port}"
    for line in result.stdout.splitlines():
        if "LISTENING" not in line or marker not in line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue
    return pids


def _kill_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    subprocess.run(["kill", "-9", str(pid)], check=False)


def _free_port(port: int) -> None:
    """Drop leftover API/UI processes so a new run can bind the port."""
    for pid in _pids_on_port(port):
        print(f"Port {port} is busy (pid {pid}). Stopping it.")
        _kill_pid(pid)
    if _pids_on_port(port):
        time.sleep(0.6)


def _http_ready(host: str, port: int, timeout: float = 0.8) -> bool:
    """True when the worker answers /docs. Avoid urllib — it hangs on Windows."""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("GET", "/docs")
            response = conn.getresponse()
            response.read(128)
            return response.status < 500
        finally:
            conn.close()
    except OSError:
        return False


def _wait_for_api(host: str, port: int, process: subprocess.Popen, timeout: float = 90) -> bool:
    deadline = time.time() + timeout
    next_note = time.time() + 8
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        if _http_ready(host, port):
            return True
        if time.time() >= next_note:
            print(f"Waiting for API at http://{host}:{port}/docs ...")
            next_note = time.time() + 8
        time.sleep(0.3)
    return False


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    _kill_pid(process.pid)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG API and Streamlit UI together.")
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn --reload")
    args = parser.parse_args()

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
    ]
    if not args.no_reload:
        api_cmd.extend(["--reload", "--reload-dir", str(ROOT / "app")])

    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "frontend" / "app.py"),
        "--server.port",
        str(args.ui_port),
        "--server.headless",
        "true",
    ]

    print(f"Starting API  http://localhost:{args.api_port}")
    print(f"Starting UI   http://localhost:{args.ui_port}")
    print("Press Ctrl+C to stop both.\n")

    _free_port(args.api_port)
    _free_port(args.ui_port)

    api: subprocess.Popen | None = None
    ui: subprocess.Popen | None = None
    try:
        api = _spawn(api_cmd)
        if not _wait_for_api("127.0.0.1", args.api_port, api):
            print("FastAPI did not become ready on /docs.")
            print("Check the uvicorn traceback above.")
            return 1

        print("API is ready.")
        ui = _spawn(ui_cmd)
        while True:
            if any(process.poll() is not None for process in (api, ui)):
                break
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nStopping API and Streamlit...")
    finally:
        _stop(ui)
        _stop(api)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
