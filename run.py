"""Start FastAPI and Streamlit together with one command: python run.py"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _spawn(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(args, cwd=ROOT)


def _wait_for_http(url: str, process: subprocess.Popen, timeout: float = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(0.4)
    return False


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


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
        api_cmd.append("--reload")

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

    api = _spawn(api_cmd)
    if not _wait_for_http(f"http://127.0.0.1:{args.api_port}/docs", api):
        print("FastAPI did not start. Streamlit login will fail until the API is up.")
        print("Check the uvicorn traceback above.")
        _stop(api)
        return 1

    ui = _spawn(ui_cmd)
    processes = (api, ui)

    try:
        while True:
            if any(process.poll() is not None for process in processes):
                break
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nStopping API and Streamlit...")
    finally:
        for process in processes:
            _stop(process)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
