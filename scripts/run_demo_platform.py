#!/usr/bin/env python
"""Run the Vividly demo platform: upload a video, pick a model of record, watch the
harness. One command, local only:

    uv run python scripts/run_demo_platform.py            # opens http://127.0.0.1:8700

Uploads and run artifacts live under outputs/demo_uploads/ and outputs/demo_runs/
(gitignored; demo material only, never training data). The model picker rescans the
final_models/ lanes and the dropped-in .onnx uploads under outputs/demo_uploads/models/
on every fetch, so a sidecar written during a training day appears without a restart.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default loopback)")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--max-upload-mb", type=int, default=2048)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    import uvicorn

    from vividly.demo.app import create_demo_app
    from vividly.demo.jobs import JobManager

    manager = JobManager(REPO_ROOT)
    app = create_demo_app(manager, max_upload_mb=args.max_upload_mb)
    url = f"http://{args.host}:{args.port}"
    print(f"[demo] Vividly Model Demo at {url}  (ctrl-c to stop)")
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        manager.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
