"""The live appliance, unchanged, with the camera replaced by a video file played in real time.

    uv run python scripts/live_from_file.py --start 84 \\
        --url takes/20260909-203345-take1.mp4 --config configs/eval/live_final.yaml \\
        --det-hz 8 --score 0.5 --hud demo --port 8089 --fall-instant

Every other argument is live_stream.py's own. Only the capture changes: the camera reader
never sleeps (a live camera paces itself), so a file would play at decode speed and the
8 Hz detector would sample it a handful of times. This reader sleeps to the file's frame
rate instead, so the detector thread, the rescue, the pose thread, the analyzers, the
compositor and the MJPEG server all run exactly as they do on the rig, in real time. It is
how the live path was tested on 2026-09-10 while the camera was unplugged.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _appliance():
    spec = importlib.util.spec_from_file_location("live_stream", REPO / "scripts" / "live_stream.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    argv = sys.argv[1:]
    start = 0.0
    if "--start" in argv:
        i = argv.index("--start")
        start = float(argv[i + 1])
        del argv[i : i + 2]
    ls = _appliance()

    class PacedFile(ls.LowLatencyCapture):
        def open(self) -> None:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                raise RuntimeError(f"could not open {self.url}")
            self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or self.fps_fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * self.fps))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.cap = cap
            self._t0 = time.perf_counter()
            self._n = 0

        def read(self):
            wait = self._t0 + self._n / self.fps - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            ok, img = self.cap.read()
            self._n += 1
            return (img if ok else None), time.time(), 0

    ls.LowLatencyCapture = PacedFile
    sys.argv = ["live_stream.py", *argv]
    return ls.main()


if __name__ == "__main__":
    sys.exit(main())
