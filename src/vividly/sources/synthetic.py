"""SyntheticDetectionSource: tracker observations from a JSONL fixture, zero footage.

This is how the rules, store and dashboard layers run with no video, no model weights
and no network: a fixture file stands in for detector + pose + tracker, and everything
downstream of the tracker sees exactly what it would see in a real run.

It is deliberately not a FrameSource. It yields no images, because it has none.

Line format, one JSON object per line, in non-decreasing t order:

    {"t": 12.5, "observations": [
        {"track_id": 3, "cx": 100.0, "cy": 240.0, "w": 60.0, "h": 150.0,
         "keypoints": [[x, y, conf], ... 17 of them] or null}
    ]}

t is clip time in seconds and becomes TrackObservation.t for every observation on that
line. keypoints is COCO-17 in COCO order or null when the fixture models a detection
without pose. Blank lines are skipped. Every error names the file and the 1-based line.

Note on POLICY.md, rule 4, no simulated or acted medical events in training: fixtures
written against this format describe positions of abstract tracks. They are not simulated
medical events, and none of this is training or eval data. It exists so a demo can run
without footage.

Import discipline: standard library plus vividly.tracks.state and vividly.sources.base
only. No numpy, no cv2, so linux CI imports this module happily.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from vividly.sources.base import resolve_path
from vividly.tracks.state import TrackObservation

N_KEYPOINTS = 17


class SyntheticDetectionSource:
    """Replay a JSONL fixture as per-frame batches of TrackObservation.

    Not a FrameSource: it has no images, no fps and nothing to seek. The pipeline
    swaps it in for the whole detect + pose + track stack via `detections_from` in the
    pipeline config.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = resolve_path(path)

    def observations(self) -> Iterator[list[TrackObservation]]:
        """Yield one list of TrackObservation per fixture line, in file order.

        A line with an empty observations list yields an empty list: an empty waiting
        room is a real state and the track layer needs to see it.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"synthetic detection fixture not found: {self.path}")
        previous_t: float | None = None
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                record = self._parse_line(line, lineno)
                t = self._require_float(record, "t", lineno)
                if previous_t is not None and t < previous_t:
                    raise ValueError(
                        f"{self.path}:{lineno}: t went backwards ({t} after {previous_t}); "
                        "fixture lines must be in non-decreasing t order"
                    )
                previous_t = t
                yield self._build_batch(record, t, lineno)

    # -- internals ----------------------------------------------------------
    def _parse_line(self, line: str, lineno: int) -> dict[str, Any]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.path}:{lineno}: not valid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"{self.path}:{lineno}: expected a JSON object, got {type(record).__name__}"
            )
        return record

    def _build_batch(self, record: dict[str, Any], t: float, lineno: int) -> list[TrackObservation]:
        if "observations" not in record:
            raise ValueError(f"{self.path}:{lineno}: missing required key 'observations'")
        raw_obs = record["observations"]
        if not isinstance(raw_obs, list):
            raise ValueError(
                f"{self.path}:{lineno}: 'observations' must be a list, got {type(raw_obs).__name__}"
            )
        batch: list[TrackObservation] = []
        for position, item in enumerate(raw_obs):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{self.path}:{lineno}: observation {position} must be an object, "
                    f"got {type(item).__name__}"
                )
            batch.append(
                TrackObservation(
                    track_id=self._require_int(item, "track_id", lineno),
                    t=t,
                    cx=self._require_float(item, "cx", lineno),
                    cy=self._require_float(item, "cy", lineno),
                    w=self._require_float(item, "w", lineno),
                    h=self._require_float(item, "h", lineno),
                    keypoints=self._parse_keypoints(item.get("keypoints"), lineno),
                )
            )
        return batch

    def _parse_keypoints(
        self, raw: Any, lineno: int
    ) -> tuple[tuple[float, float, float], ...] | None:
        if raw is None:
            return None
        if not isinstance(raw, list) or len(raw) != N_KEYPOINTS:
            n = len(raw) if isinstance(raw, list) else type(raw).__name__
            raise ValueError(
                f"{self.path}:{lineno}: keypoints must be null or a list of "
                f"{N_KEYPOINTS} COCO-17 triples, got {n}"
            )
        out: list[tuple[float, float, float]] = []
        for position, triple in enumerate(raw):
            if not isinstance(triple, list | tuple) or len(triple) != 3:
                raise ValueError(f"{self.path}:{lineno}: keypoint {position} must be [x, y, conf]")
            try:
                out.append((float(triple[0]), float(triple[1]), float(triple[2])))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self.path}:{lineno}: keypoint {position} is not numeric"
                ) from exc
        return tuple(out)

    def _require_float(self, record: dict[str, Any], key: str, lineno: int) -> float:
        if key not in record:
            raise ValueError(f"{self.path}:{lineno}: missing required key {key!r}")
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{self.path}:{lineno}: {key!r} must be a number, got {value!r}")
        return float(value)

    def _require_int(self, record: dict[str, Any], key: str, lineno: int) -> int:
        if key not in record:
            raise ValueError(f"{self.path}:{lineno}: missing required key {key!r}")
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{self.path}:{lineno}: {key!r} must be an integer, got {value!r}")
        return int(value)


def write_observations_jsonl(
    path: str | Path,
    batches: Iterable[list[TrackObservation]],
) -> Path:
    """Write per-frame observation batches in the format SyntheticDetectionSource reads.

    One line per batch, t taken from the first observation in the batch. An empty batch
    would have no t of its own, so callers that need empty frames must interleave them
    themselves; this helper skips them rather than inventing a timestamp.
    """
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for batch in batches:
            if not batch:
                continue
            record = {
                "t": batch[0].t,
                "observations": [
                    {
                        "track_id": obs.track_id,
                        "cx": obs.cx,
                        "cy": obs.cy,
                        "w": obs.w,
                        "h": obs.h,
                        "keypoints": (
                            None
                            if obs.keypoints is None
                            else [list(triple) for triple in obs.keypoints]
                        ),
                    }
                    for obs in batch
                ],
            }
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    return out
