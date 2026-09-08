"""Metric sinks. JSONL is unconditional and local; W&B is optional and offline-first.

POLICY.md, rule 7: no component may assume cloud connectivity at inference time.
W&B must therefore never be a module-level import in this tree; it is imported lazily,
inside WandbSink.__init__.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


class MetricSink(Protocol):
    def log_config(self, cfg: dict) -> None: ...
    def log_row(self, row: dict) -> None: ...
    def log_summary(self, summary: dict) -> None: ...
    def close(self) -> None: ...


class JsonlSink:
    """Always on. One JSON object per line; the file is the record of the run."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a")

    def _w(self, record: str, payload: dict) -> None:
        """`record` tags the line as "config" | "row" | "summary".

        It is deliberately not called "kind": an eval row carries its own `kind` (the
        condition kind, passthrough/photometric/geometric/composite), so `{"kind": tag,
        **payload}` let the payload clobber the tag and every row line came out tagged
        "passthrough". The tag key must be one no payload owns.
        """
        self._fh.write(json.dumps({"record": record, **payload}, cls=_NpEncoder) + "\n")
        self._fh.flush()

    def log_config(self, cfg):
        self._w("config", cfg)

    def log_row(self, row):
        self._w("row", row)

    def log_summary(self, summary):
        self._w("summary", summary)

    def close(self):
        self._fh.close()


class WandbSink:
    """Optional. Defaults to offline; entity comes from the environment, never from code."""

    def __init__(
        self,
        project: str = "vividly",
        run_name: str | None = None,
        mode: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        import wandb  # lazy: never at module import

        os.environ.setdefault("WANDB_MODE", mode or os.environ.get("WANDB_MODE", "offline"))
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            # WANDB_ENTITY, not a VIVIDLY_-prefixed alias: this is the same variable
            # the training half reads, so one line in .env configures both.
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=run_name,
            tags=tags or [],
            reinit=True,
        )

    def log_config(self, cfg):
        self._run.config.update(cfg, allow_val_change=True)

    def log_row(self, row):
        self._run.log(row)

    def log_summary(self, summary):
        self._run.summary.update(summary)

    def close(self):
        self._run.finish()


class MultiSink:
    def __init__(self, *sinks: MetricSink) -> None:
        self._s = list(sinks)

    def log_config(self, cfg):
        [s.log_config(cfg) for s in self._s]

    def log_row(self, row):
        [s.log_row(row) for s in self._s]

    def log_summary(self, s_):
        [s.log_summary(s_) for s in self._s]

    def close(self):
        [s.close() for s in self._s]


def build_sink(cfg: dict, out_dir: Path) -> MetricSink:
    sinks: list[MetricSink] = [JsonlSink(out_dir / "metrics.jsonl")]
    w = cfg.get("logging", {}).get("wandb", {})
    if w.get("enabled", False):
        try:
            sinks.append(
                WandbSink(
                    project=w.get("project", "vividly"),
                    run_name=w.get("run_name"),
                    mode=w.get("mode", "offline"),
                    tags=w.get("tags"),
                )
            )
        except Exception as exc:  # never fail a run because logging failed
            print(f"[eval] W&B sink disabled: {exc}")
    return MultiSink(*sinks)
