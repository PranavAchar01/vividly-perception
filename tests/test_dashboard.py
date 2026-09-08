"""Dashboard and pipeline-runner tests.

Three rules hold for every test in this file.

  * No models. Nothing here downloads a weight, opens an ONNX session or decodes a video.
    The zero-footage flow is the whole point of the synthetic fixture: anything that needs
    real detector weights is not a test, it is an acceptance run against real weights.
  * No network, no server. The FastAPI app is driven in process through its ASGI
    interface; uvicorn is never started and no socket is ever bound.
  * Deterministic. The committed scene fixture is generated in closed form with no RNG and
    no clock, and test_committed_scene_fixture_is_deterministic regenerates it and compares
    bytes, so it cannot drift under anyone's feet.

Why not starlette's TestClient. TestClient is the natural tool and it is what an acceptance
run should use, but importing it imports httpx unconditionally, and httpx is not installed
in this repo's default fork (it reaches uv.lock only through the optional `curate`, `cuda`
and `eval-agpl` extras, none of which is in the default fork). Skipping the endpoint tests
on the machine that has to prove them would be worse than driving the same ASGI callable
TestClient drives, which is what AsgiClient below does: one asyncio loop, no listening
socket, the identical code path through starlette's routing, middleware and response
rendering. Adding `httpx` to the dev dependency group would let this file use TestClient
directly.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from vividly.dashboard import MOCK_LABEL, STAGES, PipelineState, create_app, mock_mrn
from vividly.events import EventStore
from vividly.rules.fixtures import collapse_and_stay_down, normal_fidget
from vividly.schemas.events import AlertRecord, make_alert_id, utc_now_iso
from vividly.sources import SyntheticDetectionSource, write_observations_jsonl
from vividly.tracks import TrackBook, TrackConfig, TrackObservation

REPO = Path(__file__).resolve().parent.parent
DASHBOARD_SRC = REPO / "src" / "vividly" / "dashboard"
RUN_PIPELINE = REPO / "scripts" / "run_pipeline.py"
SCENE_PATH = REPO / "tests" / "fixtures" / "pipeline_synth" / "scene_60s.jsonl"
SYNTHETIC_CONFIG = REPO / "configs" / "pipeline" / "synthetic_demo.yaml"
REPLAY_CONFIG = REPO / "configs" / "pipeline" / "replay_demo.yaml"

# The socket payload keys the dashboard contract fixes. Extra keys are display context and
# may grow; these five may not go away.
CONTRACT_WS_KEYS = frozenset({"frame_b64", "tracks", "alerts", "timers", "latency_ms"})


# ------------------------------------------------------------- the scene fixture
# 60 simulated seconds, 5 observation batches per simulated second, three abstract tracks.
# 5 Hz rather than the 15 fps the rest of the repo assumes is a file-size decision and
# nothing else: at 15 Hz this fixture is a megabyte of keypoints, and nothing downstream
# can tell the difference because every rule window is expressed in seconds.
#
# POLICY.md, rule 4: this is not a simulated medical event and it is not training or evaluation
# data. It is three points moving on a plane, one of them along a fall-shaped curve,
# generated so the acute-event cascade, the store and the page have something to run on with no
# footage in existence. The unit-fixture boundary is recorded for counsel in CQ-21.
SCENE_HZ = 5.0
SCENE_SECONDS = 60.0
SCENE_LEAD_IN_S = 20.0
SCENE_FALL_S = 1.5
FALLEN_TRACK_ID = 1
FIDGET_TRACK_ID = 2
STAFF_TRACK_ID = 99
SECOND_SEAT = (1300.0, 600.0)

# Pixels are rounded to a thousandth before the fixture is written. Seventeen significant
# figures of a sinusoid is noise nobody reads and it doubles the size of a committed file;
# a thousandth of a pixel is four orders of magnitude finer than any threshold in the repo.
SCENE_DECIMALS = 3


def _rounded(obs: TrackObservation) -> TrackObservation:
    """One observation with every float rounded to SCENE_DECIMALS. Purely a file-size move."""
    return TrackObservation(
        track_id=obs.track_id,
        t=round(obs.t, SCENE_DECIMALS),
        cx=round(obs.cx, SCENE_DECIMALS),
        cy=round(obs.cy, SCENE_DECIMALS),
        w=round(obs.w, SCENE_DECIMALS),
        h=round(obs.h, SCENE_DECIMALS),
        keypoints=(
            None
            if obs.keypoints is None
            else tuple(
                (round(x, SCENE_DECIMALS), round(y, SCENE_DECIMALS), round(conf, SCENE_DECIMALS))
                for x, y, conf in obs.keypoints
            )
        ),
    )


def build_scene() -> list[list[TrackObservation]]:
    """The committed scene, in memory. Closed form: same bytes on every machine, forever.

    Track 1 fidgets for SCENE_LEAD_IN_S, falls in SCENE_FALL_S, and stays down and still
    for the rest of the minute, the only thing in the scene the acute-event cascade reacts to,
    and at the shipped thresholds (a 30 s confirmation fits a 60 s fixture natively).
    Track 2 fidgets in a second seat the whole time (the negative control: nothing may
    fire on it). Track 99 walks the patrol from vividly.rules.fixtures, so the room has a
    third, always-moving body that nothing may fire on either.
    """
    fallen = collapse_and_stay_down(
        lead_in_minutes=SCENE_LEAD_IN_S / 60.0,
        down_minutes=(SCENE_SECONDS - SCENE_LEAD_IN_S - SCENE_FALL_S) / 60.0,
        fall_s=SCENE_FALL_S,
        track_id=FALLEN_TRACK_ID,
        staff_track_id=STAFF_TRACK_ID,
        hz=SCENE_HZ,
    )
    fidget = normal_fidget(
        minutes=SCENE_SECONDS / 60.0,
        track_id=FIDGET_TRACK_ID,
        staff_track_id=None,
        hz=SCENE_HZ,
        seat=SECOND_SEAT,
    )
    return [
        [_rounded(obs) for obs in sorted(a + b, key=lambda obs: obs.track_id)]
        for a, b in zip(fallen, fidget, strict=True)
    ]


# ----------------------------------------------------------------- ASGI driving
@dataclass(frozen=True)
class Response:
    status: int
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.body)


class AsgiClient:
    """Drive a FastAPI app in process: no server, no socket, no httpx (see module docstring).

    One asyncio loop per call, which keeps every test independent and means a hung endpoint
    fails as a timeout rather than by wedging the suite.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    # -- HTTP ---------------------------------------------------------------
    def get(self, path: str) -> Response:
        return asyncio.run(self._request("GET", path, b""))

    def post(self, path: str, payload: Any = None) -> Response:
        body = json.dumps({} if payload is None else payload).encode("utf-8")
        return asyncio.run(self._request("POST", path, body))

    async def _request(self, method: str, path: str, body: bytes) -> Response:
        scope = _http_scope(method, path, len(body))
        pending = [{"type": "http.request", "body": body, "more_body": False}]
        received: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return pending.pop(0) if pending else {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            received.append(message)

        await asyncio.wait_for(self.app(scope, receive, send), timeout=10)
        status = next(m["status"] for m in received if m["type"] == "http.response.start")
        payload = b"".join(
            m.get("body", b"") for m in received if m["type"] == "http.response.body"
        )
        return Response(status=status, body=payload)

    # -- WebSocket ----------------------------------------------------------
    def stream(self, path: str, n_payloads: int = 1) -> list[dict[str, Any]]:
        """Connect, collect n_payloads pushes, disconnect. Never blocks forever."""
        return asyncio.run(self._stream(path, n_payloads))

    async def _stream(self, path: str, n_payloads: int) -> list[dict[str, Any]]:
        to_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        from_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        await to_app.put({"type": "websocket.connect"})

        async def receive() -> dict[str, Any]:
            return await to_app.get()

        async def send(message: dict[str, Any]) -> None:
            await from_app.put(message)

        task = asyncio.create_task(self.app(_ws_scope(path), receive, send))
        try:
            accept = await asyncio.wait_for(from_app.get(), timeout=10)
            assert accept["type"] == "websocket.accept", accept
            payloads = []
            for _ in range(n_payloads):
                message = await asyncio.wait_for(from_app.get(), timeout=10)
                assert message["type"] == "websocket.send", message
                payloads.append(json.loads(message["text"]))
            await to_app.put({"type": "websocket.disconnect", "code": 1000})
            await asyncio.wait_for(task, timeout=10)
            return payloads
        finally:
            task.cancel()


def _http_scope(method: str, path: str, content_length: int) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(content_length).encode("ascii")),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }


def _ws_scope(path: str) -> dict[str, Any]:
    return {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "subprotocols": [],
    }


# ------------------------------------------------------------------- fixtures
def load_runner() -> Any:
    """Import scripts/run_pipeline.py as a module. It is a script, not a package.

    The module goes into sys.modules before it is executed: a frozen dataclass resolves its
    own module out of sys.modules while the decorator is running, and a module that is not
    there yet fails with an unhelpful AttributeError from inside dataclasses.
    """
    name = "vividly_run_pipeline_script"
    spec = importlib.util.spec_from_file_location(name, RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


@pytest.fixture
def state(tmp_path: Path) -> Iterator[PipelineState]:
    """A live PipelineState over a real TrackBook and a real on-disk EventStore.

    Real store, not a double: "feedback reaches sqlite" is the claim under test, and a fake
    store would prove the endpoint calls a method rather than that a verdict survives.
    """
    book = TrackBook(TrackConfig(baseline_window_s=15.0, ring_size=1000))
    store = EventStore(tmp_path / "alerts.sqlite")
    pipeline_state = PipelineState(
        book=book,
        store=store,
        run_name="test",
        mode="synthetic",
        source_label=str(SCENE_PATH),
        stream_hz=50.0,
    )
    yield pipeline_state
    store.close()


@pytest.fixture
def client(state: PipelineState) -> AsgiClient:
    return AsgiClient(create_app(state))


def ingest_scene(book: TrackBook, n_batches: int) -> None:
    """Feed the first n_batches of the committed fixture through the synthetic source."""
    source = SyntheticDetectionSource(SCENE_PATH)
    for index, batch in enumerate(source.observations()):
        if index >= n_batches:
            return
        book.ingest(batch)


def stored_alert(store: EventStore, track_id: int = FALLEN_TRACK_ID) -> AlertRecord:
    record = AlertRecord(
        alert_id=make_alert_id("fall_then_still_v0", track_id, 20.0),
        track_id=track_id,
        rule_id="fall_then_still_v0",
        rule_version="0.1.0",
        onset_ts=20.0,
        detected_ts=50.0,
        created_utc=utc_now_iso(),
        baseline=None,
        details={"down_s": 30.0},
    )
    store.insert_alert(record)
    return record


# ------------------------------------------------------------------ the page
def test_index_returns_200_and_labels_the_mock_binding(client: AsgiClient) -> None:
    response = client.get("/")
    assert response.status == 200
    assert MOCK_LABEL in response.text
    # The label has to reach the binding affordance itself, not merely appear somewhere on
    # the page: an unlabelled fake MRN is the one thing here that could read as an
    # integration. The page builds that button text from MOCK_LABEL_TEXT.
    assert "MOCK_LABEL_TEXT" in response.text
    assert f'"{MOCK_LABEL}"' in response.text


def test_index_loads_no_external_asset(client: AsgiClient) -> None:
    """POLICY.md, rule 7: this page has to render with the network cable out."""
    page = client.get("/").text
    for forbidden in ('src="http', 'href="http', "cdn.", "googleapis", "unpkg", "jsdelivr"):
        assert forbidden not in page, f"page reaches for an external asset: {forbidden}"


def test_index_carries_every_reviewer_affordance(client: AsgiClient) -> None:
    """The four things a reviewer must be able to do, plus the timers, all on one page."""
    page = client.get("/").text
    for needle in (
        "Agree",
        "False alarm",
        "/feedback",
        "agree",
        "false_alarm",
        "/staff",
        "/chart-bind",
        "/discharged",
        "/departures/",
        "timers",
        "latency",
    ):
        assert needle in page, f"the page is missing {needle!r}"


def test_docs_routes_are_off(client: AsgiClient) -> None:
    """Swagger and ReDoc load their JavaScript from a CDN, so both are disabled."""
    assert client.get("/docs").status == 404
    assert client.get("/redoc").status == 404


# ------------------------------------------------------------------- feedback
def test_feedback_post_writes_through_to_sqlite(state: PipelineState, client: AsgiClient) -> None:
    record = stored_alert(state.store)
    response = client.post(f"/alerts/{record.alert_id}/feedback", {"label": "agree"})
    assert response.status == 200
    assert response.json()["stored"] is True

    reopened = EventStore(state.store.path)
    try:
        stored = reopened.get(record.alert_id)
    finally:
        reopened.close()
    assert stored is not None
    assert stored.feedback == "agree"


def test_feedback_post_accepts_false_alarm(state: PipelineState, client: AsgiClient) -> None:
    record = stored_alert(state.store)
    assert (
        client.post(f"/alerts/{record.alert_id}/feedback", {"label": "false_alarm"}).status == 200
    )
    assert state.store.get(record.alert_id).feedback == "false_alarm"


def test_feedback_rejects_an_unknown_label(state: PipelineState, client: AsgiClient) -> None:
    """A free-text label would poison every later agreement count, so it is a 400."""
    record = stored_alert(state.store)
    assert client.post(f"/alerts/{record.alert_id}/feedback", {"label": "maybe"}).status == 400
    assert state.store.get(record.alert_id).feedback is None


def test_feedback_on_an_unknown_alert_is_a_404(client: AsgiClient) -> None:
    assert client.post("/alerts/nope-1-0/feedback", {"label": "agree"}).status == 404


# ---------------------------------------------------------------- staff tagging
def test_staff_post_flips_trackbook_state(state: PipelineState, client: AsgiClient) -> None:
    ingest_scene(state.book, 5)
    assert state.book.tracks()[FALLEN_TRACK_ID].is_staff is False

    assert client.post(f"/tracks/{STAFF_TRACK_ID}/staff", {"staff": True}).status == 200
    assert state.book.tracks()[STAFF_TRACK_ID].is_staff is True

    assert client.post(f"/tracks/{STAFF_TRACK_ID}/staff", {"staff": False}).status == 200
    assert state.book.tracks()[STAFF_TRACK_ID].is_staff is False


def test_staff_post_rejects_a_body_without_a_boolean(
    state: PipelineState, client: AsgiClient
) -> None:
    ingest_scene(state.book, 5)
    assert client.post(f"/tracks/{STAFF_TRACK_ID}/staff", {}).status == 400
    assert client.post(f"/tracks/{STAFF_TRACK_ID}/staff", {"staff": "yes"}).status == 400
    assert state.book.tracks()[STAFF_TRACK_ID].is_staff is False


def test_staff_post_on_an_unknown_track_is_a_404(client: AsgiClient) -> None:
    assert client.post("/tracks/4242/staff", {"staff": True}).status == 404


# ------------------------------------------------------------- mock chart bind
def test_chart_bind_binds_a_labelled_mock_mrn(state: PipelineState, client: AsgiClient) -> None:
    ingest_scene(state.book, 5)
    response = client.post(f"/tracks/{FALLEN_TRACK_ID}/chart-bind", {})
    assert response.status == 200
    body = response.json()
    assert body["mock"] is True
    assert body["label"] == MOCK_LABEL
    assert body["mrn_binding"] == mock_mrn(FALLEN_TRACK_ID) == "MRN-0001"
    assert state.book.tracks()[FALLEN_TRACK_ID].mrn_binding == "MRN-0001"


def test_chart_bind_ignores_whatever_is_in_the_body(
    state: PipelineState, client: AsgiClient
) -> None:
    """The id is derived from the tracker id so a demo cannot type a real MRN into it."""
    ingest_scene(state.book, 5)
    client.post(f"/tracks/{FALLEN_TRACK_ID}/chart-bind", {"mrn": "1234567"})
    assert state.book.tracks()[FALLEN_TRACK_ID].mrn_binding == "MRN-0001"


def test_mock_mrn_is_deterministic_and_zero_padded() -> None:
    assert mock_mrn(1) == "MRN-0001"
    assert mock_mrn(99) == "MRN-0099"
    assert mock_mrn(12345) == "MRN-12345"


# ---------------------------------------------------------------- the WS stream
def test_ws_streams_a_frame_dict_with_the_contract_keys(
    state: PipelineState, client: AsgiClient
) -> None:
    ingest_scene(state.book, 25)
    state.publish(
        frame_b64=None,
        frame_index=24,
        t_source=4.8,
        latency_ms={"source": 0.2, "rules": 1.5},
        latency_p50_ms={"source": 0.19},
        latency_p95_ms={"source": 0.4},
    )

    payload = client.stream("/ws", n_payloads=1)[0]
    assert set(payload) >= CONTRACT_WS_KEYS
    assert payload["frame_b64"] is None
    assert {track["track_id"] for track in payload["tracks"]} == {
        FALLEN_TRACK_ID,
        FIDGET_TRACK_ID,
        STAFF_TRACK_ID,
    }
    assert set(payload["timers"]) == {
        str(FALLEN_TRACK_ID),
        str(FIDGET_TRACK_ID),
        str(STAFF_TRACK_ID),
    }
    assert payload["timers"][str(FALLEN_TRACK_ID)] == pytest.approx(4.8, abs=0.01)
    assert payload["latency_ms"]["rules"] == pytest.approx(1.5)
    assert payload["mock_label"] == MOCK_LABEL
    assert payload["stages"] == list(STAGES)


def test_ws_payload_carries_stored_alerts(state: PipelineState, client: AsgiClient) -> None:
    ingest_scene(state.book, 5)
    record = stored_alert(state.store)
    payload = client.stream("/ws", n_payloads=1)[0]
    assert [alert["alert_id"] for alert in payload["alerts"]] == [record.alert_id]


def test_ws_payload_json_never_carries_nan(state: PipelineState) -> None:
    """A nan clock is the normal state of a book that has seen nothing, and JSON.parse
    rejects the literal Python emits for it, so the page would break rather than the
    number. Serialising with allow_nan=False is the pin."""
    payload = state.snapshot()
    json.dumps(payload, allow_nan=False)


def test_alert_details_floats_are_scrubbed_in_the_snapshot(state: PipelineState) -> None:
    """Rule-authored details and baseline dicts reach the socket unchanged from the store,
    so one Infinity from a future rule would make JSON.parse reject the whole payload and
    blank the page. The boundary scrubs nested floats to None instead of trusting rules."""
    state.store.insert_alert(
        AlertRecord(
            alert_id=make_alert_id("seizure_candidate_v0", 9, 5.0),
            track_id=9,
            rule_id="seizure_candidate_v0",
            rule_version="0.1.0",
            onset_ts=5.0,
            detected_ts=6.0,
            created_utc=utc_now_iso(),
            baseline={"movement_var": float("nan")},
            details={"slope": float("inf"), "note": "kept", "samples": 12},
        )
    )
    payload = state.snapshot()
    json.dumps(payload, allow_nan=False)
    (alert,) = [a for a in payload["alerts"] if a["track_id"] == 9]
    assert alert["details"]["slope"] is None
    assert alert["baseline"]["movement_var"] is None
    assert alert["details"]["note"] == "kept"
    assert alert["details"]["samples"] == 12


# -------------------------------------------------------------- scene fixture
def test_committed_scene_fixture_is_deterministic(tmp_path: Path) -> None:
    """Regenerate the committed fixture and compare bytes. No RNG, no clock, no drift."""
    regenerated = write_observations_jsonl(tmp_path / "scene_60s.jsonl", build_scene())
    assert regenerated.read_bytes() == SCENE_PATH.read_bytes()


def test_scene_fixture_holds_three_tracks_over_sixty_seconds() -> None:
    batches = list(SyntheticDetectionSource(SCENE_PATH).observations())
    assert len(batches) == round(SCENE_SECONDS * SCENE_HZ) + 1
    assert {obs.track_id for batch in batches for obs in batch} == {
        FALLEN_TRACK_ID,
        FIDGET_TRACK_ID,
        STAFF_TRACK_ID,
    }
    assert batches[0][0].t == 0.0
    assert batches[-1][0].t == pytest.approx(SCENE_SECONDS)
    assert all(obs.keypoints is not None for batch in batches for obs in batch)


def test_scene_fixture_stays_small() -> None:
    """Committed fixtures are a few hundred KB, not a few MB.

    Raising SCENE_HZ to the 15 fps the rest of the repo assumes triples this file for no
    behavioural gain, and dropping SCENE_DECIMALS puts seventeen significant figures of a
    sinusoid into git. Both are caught here.
    """
    assert SCENE_PATH.stat().st_size < 512 * 1024


# --------------------------------------------------------------- latency maths
def test_percentile_matches_the_numpy_default_method() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert RUNNER.percentile(values, 50.0) == pytest.approx(2.5)
    assert RUNNER.percentile(values, 95.0) == pytest.approx(3.85)
    assert RUNNER.percentile([7.0], 95.0) == pytest.approx(7.0)
    assert RUNNER.percentile([], 50.0) is None
    with pytest.raises(ValueError, match="q must be"):
        RUNNER.percentile(values, 101.0)


def test_stage_timer_reports_none_for_a_stage_that_never_ran() -> None:
    timer = RUNNER.StageTimer("synthetic-fixture")
    timer.start_frame()
    timer.run("source", 1, lambda: None)
    frame = timer.frame_ms()
    assert set(frame) == set(STAGES)
    assert frame["source"] is not None
    assert frame["detect"] is None
    assert timer.quantile_ms(50.0)["detect"] is None
    assert "stages that did not run" in timer.table()
    with pytest.raises(KeyError):
        timer.record("inference", 1.0, 1)


# ------------------------------------------------- the footage path, model-free
class FakeDetections:
    """Just enough of vividly.models.base.Detections to exercise the conversion.

    Lists rather than numpy arrays on purpose: the conversion has to work without numpy
    installed, and a test that needed numpy could not run on the machine that proves it.
    """

    def __init__(self, xyxy: list[list[float]], track_ids: list[int] | None) -> None:
        self.xyxy = xyxy
        self.track_ids = track_ids

    def __len__(self) -> int:
        return len(self.xyxy)


class FakePoses:
    def __init__(self, keypoints: list[list[list[float]]], scores: list[list[float]]) -> None:
        self.keypoints = keypoints
        self.scores = scores

    def __len__(self) -> int:
        return len(self.keypoints)


def test_tracked_boxes_and_poses_become_track_observations(tmp_path: Path) -> None:
    """Boxes to centres, poses to COCO-17 triples, ids straight through from the tracker."""
    pipeline = RUNNER.Pipeline(
        {"run_name": "conv", "detections_from": str(SCENE_PATH), "output": {"dir": str(tmp_path)}},
        REPO,
    )
    try:
        tracked = FakeDetections([[10.0, 20.0, 30.0, 60.0]], track_ids=[7])
        poses = FakePoses(
            keypoints=[[[float(i), float(i) + 1.0] for i in range(17)]],
            scores=[[0.5] * 17],
        )
        observations = pipeline._observations_from(tracked, poses, t=1.5)
    finally:
        pipeline.close()

    assert len(observations) == 1
    obs = observations[0]
    assert (obs.track_id, obs.t) == (7, 1.5)
    assert (obs.cx, obs.cy, obs.w, obs.h) == (20.0, 40.0, 20.0, 40.0)
    assert obs.keypoints is not None and len(obs.keypoints) == 17
    assert obs.keypoints[0] == (0.0, 1.0, 0.5)


def test_untracked_detections_produce_no_observations(tmp_path: Path) -> None:
    """A tracker that returned no ids has nothing the track layer can key on."""
    pipeline = RUNNER.Pipeline(
        {"run_name": "conv", "detections_from": str(SCENE_PATH), "output": {"dir": str(tmp_path)}},
        REPO,
    )
    try:
        untracked = FakeDetections([[0.0, 0.0, 1.0, 1.0]], track_ids=None)
        assert pipeline._observations_from(untracked, None, t=0.0) == []
    finally:
        pipeline.close()


def test_render_stage_encodes_a_frame() -> None:
    """The render stage is a JPEG encode and a base64, and nothing else."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from vividly.dashboard import encode_jpeg_b64

    encoded = encode_jpeg_b64(np.zeros((16, 24, 3), np.uint8), quality=70)
    assert isinstance(encoded, str) and encoded


# ------------------------------------------------------------- the headless run
def synthetic_run_config(tmp_path: Path, **overrides: Any) -> Path:
    """The shipped synthetic config, redirected to a temp output directory.

    Loading the real configs/pipeline/synthetic_demo.yaml rather than writing a bespoke one
    is deliberate: the thresholds that fire in this test are the thresholds that ship.
    """
    cfg = yaml.safe_load(SYNTHETIC_CONFIG.read_text(encoding="utf-8"))
    cfg["run_name"] = "test_synth"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    cfg.update(overrides)
    path = tmp_path / "synthetic.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_headless_synthetic_run_writes_alerts_and_latency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = synthetic_run_config(tmp_path)
    assert RUNNER.main(["--config", str(config), "--headless"]) == 0

    out_dir = tmp_path / "outputs" / "test_synth"
    db_path = out_dir / "alerts.sqlite"
    latency_path = out_dir / "latency.jsonl"
    assert db_path.exists()
    assert latency_path.exists()

    store = EventStore(db_path)
    try:
        alerts = store.alerts(limit=100)
    finally:
        store.close()
    falls = [a for a in alerts if a.rule_id == "fall_then_still_v0"]
    assert falls, f"no fall_then_still alert in {[a.rule_id for a in alerts]}"
    fired = falls[0]
    assert fired.track_id == FALLEN_TRACK_ID
    # The onset is the fall itself (the last upright sample the rule could anchor on),
    # not the moment the confirmation window closed: the rule walks back through stored
    # posture history to find it.
    assert fired.onset_ts == pytest.approx(SCENE_LEAD_IN_S, abs=0.6)
    # Detection lands once the down span reaches the shipped down_still_min_s (30 s):
    # this run uses the unshrunk defaults, and the confirmation stage is the latency.
    assert fired.detected_ts == pytest.approx(fired.onset_ts + SCENE_FALL_S + 30.0, abs=1.5)
    assert fired.feedback is None
    # The track's own baseline travels with the alert, so the record stays readable after
    # the TrackBook that produced it is gone.
    assert fired.baseline is not None and fired.baseline["complete"] is True
    # Exactly one firing on this scene, on the one person who fell and stayed down. The
    # fidgeting neighbour and the patrol are both negative controls, and under the
    # acute-event rule set, a firing on either would be the defect class that matters most.
    assert [(a.rule_id, a.track_id) for a in alerts] == [("fall_then_still_v0", FALLEN_TRACK_ID)]

    records = [json.loads(line) for line in latency_path.read_text(encoding="utf-8").splitlines()]
    frames = [r for r in records if not r.get("summary")]
    assert len(frames) == round(SCENE_SECONDS * SCENE_HZ) + 1
    for row in frames:
        assert set(row["ms"]) == set(STAGES)
        assert row["ms"]["source"] is not None
        assert row["ms"]["rules"] is not None
        # Zero footage: no detector, no pose, no tracker, nothing to render.
        assert row["ms"]["detect"] is None
        assert row["ms"]["pose"] is None
        assert row["ms"]["track"] is None
        assert row["ms"]["render"] is None

    # The percentiles are written as well as printed, as the last line of the same file.
    summary = records[-1]
    assert summary["summary"] is True
    assert summary["n_frames"] == len(frames)
    assert set(summary["p50_ms"]) == set(STAGES)
    assert set(summary["p95_ms"]) == set(STAGES)
    assert summary["p50_ms"]["source"] is not None
    assert summary["p95_ms"]["detect"] is None
    assert "PROVISIONAL" in summary["threshold_status"]

    printed = capsys.readouterr().out
    assert "p50 ms" in printed and "p95 ms" in printed
    assert "PROVISIONAL" in printed
    for stage in STAGES:
        assert stage in printed


def test_headless_run_is_idempotent_over_the_same_store(tmp_path: Path) -> None:
    """Alert ids are derived, so a second pass over the same clip doubles nothing."""
    config = synthetic_run_config(tmp_path)
    RUNNER.main(["--config", str(config), "--headless"])
    db_path = tmp_path / "outputs" / "test_synth" / "alerts.sqlite"
    store = EventStore(db_path)
    try:
        first = store.count()
    finally:
        store.close()

    RUNNER.main(["--config", str(config), "--headless"])
    store = EventStore(db_path)
    try:
        assert store.count() == first
    finally:
        store.close()


def latency_rows(tmp_path: Path, run_name: str = "test_synth") -> list[dict[str, Any]]:
    path = tmp_path / "outputs" / run_name / "latency.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_max_frames_stops_the_run_early(tmp_path: Path) -> None:
    config = synthetic_run_config(tmp_path)
    assert RUNNER.main(["--config", str(config), "--headless", "--max-frames", "10"]) == 0
    records = latency_rows(tmp_path)
    assert len([r for r in records if not r.get("summary")]) == 10
    assert records[-1]["n_frames"] == 10


def test_max_frames_zero_opens_nothing(tmp_path: Path) -> None:
    """Zero is a config check, not a one-frame run: the source is never touched."""
    config = synthetic_run_config(tmp_path)
    assert RUNNER.main(["--config", str(config), "--headless", "--max-frames", "0"]) == 0
    records = latency_rows(tmp_path)
    assert [r for r in records if not r.get("summary")] == []
    assert records[-1]["p50_ms"] == dict.fromkeys(STAGES)


def test_pipeline_state_is_live_during_a_run(tmp_path: Path) -> None:
    """The dashboard reads the same objects the loop writes; nothing is copied across."""
    cfg = yaml.safe_load(SYNTHETIC_CONFIG.read_text(encoding="utf-8"))
    cfg["run_name"] = "test_live"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    pipeline = RUNNER.Pipeline(cfg, REPO)
    try:
        pipeline.run(max_frames=30)
        snapshot = pipeline.state.snapshot()
    finally:
        pipeline.close()
    assert snapshot["mode"] == "synthetic"
    assert snapshot["frame_b64"] is None
    assert set(snapshot["timers"]) == {"1", "2", "99"}
    assert snapshot["latency_ms"]["detect"] is None
    assert snapshot["frame_index"] == 29


# --------------------------------------------------------------- config wiring
def test_synthetic_demo_config_points_at_the_committed_fixture() -> None:
    cfg = yaml.safe_load(SYNTHETIC_CONFIG.read_text(encoding="utf-8"))
    assert (REPO / cfg["detections_from"]).resolve() == SCENE_PATH.resolve()
    assert cfg["headless"] is True
    assert "detector" not in cfg and "pose" not in cfg and "tracker" not in cfg
    # No rules block at all, on purpose: the demo runs the RuleConfig defaults, which
    # test_rules.py pins equal to configs/pipeline/rules.yaml. The acute-event cascade fires on
    # the fixture at the shipped thresholds, unshrunk: that is the demo's claim.
    assert "rules" not in cfg and "rules_config" not in cfg
    TrackConfig.from_mapping(cfg["tracks"])


def test_replay_demo_config_names_backends_and_the_shared_thresholds() -> None:
    cfg = yaml.safe_load(REPLAY_CONFIG.read_text(encoding="utf-8"))
    # The eyes lane of record is served through the rtdetr_onnx backend on
    # final_models/eyes/model.onnx; joints stays on rtmlib.
    assert cfg["detector"]["backend"] == "rtdetr_onnx"
    assert cfg["pose"]["backend"] == "rtmlib"
    # botsort since the 2026-08-31 tracker swap (research/v2-training-stack.md, section 7a);
    # bytetrack is the frozen baseline.
    assert cfg["tracker"]["backend"] == "botsort"
    assert cfg["detector"]["device"] == "cpu", "the CoreML EP crashes both detector graphs"
    assert cfg["dashboard"]["port"] == 8600
    assert cfg["dashboard"]["host"] == "127.0.0.1"
    assert (REPO / cfg["source"]["path"]).exists()
    assert (REPO / cfg["rules_config"]).exists()
    assert "rules" not in cfg, "rules_config and an inline rules block are mutually exclusive"


def test_every_pipeline_config_marks_its_thresholds_provisional() -> None:
    for config in sorted((REPO / "configs" / "pipeline").glob("*.yaml")):
        text = config.read_text(encoding="utf-8")
        assert "PROVISIONAL" in text, f"{config.name} does not say its thresholds are provisional"
        assert "CALIBRATION.md" in text, f"{config.name} does not point at the calibration doc"


def test_runner_refuses_two_sources_of_truth_for_a_threshold(tmp_path: Path) -> None:
    cfg = yaml.safe_load(SYNTHETIC_CONFIG.read_text(encoding="utf-8"))
    cfg["rules"] = {"refire_after_s": 900.0}
    cfg["rules_config"] = "configs/pipeline/rules.yaml"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    with pytest.raises(RUNNER.ConfigError, match="both"):
        RUNNER.Pipeline(cfg, REPO)


def test_runner_refuses_a_footage_config_with_no_source(tmp_path: Path) -> None:
    cfg = {"run_name": "broken", "output": {"dir": str(tmp_path / "outputs")}}
    with pytest.raises(RUNNER.ConfigError, match="detections_from"):
        RUNNER.Pipeline(cfg, REPO)


def test_a_config_error_writes_no_artifacts(tmp_path: Path) -> None:
    """Validation happens before the store is opened, so a bad config leaves no debris."""
    cfg = {"run_name": "broken", "output": {"dir": str(tmp_path / "outputs")}}
    with pytest.raises(RUNNER.ConfigError):
        RUNNER.Pipeline(cfg, REPO)
    assert not (tmp_path / "outputs").exists()


def test_main_reports_a_config_error_as_exit_3(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    assert RUNNER.main(["--config", str(missing)]) == 3


# -------------------------------------------------------------- import hygiene
def _module_level_statements(tree: ast.Module) -> list[ast.stmt]:
    """Top-level statements, skipping the bodies of `if TYPE_CHECKING:` blocks."""
    out: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            continue
        out.append(node)
    return out


def test_dashboard_and_runner_have_no_top_level_heavy_imports() -> None:
    """Linux CI collects this file with no numpy and no cv2 installed.

    fastapi, uvicorn and websockets are pure python and cross-platform, so they are fine at
    module level. cv2 and numpy are not, and an import of either at module level in these
    files turns CI red rather than failing a test, which is why this is an AST check.
    """
    offenders: list[str] = []
    for py in [*sorted(DASHBOARD_SRC.rglob("*.py")), RUN_PIPELINE]:
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in _module_level_statements(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name in {"cv2", "numpy", "torch", "rtmlib", "trackeval", "supervision"}:
                    offenders.append(f"{py.name}:{node.lineno}: {name}")
    assert not offenders, f"module-level heavy imports: {offenders}"


def test_importing_the_runner_pulls_in_no_model_stack() -> None:
    """Belt and braces over the AST check: import it for real and look at sys.modules."""
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('rp', {str(RUN_PIPELINE)!r});"
        "module = importlib.util.module_from_spec(spec);"
        "sys.modules['rp'] = module;"
        "spec.loader.exec_module(module);"
        "print([m for m in ('numpy', 'cv2', 'torch', 'rtmlib', 'trackeval') if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert proc.stdout.strip() == "[]"
