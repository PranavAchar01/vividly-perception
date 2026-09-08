"""The dashboard's re-association proposal cards: the surface where a human answers.

What these tests pin, in one sentence: beyond the tracker's own re-association window this
app shows a question and never answers it, and whichever answer a person gives is written
down as a label. Everything else here is scaffolding around that claim.
`test_confirm_writes_the_label_and_fires_the_merge_hook` and
`test_reject_writes_the_label_and_the_pair_is_never_asked_again` are the two load-bearing
ones; `test_history_outlives_the_cleared_gallery` is the one CQ-16 in
docs/COUNSEL_QUESTIONS.md points at;
`test_a_double_decision_is_a_409_and_changes_nothing` is the one that keeps a stored label
from being whichever click landed last.

Four rules hold throughout, three of them inherited from tests/test_dashboard.py.

  * No models. Nothing here downloads a weight, opens an ONNX session or decodes a video.
    The embeddings come from `vividly.reid.synth`'s invented cone geometry through the
    synthetic embedding source, so a green run proves the wiring carries an answer and says
    nothing whatsoever about real re-identification (docs/CALIBRATION.md owns that).
  * No network, no server. The FastAPI app is driven in process through its ASGI interface;
    uvicorn is never started and no socket is ever bound.
  * Deterministic. No RNG, no wall-clock assertion. The one wall clock in the mechanism is
    `decided_ts`, and it is asserted to be finite and non-decreasing, never to a value.
  * Real store, real book, real stage. "The label reaches sqlite" and "the confirmation
    reaches the merge hook" are claims about wiring, and a double would prove that a method
    was called rather than that anything survived.

Why not starlette's TestClient. Same reason tests/test_dashboard.py gives: importing it
imports httpx unconditionally, httpx is not installed in this repo's default fork, and
skipping the endpoint tests on the machine that has to prove them would be worse than
driving the same ASGI callable TestClient drives. `AsgiClient` below is that driver, kept
local rather than imported from the sibling test module so that neither file's fixtures can
break the other's.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from vividly.dashboard import (
    BUSY_RETRY_AFTER_S,
    CROPS_DIRNAME,
    MAX_CROP_BYTES,
    MAX_VIA_CHARS,
    PAGE_HTML,
    PipelineState,
    create_app,
    crop_name,
    find_crop,
    identity_color,
    is_servable_crop_name,
    resolve_crop_path,
)
from vividly.events import EventStore
from vividly.reid.gallery import GalleryConfig
from vividly.reid.stage import EmbeddingsSpec, ReidConfig, build_reid_stage
from vividly.schemas.events import AlertRecord, make_alert_id, utc_now_iso
from vividly.schemas.reassoc import make_proposal_id
from vividly.tracks import TrackBook, TrackConfig, TrackObservation

REPO = Path(__file__).resolve().parent.parent
PIPELINE_DOC = REPO / "docs" / "PIPELINE.md"

# Stand-in crop bytes. The route hands a file to a browser and never decodes one, so a real
# encoder would add a cv2 dependency to a test that is about path containment and caching.
# The PNG signature is here only so a human looking at the fixture knows what it stands for.
CROP_BYTES = b"\x89PNG\r\n\x1a\n stand-in crop bytes, never decoded by anything"


# ------------------------------------------------------------------ ASGI driver
@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    #: Raw response headers, in the order the app sent them. Kept as bytes pairs because
    #: that is what ASGI carries, and decoded on lookup rather than up front.
    headers: tuple[tuple[bytes, bytes], ...] = ()

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    @property
    def json(self) -> Any:
        return json.loads(self.body)

    def header(self, name: str) -> str | None:
        """One header by case-insensitive name, or None. `Retry-After` is why this exists."""
        wanted = name.lower().encode("ascii")
        for key, value in self.headers:
            if key.lower() == wanted:
                return value.decode("latin-1")
        return None


class AsgiClient:
    """The app's own ASGI callable, driven directly. No socket, no httpx, no uvicorn.

    Every await is wrapped in a timeout so a handler that never returns fails as a timeout
    rather than by wedging the suite.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

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
        start = next(m for m in received if m["type"] == "http.response.start")
        payload = b"".join(
            m.get("body", b"") for m in received if m["type"] == "http.response.body"
        )
        return Response(
            status=start["status"],
            body=payload,
            headers=tuple((k, v) for k, v in start.get("headers", [])),
        )

    def stream(
        self,
        path: str = "/ws",
        n_payloads: int = 1,
        between: Callable[[int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Connect, collect n_payloads pushes, disconnect. Never blocks forever.

        `between` runs on the loop thread after each payload, which is how the live-push
        test writes a proposal into the store partway through a stream and then proves the
        next push carries it.
        """
        return asyncio.run(self._stream(path, n_payloads, between))

    async def _stream(
        self, path: str, n_payloads: int, between: Callable[[int], None] | None
    ) -> list[dict[str, Any]]:
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
            for index in range(n_payloads):
                message = await asyncio.wait_for(from_app.get(), timeout=10)
                assert message["type"] == "websocket.send", message
                payloads.append(json.loads(message["text"]))
                if between is not None:
                    between(index)
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


# ---------------------------------------------------------------------- fixtures
def obs(track_id: int, t: float, cx: float = 0.0, cy: float = 0.0) -> TrackObservation:
    return TrackObservation(track_id=track_id, t=t, cx=cx, cy=cy, w=10.0, h=20.0)


REAP_AFTER_S = 5.0
#: Tracks 1 and 2 are the same synthetic identity, 3 is a different one. The answer key
#: lives here rather than in the assertions so that "the system asked about the right pair"
#: is a statement about the fixture, not about whatever the embedder happened to do.
IDENTITY_BY_TRACK = {1: 0, 2: 0, 3: 1}


class Harness:
    """A live pipeline state with a real store, a real book and a real re-ID stage."""

    def __init__(self, tmp_path: Path, *, with_reid: bool = True, run_dir: Path | None = None):
        self.store = EventStore(tmp_path / "alerts.sqlite")
        self.book = TrackBook(
            TrackConfig(reap_after_s=REAP_AFTER_S, baseline_window_s=1.0, ring_size=100)
        )
        self.stage: Any = None
        if with_reid:
            config = ReidConfig(
                enabled=True,
                gallery=GalleryConfig(dim=16),
                embeddings=EmbeddingsSpec(
                    source="synthetic", dim=16, identity_by_track=IDENTITY_BY_TRACK
                ),
            )
            self.stage = build_reid_stage(
                config, store=self.store, book=self.book, reap_after_s=REAP_AFTER_S
            )
        self.state = PipelineState(
            book=self.book,
            store=self.store,
            reid=self.stage,
            run_dir=run_dir,
            run_name="test",
            mode="synthetic",
            stream_hz=200.0,
        )
        self.client = AsgiClient(create_app(self.state))

    def close(self) -> None:
        if self.stage is not None:
            self.stage.close()
        self.store.close()

    # -- driving the stage -------------------------------------------------
    def propose_a_return(self) -> dict[str, Any]:
        """Track 1 leaves, track 2 arrives looking like it, and the stage asks about it.

        This is the whole mechanism in five lines and it is the fixture every decision test
        starts from. The gap between the two is far wider than the book's reap window, so
        the automatic path has already declined: no proposal here is a near miss of the
        tracker's own re-association.
        """
        assert self.stage is not None
        self.book.ingest([obs(1, 0.0)])
        self.stage.step([obs(1, 0.0)], 0.0)
        self.book.reap(20.0)
        self.book.ingest([obs(2, 30.0)])
        result = self.stage.step([obs(2, 30.0)], 30.0)
        assert len(result.proposals) == 1, result
        return {"record": result.proposals[0], "id": result.proposals[0].proposal_id}


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    built = Harness(tmp_path)
    yield built
    built.close()


@pytest.fixture
def bare(tmp_path: Path) -> Iterator[Harness]:
    """A dashboard with no re-ID stage at all: the default and the shipped state."""
    built = Harness(tmp_path, with_reid=False)
    yield built
    built.close()


@pytest.fixture
def with_crops(tmp_path: Path) -> Iterator[Harness]:
    run_dir = tmp_path / "run"
    (run_dir / CROPS_DIRNAME).mkdir(parents=True)
    built = Harness(tmp_path, run_dir=run_dir)
    yield built
    built.close()


def write_crop(run_dir: Path, track_id: int, suffix: str = ".png") -> Path:
    path = run_dir / CROPS_DIRNAME / crop_name(track_id, suffix)
    path.write_bytes(CROP_BYTES)
    return path


# =============================================================== the page itself
def test_the_page_carries_every_proposal_affordance() -> None:
    """The card, both answers, and the endpoints they reach, all in the one document."""
    for needle in (
        "Confirm same person",
        "Reject, different people",
        "/api/proposals/",
        "/confirm",
        "/reject",
        "renderProposals",
        "identityTile",
    ):
        assert needle in PAGE_HTML, f"the page is missing {needle!r}"


def test_the_page_says_a_human_decides_and_nothing_is_auto_bound() -> None:
    """The semantic, printed on the page rather than left in a docstring.

    A reviewer looking at two photographs of two patients has to be able to read, without
    leaving the page, that the system did not decide this and that their click is what
    writes the answer down.
    """
    page = PAGE_HTML
    assert "automatic and unchanged" in page
    assert "nothing is ever bound automatically" in page
    assert "Nothing is bound" in page
    assert "labels" in page


def test_the_page_states_the_session_scoped_retention_rule() -> None:
    """CQ-16 on the card list header: the labels persist, the embeddings do not."""
    page = PAGE_HTML
    assert "session-scoped" in page
    assert "CQ-16" in page
    assert "CLEARED" in page
    assert "no embedding, no crop and no appearance vector of any kind" in page
    # And the panel must not let a score read as a probability of identity.
    assert "PROVISIONAL" in page
    assert "not a probability" in page


def test_the_proposal_panel_loads_no_external_asset() -> None:
    """POLICY.md, rule 7 again, over the proposal markup."""
    for forbidden in ('src="http', 'href="http', "cdn.", "googleapis", "unpkg", "jsdelivr"):
        assert forbidden not in PAGE_HTML, f"page reaches for an external asset: {forbidden}"


# ======================================================== pure display helpers
def test_identity_color_is_deterministic_and_separates_neighbours() -> None:
    assert identity_color(7) == identity_color(7)
    assert identity_color(7) != identity_color(8)
    # A display device and nothing more: same id, same colour, whatever the run.
    assert identity_color(0).startswith("hsl(")


def test_crop_name_is_zero_padded_so_a_listing_sorts_readably() -> None:
    assert crop_name(4) == "track_000004.jpg"
    assert crop_name(4, ".png") == "track_000004.png"
    assert sorted([crop_name(2), crop_name(10)]) == [crop_name(2), crop_name(10)]


def test_find_crop_is_none_without_a_directory_and_without_a_file(tmp_path: Path) -> None:
    assert find_crop(None, 1) is None
    assert find_crop(tmp_path, 1) is None
    (tmp_path / crop_name(1, ".png")).write_bytes(CROP_BYTES)
    assert find_crop(tmp_path, 1) == crop_name(1, ".png")


def test_is_servable_crop_name_refuses_anything_that_is_not_an_image() -> None:
    assert is_servable_crop_name("track_000001.jpg")
    assert is_servable_crop_name("track_000001.PNG")
    for bad in ("", "alerts.sqlite", "notes.txt", "track_000001.jpg\x00.txt", "a\\b.jpg"):
        assert not is_servable_crop_name(bad), bad


def test_resolve_crop_path_accepts_a_name_inside_the_directory(tmp_path: Path) -> None:
    crops = tmp_path / CROPS_DIRNAME
    crops.mkdir()
    (crops / "track_000001.jpg").write_bytes(CROP_BYTES)
    resolved = resolve_crop_path(crops, "track_000001.jpg")
    assert resolved is not None
    assert resolved == (crops / "track_000001.jpg").resolve()


def test_resolve_crop_path_refuses_every_way_out_of_the_directory(tmp_path: Path) -> None:
    """Dot-dot, absolute, and a symlink pointing out. All three land outside; all three fail."""
    crops = tmp_path / CROPS_DIRNAME
    crops.mkdir()
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"not for the browser")
    (crops / "escape.jpg").symlink_to(outside)

    assert resolve_crop_path(crops, "../secret.jpg") is None
    assert resolve_crop_path(crops, "../../secret.jpg") is None
    assert resolve_crop_path(crops, str(outside)) is None
    assert resolve_crop_path(crops, "escape.jpg") is None


# ============================================================== GET /api/proposals
def test_a_pending_proposal_reaches_the_endpoint(harness: Harness) -> None:
    asked = harness.propose_a_return()
    body = harness.client.get("/api/proposals").json
    assert body["available"] is True
    assert body["enabled"] is True
    assert [p["proposal_id"] for p in body["pending"]] == [asked["id"]]
    assert body["decided"] == []
    card = body["pending"][0]
    assert (card["new_track"], card["candidate_track"]) == (2, 1)
    assert card["status"] == "pending"


def test_a_card_carries_the_facts_a_reviewer_needs(harness: Harness) -> None:
    """Score, gallery size, support, what is known about the absence, and the two colours.

    `away_at_least_s` rather than an exact gap is deliberate and it is the honest number: a
    proposal exists only because the candidate cleared the tracker's window, so the absence
    is at least that long, and the exact figure is not recoverable from a stored row today.
    An invented precise number next to a patient identity decision would be worse than a
    bound.
    """
    harness.propose_a_return()
    card = harness.client.get("/api/proposals").json["pending"][0]
    assert 0.0 <= card["confidence"] <= 1.0
    assert card["gallery_size"] == 1
    assert card["candidate_support"] == 1
    assert card["away_at_least_s"] == REAP_AFTER_S
    assert card["away_s"] is None
    assert card["asked_ago_s"] == 0.0
    assert card["new_track_live"] is True
    assert card["new_color"] == identity_color(2)
    assert card["candidate_color"] == identity_color(1)
    # A card holds nothing derived from appearance. It is two ids, a score and a count.
    assert not any("embedding" in key or "vector" in key for key in card)


def test_the_session_block_reports_live_retention_state(harness: Harness) -> None:
    harness.propose_a_return()
    session = harness.client.get("/api/proposals").json["session"]
    assert session["gallery_cleared"] is False
    assert session["n_identities"] == 1
    assert session["n_gallery_rows"] >= 1
    assert session["min_gap_s"] == REAP_AFTER_S
    assert session["reid"]["n_proposals"] == 1
    assert session["reid"]["n_confirmed"] == 0
    assert "PROVISIONAL" in session["reid"]["threshold_status"]


def test_the_gallery_latency_panel_reports_cost_beside_gallery_size(harness: Harness) -> None:
    """POLICY.md, rule 9's shape: a latency number is only readable next to what produced it."""
    harness.propose_a_return()
    latency = harness.client.get("/api/proposals").json["session"]["gallery_latency"]
    assert latency["n_calls"] >= 1
    assert latency["last_ms"] >= 0.0
    assert latency["last_gallery_size"] >= 1
    assert latency["max_gallery_size"] >= latency["last_gallery_size"]


def test_a_gallery_that_has_never_matched_reports_null_rather_than_nan(harness: Harness) -> None:
    """ProposeStats is nan before the first call, and nan is not JSON a browser can parse."""
    body = harness.client.get("/api/proposals")
    assert "NaN" not in body.text
    latency = body.json["session"]["gallery_latency"]
    assert latency["last_ms"] is None
    assert latency["mean_s"] is None


def test_pending_is_oldest_first_so_a_cap_never_hides_the_oldest_question(
    harness: Harness,
) -> None:
    """The queue order the store promises, carried through the endpoint unchanged."""
    for index in range(3):
        harness.store.record_proposal(
            new_track=10 + index,
            candidate_track=1,
            confidence=0.9,
            ts=float(index),
            gallery_size=1,
        )
    harness.state.proposal_limit = 2
    body = harness.client.get("/api/proposals").json
    assert [p["ts"] for p in body["pending"]] == [0.0, 1.0]
    assert body["n_pending_shown"] == 2
    assert body["limits"]["pending"] == 2


def test_decided_cards_come_back_newest_first(harness: Harness) -> None:
    for index in range(3):
        record = harness.store.record_proposal(
            new_track=10 + index,
            candidate_track=1,
            confidence=0.9,
            ts=float(index),
            gallery_size=1,
        )
        harness.store.decide_proposal(record.proposal_id, "rejected", 100.0, "test")
    body = harness.client.get("/api/proposals").json
    assert [p["ts"] for p in body["decided"]] == [2.0, 1.0, 0.0]
    assert body["pending"] == []


def test_a_run_without_reid_says_so_instead_of_showing_an_empty_queue(bare: Harness) -> None:
    """An empty queue and a queue that cannot exist look identical and mean the opposite."""
    body = bare.client.get("/api/proposals").json
    assert body["available"] is True
    assert body["enabled"] is False
    assert body["pending"] == []
    assert body["session"]["gallery_cleared"] is None
    assert "reid" not in body["session"]


def test_a_closed_store_is_a_503_and_not_a_traceback(harness: Harness) -> None:
    harness.stage.close()
    harness.store.close()
    response = harness.client.get("/api/proposals")
    assert response.status == 503
    assert "closed" in response.json["detail"]


# ================================================================ the two answers
def test_confirm_writes_the_label_and_fires_the_merge_hook(harness: Harness) -> None:
    """The load-bearing test. A person clicks, and only then does anything join.

    Before the click the two tracks are unrelated in the book. After it, track 2 carries
    `reassociated_from = 1`, which is `TrackBook.merge_identity` and nothing else: the
    dashboard never touches a TrackState itself.
    """
    asked = harness.propose_a_return()
    assert harness.book.tracks()[2].reassociated_from is None

    response = harness.client.post(f"/api/proposals/{asked['id']}/confirm")
    assert response.status == 200
    body = response.json
    assert body["decision"] == "confirmed"
    assert body["status"] == "confirmed"
    assert body["merged"] is True
    assert body["stored"] is True

    assert harness.book.tracks()[2].reassociated_from == 1
    stored = harness.store.get_proposal(asked["id"])
    assert stored is not None
    assert stored.status == "confirmed"
    assert stored.decided_via == "dashboard"
    assert stored.decided_ts is not None


def test_a_confirmation_carries_the_staff_tag_and_the_mock_binding_forward(
    harness: Harness,
) -> None:
    """What "the same person" has to mean downstream: their tags come with them."""
    harness.book.ingest([obs(1, 0.0)])
    harness.stage.step([obs(1, 0.0)], 0.0)
    harness.book.set_staff(1, True)
    harness.book.bind_mrn(1, "MRN-0001")
    harness.book.reap(20.0)
    harness.book.ingest([obs(2, 30.0)])
    result = harness.stage.step([obs(2, 30.0)], 30.0)

    harness.client.post(f"/api/proposals/{result.proposals[0].proposal_id}/confirm")
    track = harness.book.tracks()[2]
    assert track.is_staff is True
    assert track.mrn_binding == "MRN-0001"


def test_reject_writes_the_label_and_the_pair_is_never_asked_again(harness: Harness) -> None:
    """A rejection is a label, not a dismissal, and it is what a false-merge rate needs.

    The second half is the exclusion: new evidence arrives (a third person is enrolled, so
    every live track becomes probe-eligible again) and the rejected pair is not put in front
    of anybody a second time.
    """
    asked = harness.propose_a_return()
    response = harness.client.post(f"/api/proposals/{asked['id']}/reject")
    assert response.status == 200
    assert response.json["merged"] is False
    assert "different people" in response.json["note"]

    stored = harness.store.get_proposal(asked["id"])
    assert stored is not None and stored.status == "rejected"
    assert harness.book.tracks()[2].reassociated_from is None

    # New evidence: track 3 leaves and is enrolled, so track 2 is probed again.
    harness.book.ingest([obs(2, 31.0), obs(3, 31.0)])
    harness.stage.step([obs(2, 31.0), obs(3, 31.0)], 31.0)
    harness.book.ingest([obs(2, 60.0)])
    harness.book.reap(60.0)
    harness.book.ingest([obs(2, 61.0)])
    result = harness.stage.step([obs(2, 61.0)], 61.0)
    assert result.enrolled == (3,)
    assert result.probed == (2,)
    assert result.proposals == ()
    assert harness.stage.stats().n_skipped_asked == 1
    assert harness.client.get("/api/proposals").json["pending"] == []


def test_the_rejected_label_reaches_the_flywheel_export(harness: Harness) -> None:
    """The point of storing a rejection: it comes back out as a labelled score."""
    asked = harness.propose_a_return()
    harness.client.post(f"/api/proposals/{asked['id']}/reject", {"via": "dashboard:charge_nurse"})
    rows = harness.store.labels_as_rows()
    assert len(rows) == 1
    assert rows[0]["decision"] == "rejected"
    assert rows[0]["label"] == 0
    assert rows[0]["decided_via"] == "dashboard:charge_nurse"
    assert 0.0 <= rows[0]["confidence"] <= 1.0
    assert rows[0]["gallery_size"] == 1


def test_a_double_decision_is_a_409_and_changes_nothing(harness: Harness) -> None:
    """Decisions are written once. The second reviewer is told, not silently overruled."""
    asked = harness.propose_a_return()
    assert harness.client.post(f"/api/proposals/{asked['id']}/confirm").status == 200

    second = harness.client.post(f"/api/proposals/{asked['id']}/confirm")
    assert second.status == 409
    assert "already" in second.json["detail"]

    disagreeing = harness.client.post(f"/api/proposals/{asked['id']}/reject")
    assert disagreeing.status == 409

    stored = harness.store.get_proposal(asked["id"])
    assert stored is not None and stored.status == "confirmed"


# ------------------------------------------------------- identity exclusivity
# Two properties, one sentence each, and both of them are about a chart binding moving to
# the wrong person: a departed identity is claimed by at most one live track, and a live
# track holds at most one identity. The store enforces both per run in its own indexes and
# the stage enforces both in memory; this harness runs unscoped (`run_id=None`), which is
# precisely the case the database cannot refuse, so what these two tests exercise is the
# stage's own guard through the routes a person actually clicks. The database-index path is
# covered in tests/test_reid_pipeline.py, where the runner supplies a real run id.


def test_two_live_tracks_cannot_both_be_confirmed_onto_one_departed_track(
    harness: Harness,
) -> None:
    """The second confirm is refused, its label is still stored, and no chart binds twice.

    The one thing that must not happen is the second half: track 1's mock chart binding
    reaching two different live people because two reviewers each answered a card about the
    same departed person. The refusal is not silence, either. A confirm a person really gave
    is recorded, as a rejection (which is what the store's ConfirmConflictError contract asks
    of the stage), so the answer survives even though nothing was bound.

    The second question is written straight into the store because this fixture's answer key
    holds only one returning identity, and what is under test is the answer, not the asking.
    """
    harness.book.ingest([obs(1, 0.0)])
    harness.stage.step([obs(1, 0.0)], 0.0)
    harness.book.bind_mrn(1, "MRN-0001")
    harness.book.reap(20.0)
    harness.book.ingest([obs(2, 30.0)])
    first = harness.stage.step([obs(2, 30.0)], 30.0).proposals[0]

    # A second live person is in the room, and a card about the same departed track exists.
    harness.book.ingest([obs(2, 31.0), obs(3, 31.0)])
    second = harness.store.record_proposal(
        new_track=3, candidate_track=1, confidence=0.9, ts=31.0, gallery_size=1
    )

    won = harness.client.post(f"/api/proposals/{first.proposal_id}/confirm").json
    assert won["merged"] is True
    assert harness.book.tracks()[2].mrn_binding == "MRN-0001"

    lost = harness.client.post(f"/api/proposals/{second.proposal_id}/confirm")
    assert lost.status == 200, lost.text
    assert lost.json["merged"] is False
    # The route was /confirm and the stored answer is a rejection. Both are reported: the
    # click is what the person did, the status is what the record now says.
    assert lost.json["decision"] == "confirmed"
    assert lost.json["status"] == "rejected"
    assert "already claimed by live track 2" in lost.json["note"]

    # The label landed, with its provenance, so the answer is not lost.
    stored = harness.store.get_proposal(second.proposal_id)
    assert stored is not None
    assert stored.status == "rejected"
    assert stored.decided_via == "dashboard"
    assert stored.decided_ts is not None
    assert [row["label"] for row in harness.store.labels_as_rows()] == [1, 0]

    # And nothing bound twice: track 3 keeps its own blank slate.
    assert harness.book.tracks()[3].mrn_binding is None
    assert harness.book.tracks()[3].reassociated_from is None


def test_one_live_track_cannot_take_two_identities(harness: Harness) -> None:
    """A track that has been given an identity does not get a second one.

    Same refusal from the other side. Track 2 is confirmed as track 1 returning; a later
    card asking whether it is instead track 4 returning is refused, the answer is kept as a
    rejection, and the link the first answer made is untouched. Track 4 is a person who left
    without ever being embedded (no identity in the fixture's answer key), which is why the
    second card has to be written into the store directly.
    """
    harness.book.ingest([obs(1, 0.0), obs(4, 0.0)])
    harness.stage.step([obs(1, 0.0), obs(4, 0.0)], 0.0)
    harness.book.reap(20.0)
    harness.book.ingest([obs(2, 30.0)])
    first = harness.stage.step([obs(2, 30.0)], 30.0).proposals[0]
    second = harness.store.record_proposal(
        new_track=2, candidate_track=4, confidence=0.9, ts=31.0, gallery_size=1
    )

    assert harness.client.post(f"/api/proposals/{first.proposal_id}/confirm").json["merged"] is True

    lost = harness.client.post(f"/api/proposals/{second.proposal_id}/confirm")
    assert lost.status == 200, lost.text
    assert lost.json["merged"] is False
    assert lost.json["status"] == "rejected"
    assert "already has a confirmed identity" in lost.json["note"]

    stored = harness.store.get_proposal(second.proposal_id)
    assert stored is not None and stored.status == "rejected"
    # The first answer stands: track 2 is still the person track 1 was, and only that.
    assert harness.book.tracks()[2].reassociated_from == 1


# --------------------------------------------------------------- a busy store
def hold_the_write_lock(harness: Harness) -> sqlite3.Connection:
    """Take SQLite's single write lock from another connection and keep it.

    The store's own busy timeout is dropped to zero first, exactly as
    tests/test_reassoc_store.py does it, so the test exercises the translation rather than
    waiting five seconds for it. `_conn` is private and this is the only way to reach the
    pragma; the alternative is a test that sleeps.
    """
    harness.store._conn.execute("PRAGMA busy_timeout = 0")
    blocker = sqlite3.connect(str(harness.store.path))
    blocker.isolation_level = None
    blocker.execute("BEGIN IMMEDIATE")
    return blocker


def test_a_busy_store_is_a_503_with_retry_after_and_the_label_is_not_lost(
    harness: Harness,
) -> None:
    """A lock is not a crash, and a refused answer is not a stored answer.

    The write could not land, so the honest report is 503 with Retry-After: nothing was
    written, try again shortly. What must not happen is a 500 that leaves a reviewer unable
    to tell whether their answer about two patients' identities was recorded. The proposal
    stays pending, which is what puts the card back in the queue with its buttons live, and
    the same click stores the same label once the lock frees.
    """
    asked = harness.propose_a_return()
    blocker = hold_the_write_lock(harness)
    try:
        refused = harness.client.post(f"/api/proposals/{asked['id']}/confirm")
        assert refused.status == 503, refused.text
        assert refused.header("Retry-After") == str(BUSY_RETRY_AFTER_S)
        assert "busy" in refused.json["detail"]
        assert "Nothing was written" in refused.json["detail"]

        # Nothing was written, and the question is still answerable.
        stored = harness.store.get_proposal(asked["id"])
        assert stored is not None and stored.status == "pending"
        assert [p["proposal_id"] for p in harness.client.get("/api/proposals").json["pending"]] == [
            asked["id"]
        ]
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()

    # The lock is gone: the same click a moment later stores the same label.
    retried = harness.client.post(f"/api/proposals/{asked['id']}/confirm")
    assert retried.status == 200, retried.text
    assert retried.json["merged"] is True
    stored = harness.store.get_proposal(asked["id"])
    assert stored is not None and stored.status == "confirmed"


def test_a_busy_store_is_a_503_with_retry_after_on_a_rejection_too(harness: Harness) -> None:
    """The other answer takes the same path, and a rejection is a label like any other."""
    asked = harness.propose_a_return()
    blocker = hold_the_write_lock(harness)
    try:
        refused = harness.client.post(f"/api/proposals/{asked['id']}/reject")
        assert refused.status == 503
        assert refused.header("Retry-After") == str(BUSY_RETRY_AFTER_S)
        stored = harness.store.get_proposal(asked["id"])
        assert stored is not None and stored.status == "pending"
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


def test_a_busy_store_is_a_503_with_retry_after_on_the_feedback_route(harness: Harness) -> None:
    """The other route in this app that writes a person's verdict, held to the same rule."""
    record = AlertRecord(
        alert_id=make_alert_id("posture_still_v0", 1, 1.0),
        track_id=1,
        rule_id="posture_still_v0",
        rule_version="v0",
        onset_ts=1.0,
        detected_ts=2.0,
        created_utc=utc_now_iso(),
        baseline=None,
        details={},
    )
    harness.store.insert_alert(record)

    blocker = hold_the_write_lock(harness)
    try:
        refused = harness.client.post(
            f"/alerts/{record.alert_id}/feedback", {"label": "false_alarm"}
        )
        assert refused.status == 503, refused.text
        assert refused.header("Retry-After") == str(BUSY_RETRY_AFTER_S)
        stored = harness.store.get(record.alert_id)
        assert stored is not None and stored.feedback is None
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()

    assert (
        harness.client.post(f"/alerts/{record.alert_id}/feedback", {"label": "agree"}).status == 200
    )
    stored = harness.store.get(record.alert_id)
    assert stored is not None and stored.feedback == "agree"


def test_deciding_an_unknown_proposal_is_a_404(harness: Harness) -> None:
    missing = make_proposal_id(99, 98, 1.0)
    assert harness.client.post(f"/api/proposals/{missing}/confirm").status == 404
    assert harness.client.post(f"/api/proposals/{missing}/reject").status == 404


def test_a_confirmation_on_a_track_that_has_left_still_stores_the_label(
    harness: Harness,
) -> None:
    """The label survives the person. Losing it because there was nothing to mutate is worse."""
    asked = harness.propose_a_return()
    harness.book.reap(100.0)  # track 2 leaves before anybody gets to the card
    assert 2 not in harness.book.tracks()

    body = harness.client.post(f"/api/proposals/{asked['id']}/confirm").json
    assert body["status"] == "confirmed"
    assert body["merged"] is False
    assert "already been reaped" in body["note"]
    stored = harness.store.get_proposal(asked["id"])
    assert stored is not None and stored.status == "confirmed"


def test_a_card_warns_before_the_click_when_the_new_track_has_left(harness: Harness) -> None:
    harness.propose_a_return()
    harness.book.reap(100.0)
    card = harness.client.get("/api/proposals").json["pending"][0]
    assert card["new_track_live"] is False


def test_the_decision_is_the_route_and_never_a_body_field(harness: Harness) -> None:
    """One endpoint taking {"decision": ...} would put confirm and reject one typo apart."""
    asked = harness.propose_a_return()
    body = harness.client.post(
        f"/api/proposals/{asked['id']}/confirm", {"decision": "rejected", "status": "rejected"}
    ).json
    assert body["decision"] == "confirmed"
    stored = harness.store.get_proposal(asked["id"])
    assert stored is not None and stored.status == "confirmed"


def test_via_is_stored_as_the_provenance_of_the_answer(harness: Harness) -> None:
    asked = harness.propose_a_return()
    harness.client.post(f"/api/proposals/{asked['id']}/confirm", {"via": " dashboard:triage "})
    stored = harness.store.get_proposal(asked["id"])
    assert stored is not None and stored.decided_via == "dashboard:triage"


def test_an_unusable_via_is_a_400_before_anything_is_written(harness: Harness) -> None:
    asked = harness.propose_a_return()
    for bad in ({"via": ""}, {"via": "   "}, {"via": 7}, {"via": "x" * (MAX_VIA_CHARS + 1)}):
        response = harness.client.post(f"/api/proposals/{asked['id']}/confirm", bad)
        assert response.status == 400, bad
    stored = harness.store.get_proposal(asked["id"])
    assert stored is not None and stored.status == "pending"


def test_a_decision_without_a_reid_stage_still_stores_the_label(bare: Harness) -> None:
    """A store that holds the question can hold the answer; there is just nothing to merge."""
    record = bare.store.record_proposal(
        new_track=2, candidate_track=1, confidence=0.8, ts=30.0, gallery_size=1
    )
    body = bare.client.post(f"/api/proposals/{record.proposal_id}/confirm").json
    assert body["status"] == "confirmed"
    assert body["merged"] is False
    assert "no re-ID stage" in body["note"]
    stored = bare.store.get_proposal(record.proposal_id)
    assert stored is not None and stored.status == "confirmed"


def test_decided_ts_is_a_finite_wall_clock_and_is_never_a_source_time(harness: Harness) -> None:
    """The one wall clock in the mechanism, and the reason it is one.

    `ts` is source time and can be tiny under replay; `decided_ts` answers "when did a
    person answer", which is a fact about the review. The store exports the two side by side
    and never subtracts them.
    """
    asked = harness.propose_a_return()
    harness.client.post(f"/api/proposals/{asked['id']}/confirm")
    row = harness.store.labels_as_rows()[0]
    assert row["ts"] == 30.0
    assert row["decided_ts"] is not None
    assert row["decided_ts"] > row["ts"]


# ============================================================= the live socket
def test_the_socket_payload_carries_the_proposal_block(harness: Harness) -> None:
    asked = harness.propose_a_return()
    payload = harness.client.stream("/ws", 1)[0]
    # The socket contract keys are untouched by this addition.
    for key in ("frame_b64", "tracks", "alerts", "timers", "latency_ms"):
        assert key in payload
    block = payload["proposals"]
    assert block["available"] is True
    assert block["enabled"] is True
    assert [p["proposal_id"] for p in block["pending"]] == [asked["id"]]
    assert set(block) == {
        "available",
        "enabled",
        "pending",
        "decided",
        "n_pending_shown",
        "n_decided_shown",
        "limits",
        "crops_available",
        "session",
    }


def test_the_socket_pushes_a_proposal_written_after_the_page_connected(
    harness: Harness,
) -> None:
    """Live, not on a page refresh: the pipeline thread writes and the next push carries it."""

    def write_one(index: int) -> None:
        if index == 0:
            harness.propose_a_return()

    payloads = harness.client.stream("/ws", 2, between=write_one)
    assert payloads[0]["proposals"]["pending"] == []
    assert len(payloads[1]["proposals"]["pending"]) == 1
    assert payloads[1]["proposals"]["pending"][0]["candidate_track"] == 1


def test_the_socket_payload_is_json_a_browser_can_parse(harness: Harness) -> None:
    """No NaN and no Infinity anywhere, including through the gallery's empty stats."""
    harness.propose_a_return()
    payload = harness.client.stream("/ws", 1)[0]
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_a_store_closed_under_the_socket_degrades_one_panel(harness: Harness) -> None:
    """The stream is a push loop; one closed store must not blank the whole page.

    Driven through `snapshot()` rather than the socket because the alert feed reads the same
    closed store one line earlier: the claim under test is that the proposal block reports
    the failure instead of raising through it.
    """
    harness.stage.close()
    harness.store.close()
    block = harness.state._proposals_or_error()
    assert block["available"] is False
    assert block["enabled"] is True
    assert "closed" in block["error"]


# ======================================================== session-end behaviour
def test_history_outlives_the_cleared_gallery(harness: Harness) -> None:
    """CQ-16, from the reviewer's side. The labels persist; the embeddings do not.

    This is the sentence the panel header prints, asserted as behaviour: after session end
    the cards are still served from sqlite, the gallery reports itself cleared, and there is
    no identity left in it.
    """
    asked = harness.propose_a_return()
    harness.client.post(f"/api/proposals/{asked['id']}/reject")

    harness.stage.close()  # session end: SessionGallery.clear()

    body = harness.client.get("/api/proposals").json
    assert [p["proposal_id"] for p in body["decided"]] == [asked["id"]]
    assert body["decided"][0]["status"] == "rejected"
    assert body["session"]["gallery_cleared"] is True
    assert body["session"]["n_identities"] == 0
    assert body["session"]["n_gallery_rows"] == 0
    assert harness.stage.gallery.size() == 0


def test_no_embedding_is_reachable_through_any_route_after_session_end(
    harness: Harness,
) -> None:
    """The rows that survive hold two ids, a score, a count and an answer. Nothing else."""
    harness.propose_a_return()
    harness.stage.close()
    text = harness.client.get("/api/proposals").text
    for banned in ("embedding", "vector", "descriptor", "feature"):
        assert banned not in text.lower(), banned


def test_the_stored_rows_themselves_carry_no_appearance_column(harness: Harness) -> None:
    """Read straight off the file, so the claim does not depend on the serialiser.

    The set is exact and stays exact. A subset check would pass a table that had quietly
    grown an `embedding` column, which is the only thing this test exists to catch, so a new
    column has to be added here by hand and justified when it is.

    `run_id` is the one that was added (store file version 3). It is run provenance, not
    appearance: an opaque label for the run that raised the proposal, so one run cannot
    answer another run's question in a shared store file. It is not derived from a person, a
    visit, a crop or a clip's contents, and nothing about anybody can be read back out of
    it. The rows are still two integer track ids, a score, a count, a run tag and a human's
    answer.
    """
    harness.propose_a_return()
    harness.close()
    conn = sqlite3.connect(harness.store.path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(reassoc_proposals)")}
    finally:
        conn.close()
    assert columns == {
        "proposal_id",
        "schema_version",
        "ts",
        "new_track",
        "candidate_track",
        "confidence",
        "gallery_size",
        "status",
        "decided_ts",
        "decided_via",
        "run_id",
    }


# ================================================================= the crops route
def test_a_zero_footage_run_serves_placeholder_tiles_and_no_crops(harness: Harness) -> None:
    """No pixels exist, so both sides of the card are colour tiles carrying the track id."""
    harness.propose_a_return()
    body = harness.client.get("/api/proposals").json
    assert body["crops_available"] is False
    card = body["pending"][0]
    assert card["new_crop"] is None
    assert card["candidate_crop"] is None
    assert card["new_color"] and card["candidate_color"]
    assert harness.client.get("/api/proposals/crops/track_000001.jpg").status == 404


def test_a_card_names_the_crops_that_exist_on_disk(with_crops: Harness) -> None:
    run_dir = with_crops.state.run_dir
    assert run_dir is not None
    write_crop(run_dir, 1)
    with_crops.propose_a_return()
    body = with_crops.client.get("/api/proposals").json
    assert body["crops_available"] is True
    card = body["pending"][0]
    assert card["candidate_crop"] == crop_name(1, ".png")
    assert card["new_crop"] is None  # nobody has written one for track 2


def test_the_crops_route_serves_a_file_from_the_run_directory(with_crops: Harness) -> None:
    run_dir = with_crops.state.run_dir
    assert run_dir is not None
    write_crop(run_dir, 1)
    response = with_crops.client.get(f"/api/proposals/crops/{crop_name(1, '.png')}")
    assert response.status == 200
    assert response.body == CROP_BYTES


def test_the_crops_route_refuses_a_path_outside_the_run_directory(with_crops: Harness) -> None:
    """The one that matters: this route hands bytes from a directory to a browser."""
    run_dir = with_crops.state.run_dir
    assert run_dir is not None
    outside = run_dir.parent / "secret.jpg"
    outside.write_bytes(b"not for the browser")

    for name in ("../secret.jpg", "../../secret.jpg", f"{outside}"):
        response = with_crops.client.get(f"/api/proposals/crops/{name}")
        assert response.status == 403, name
        assert b"not for the browser" not in response.body


def test_the_crops_route_refuses_a_symlink_that_escapes(with_crops: Harness) -> None:
    run_dir = with_crops.state.run_dir
    assert run_dir is not None
    outside = run_dir.parent / "secret.png"
    outside.write_bytes(b"not for the browser")
    (run_dir / CROPS_DIRNAME / "escape.png").symlink_to(outside)

    response = with_crops.client.get("/api/proposals/crops/escape.png")
    assert response.status == 403
    assert b"not for the browser" not in response.body


def test_the_crops_route_refuses_a_name_that_is_not_an_image(with_crops: Harness) -> None:
    run_dir = with_crops.state.run_dir
    assert run_dir is not None
    (run_dir / CROPS_DIRNAME / "alerts.sqlite").write_bytes(b"sqlite bytes")
    response = with_crops.client.get("/api/proposals/crops/alerts.sqlite")
    assert response.status == 404
    assert b"sqlite bytes" not in response.body


def test_the_crops_route_404s_for_a_crop_that_does_not_exist(with_crops: Harness) -> None:
    assert with_crops.client.get("/api/proposals/crops/track_000042.jpg").status == 404


def test_the_crops_route_refuses_an_oversized_file(with_crops: Harness) -> None:
    """A dashboard must not become a way to read a video off the appliance."""
    run_dir = with_crops.state.run_dir
    assert run_dir is not None
    fat = run_dir / CROPS_DIRNAME / crop_name(1, ".png")
    fat.write_bytes(b"\x00" * (MAX_CROP_BYTES + 1))
    assert with_crops.client.get(f"/api/proposals/crops/{crop_name(1, '.png')}").status == 413


def test_a_served_crop_is_never_cached(with_crops: Harness) -> None:
    """An appearance image outliving the gallery that produced it would defeat CQ-16."""
    run_dir = with_crops.state.run_dir
    assert run_dir is not None
    write_crop(run_dir, 1)
    app = create_app(with_crops.state)
    headers = asyncio.run(_headers(app, f"/api/proposals/crops/{crop_name(1, '.png')}"))
    assert headers[b"cache-control"] == b"no-store"


async def _headers(app: Any, path: str) -> dict[bytes, bytes]:
    received: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        received.append(message)

    await asyncio.wait_for(app(_http_scope("GET", path, 0), receive, send), timeout=10)
    start = next(m for m in received if m["type"] == "http.response.start")
    return {k.lower(): v for k, v in start["headers"]}


# ================================================================== the state seam
def test_a_state_with_no_run_dir_has_no_crops_dir(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.sqlite")
    try:
        state = PipelineState(book=TrackBook(TrackConfig()), store=store)
        assert state.crops_dir is None
        assert state.reid is None
        with_dir = PipelineState(book=TrackBook(TrackConfig()), store=store, run_dir=str(tmp_path))
        assert with_dir.crops_dir == tmp_path / CROPS_DIRNAME
    finally:
        store.close()


def test_display_caps_are_validated_at_construction(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.sqlite")
    try:
        for bad in ({"proposal_limit": 0}, {"decided_limit": -1}):
            with pytest.raises(ValueError):
                PipelineState(book=TrackBook(TrackConfig()), store=store, **bad)
    finally:
        store.close()


def test_a_runs_pending_queue_never_lists_another_runs_question(tmp_path: Path) -> None:
    """One store file, two runs: run B's page must not show run A's card as work.

    The store has had scoped queries since file version 3; this pins that the dashboard
    actually passes its run_id through instead of falling back to the file-wide view.
    """
    store = EventStore(tmp_path / "alerts.sqlite")
    try:
        store.record_proposal(
            new_track=2, candidate_track=1, confidence=0.8, ts=30.0, gallery_size=1, run_id="run-a"
        )
        ours = store.record_proposal(
            new_track=4, candidate_track=3, confidence=0.8, ts=31.0, gallery_size=1, run_id="run-b"
        )
        scoped = PipelineState(book=TrackBook(TrackConfig()), store=store, run_id="run-b")
        pending = scoped.proposals()["pending"]
        assert [card["proposal_id"] for card in pending] == [ours.proposal_id]
        # None keeps the pre-run-scoping file-wide view, which bare harnesses rely on.
        unscoped = PipelineState(book=TrackBook(TrackConfig()), store=store)
        assert len(unscoped.proposals()["pending"]) == 2
    finally:
        store.close()


def test_the_no_stage_fallback_cannot_answer_another_runs_question(tmp_path: Path) -> None:
    """Run-scoping holds on the decide path even when no re-ID stage is attached.

    The stage's own decide path was already scoped; the store-only fallback was the one
    door through which run B could answer run A's question about two patients.
    """
    store = EventStore(tmp_path / "alerts.sqlite")
    try:
        theirs = store.record_proposal(
            new_track=2, candidate_track=1, confidence=0.8, ts=30.0, gallery_size=1, run_id="run-a"
        )
        ours = store.record_proposal(
            new_track=4, candidate_track=3, confidence=0.8, ts=31.0, gallery_size=1, run_id="run-b"
        )
        state = PipelineState(book=TrackBook(TrackConfig()), store=store, run_id="run-b")
        client = AsgiClient(create_app(state))
        assert client.post(f"/api/proposals/{theirs.proposal_id}/confirm").status == 404
        stored = store.get_proposal(theirs.proposal_id)
        assert stored is not None and stored.status == "pending"
        answered = client.post(f"/api/proposals/{ours.proposal_id}/confirm").json
        assert answered["status"] == "confirmed"
    finally:
        store.close()


def test_the_synthetic_embedding_fixture_is_the_answer_key_not_a_measurement() -> None:
    """Guard against anyone reading a green run here as re-ID evidence.

    The two tracks match because the fixture says they are the same synthetic identity. That
    is a statement about arithmetic in `vividly.reid.synth`, not about overhead footage.
    """
    spec = EmbeddingsSpec(source="synthetic", dim=16, identity_by_track=IDENTITY_BY_TRACK)
    assert spec.identity_by_track[1] == spec.identity_by_track[2]
    assert spec.identity_by_track[3] != spec.identity_by_track[1]


# ========================================================================== docs
def test_the_pipeline_doc_no_longer_calls_re_identification_out_of_scope() -> None:
    """The doc has to describe what shipped, not the fence that was lifted."""
    text = PIPELINE_DOC.read_text(encoding="utf-8")
    assert "re-identification of any kind" not in text
    # The signals that stay out inside re-identification, and the register entry that
    # carries the open legal half of what did ship.
    assert "no gait" in text and "no body-shape" in text
    assert "CQ-16" in text


def test_the_pipeline_doc_describes_the_propose_then_confirm_flow() -> None:
    text = PIPELINE_DOC.read_text(encoding="utf-8").lower()
    for needle in (
        "session-scoped",
        "confirm",
        "reject",
        "never auto-binds",
        "labels",
    ):
        assert needle in text, f"docs/PIPELINE.md is missing {needle!r}"
