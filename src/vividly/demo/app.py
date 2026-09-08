"""FastAPI app + single-page UI for the Vividly demo platform.

Same serving discipline as the clinical dashboard (src/vividly/dashboard/app.py): one
inline HTML document, vanilla JS, zero CDN/network assets (the page must load with the
network cable out), docs_url/redoc_url disabled because Swagger loads from a CDN.

Upload is a raw-body POST (?filename=...) streamed to disk: python-multipart is not a
dependency and is not wanted; fetch(url, {body: file}) sends bytes and the stream loop
enforces the size cap without ever holding the file in memory.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from vividly.demo import catalog as demo_catalog
from vividly.demo import media
from vividly.demo.jobs import JobManager, validate_zones
from vividly.schemas.events import utc_now_iso

_ACTIVE = {"queued", "normalizing", "running"}
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{8}$")
WS_PUSH_INTERVAL_S = 1.0 / 6.0

# Preview-frame cache written next to an upload by media.ensure_preview; never the video.
_PREVIEW_FILES = frozenset({"preview.jpg", "preview.json"})


def _upload_video(updir: Path) -> Path | None:
    """The uploaded clip inside incoming/<id>/, skipping the preview cache files."""
    files = sorted(p for p in updir.iterdir() if p.is_file() and p.name not in _PREVIEW_FILES)
    return files[0] if files else None


def _preview_dims(updir: Path) -> tuple[int, int] | None:
    """Dimensions of the cached preview frame, or None when no preview was made."""
    try:
        row = json.loads((updir / "preview.json").read_text())
        return int(row["width"]), int(row["height"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def create_demo_app(manager: JobManager, *, max_upload_mb: int = 2048) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)
    repo_root = manager.repo_root

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse(PAGE_HTML)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            from vividly.media.ffmpeg import ffmpeg_bin

            ffmpeg_bin()
            ffmpeg_ok, ffmpeg_error = True, None
        except RuntimeError as exc:
            ffmpeg_ok, ffmpeg_error = False, str(exc)
        cat = demo_catalog.scan_catalog(repo_root)
        return {
            "ok": True,
            "ffmpeg": ffmpeg_ok,
            "ffmpeg_error": ffmpeg_error,
            "n_detectors": len(cat["detectors"]),
            "n_poses": len(cat["poses"]),
        }

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return manager.catalog()

    @app.post("/api/models/upload")
    async def model_upload(request: Request, kind: str = "", filename: str = "") -> Any:
        """Drop-in .onnx lane: stored content-addressed under demo_uploads/models/<sha8>/,
        input size read from the graph (static shapes only, because a wrong size degrades
        silently, so a dynamic height or width is refused rather than guessed)."""
        if kind not in {"detector", "pose"}:
            return JSONResponse({"error": "kind must be detector or pose"}, status_code=400)
        if not filename.lower().endswith(".onnx"):
            return JSONResponse({"error": "file must be .onnx"}, status_code=400)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", filename[:-5].rsplit("/", 1)[-1]).strip("._")
        stem = stem[:60] or "model"
        import hashlib
        import tempfile

        cap = max_upload_mb * 1024 * 1024
        digest = hashlib.sha256()
        written = 0
        models_dir = manager.dirs["models"]
        with tempfile.NamedTemporaryFile(dir=models_dir, suffix=".part", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            async for chunk in request.stream():
                written += len(chunk)
                if written > cap:
                    tmp.close()
                    tmp_path.unlink(missing_ok=True)
                    return JSONResponse(
                        {"error": f"upload exceeds the {max_upload_mb} MB cap"}, status_code=413
                    )
                digest.update(chunk)
                tmp.write(chunk)
        if written == 0:
            tmp_path.unlink(missing_ok=True)
            return JSONResponse({"error": "empty upload"}, status_code=400)
        sha8 = digest.hexdigest()[:8]
        dest_dir = models_dir / sha8
        dest = dest_dir / f"{stem}.onnx"
        if dest_dir.is_dir() and any(dest_dir.glob("*.onnx")):
            tmp_path.unlink(missing_ok=True)  # same bytes already stored
            return {"id": f"upload/{sha8}", "duplicate": True}
        dest_dir.mkdir(exist_ok=True)
        tmp_path.rename(dest)
        try:
            input_size = demo_catalog.infer_onnx_input_size(dest)
        except Exception as exc:
            shutil.rmtree(dest_dir, ignore_errors=True)
            return JSONResponse({"error": f"cannot use this model: {exc}"}, status_code=400)
        (dest_dir / "meta.json").write_text(
            json.dumps(
                {
                    "kind": kind,
                    "display_name": stem,
                    "input_size": list(input_size),
                    "sha256": digest.hexdigest(),
                    "uploaded_utc": utc_now_iso(),
                },
                indent=2,
            )
            + "\n"
        )
        return {"id": f"upload/{sha8}", "input_size": list(input_size)}

    @app.post("/api/upload")
    async def upload(request: Request, filename: str = "") -> Any:
        try:
            safe = media.sanitize_filename(filename)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        upload_id = uuid.uuid4().hex[:8]
        dest_dir = manager.dirs["incoming"] / upload_id
        dest_dir.mkdir(parents=True, exist_ok=False)
        dest = dest_dir / safe
        cap = max_upload_mb * 1024 * 1024
        written = 0
        try:
            with dest.open("wb") as fh:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > cap:
                        raise _UploadTooLarge
                    fh.write(chunk)
        except _UploadTooLarge:
            shutil.rmtree(dest_dir, ignore_errors=True)
            return JSONResponse(
                {"error": f"upload exceeds the {max_upload_mb} MB cap"}, status_code=413
            )
        except Exception:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise
        if written == 0:
            shutil.rmtree(dest_dir, ignore_errors=True)
            return JSONResponse({"error": "empty upload"}, status_code=400)
        return {"upload_id": upload_id, "filename": safe, "bytes": written}

    @app.get("/api/uploads/{upload_id}/preview.jpg")
    def upload_preview(upload_id: str) -> Any:
        """One frame of the upload in the normalized pixel space, for the exit-zone tool.

        Extracted lazily and cached next to the upload; zone coordinates drawn on this
        frame are validated (and later processed) in exactly this frame's geometry.
        """
        updir = manager.dirs["incoming"] / upload_id
        if not _UPLOAD_ID_RE.fullmatch(upload_id) or not updir.is_dir():
            return JSONResponse({"error": "unknown upload_id"}, status_code=404)
        src = _upload_video(updir)
        if src is None:
            return JSONResponse({"error": "upload has no file"}, status_code=404)
        try:
            media.ensure_preview(src, updir)
        except (RuntimeError, ValueError, OSError) as exc:
            return JSONResponse(
                {"error": f"could not extract a preview frame: {exc}"}, status_code=422
            )
        return FileResponse(updir / "preview.jpg", media_type="image/jpeg")

    @app.post("/api/jobs")
    async def create_job(request: Request) -> Any:
        try:
            payload = json.loads((await request.body()) or b"{}")
        except ValueError:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)
        try:
            if payload.get("live"):
                if payload.get("compare_detector_id"):
                    return JSONResponse(
                        {"error": "live sessions cannot run a comparison"}, status_code=400
                    )
                if payload.get("zones"):
                    return JSONResponse(
                        {"error": "live sessions cannot use exit zones"}, status_code=400
                    )
                max_s = payload.get("max_s")
                job = manager.create_live_job(
                    detector_id=str(payload.get("detector_id") or ""),
                    pose_id=str(payload.get("pose_id") or ""),
                    score_thr=float(payload.get("score_thr", 0.4)),
                    rules_profile=str(payload.get("rules_profile") or "demo"),
                    max_s=float(max_s) if max_s not in (None, "") else None,
                )
                return {"job_id": job.job_id, "live": True}
            zones: list[dict[str, Any]] = []
            if payload.get("sample"):
                if payload.get("zones"):
                    # the sample lane skips upload, so there is no preview frame to
                    # validate zone bounds against; the UI never offers this pairing
                    return JSONResponse(
                        {"error": "exit zones need an uploaded clip"}, status_code=400
                    )
                src, label, move = manager.sample_clip, "sample_clip.mp4", False
            else:
                upload_id = str(payload.get("upload_id") or "")
                updir = manager.dirs["incoming"] / upload_id
                if not _UPLOAD_ID_RE.fullmatch(upload_id) or not updir.is_dir():
                    return JSONResponse({"error": "unknown upload_id"}, status_code=400)
                video = _upload_video(updir)
                if video is None:
                    return JSONResponse({"error": "upload has no file"}, status_code=400)
                if payload.get("zones"):
                    dims = _preview_dims(updir)
                    if dims is None:
                        # bounds can only be checked against the server's own record of
                        # the frame the zones were drawn on; without it, no zones
                        return JSONResponse(
                            {"error": "no preview frame for this upload; reload and redraw"},
                            status_code=400,
                        )
                    zones = validate_zones(payload["zones"], *dims)
                src, label, move = video, video.name, True
            max_s = payload.get("max_s")
            compare_id = str(payload.get("compare_detector_id") or "")
            common = {
                "pose_id": str(payload.get("pose_id") or ""),
                "score_thr": float(payload.get("score_thr", 0.4)),
                "rules_profile": str(payload.get("rules_profile") or "demo"),
                "max_s": float(max_s) if max_s not in (None, "") else None,
                "zones": zones,
            }
            job = manager.create_job(
                source_path=src,
                source_label=label,
                move_source=False if compare_id else move,  # side B still needs the file
                detector_id=str(payload.get("detector_id") or ""),
                **common,
            )
            job_b = None
            if compare_id:
                try:
                    job_b = manager.create_job(
                        source_path=src,
                        source_label=label,
                        move_source=move,
                        detector_id=compare_id,
                        **common,
                    )
                except ValueError:
                    # Side B was refused after side A had already been created, queued
                    # and persisted. Cancel A rather than leaving it to run as an orphan
                    # of a request the caller was told had failed.
                    manager.cancel(job.job_id)
                    raise
                manager.link_compare(job.job_id, job_b.job_id)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if move:
            shutil.rmtree(src.parent, ignore_errors=True)  # empty incoming/<id> shell
        return {"job_id": job.job_id, "compare_job_id": job_b.job_id if job_b else None}

    @app.get("/api/jobs")
    def jobs_list() -> dict[str, Any]:
        return {"jobs": manager.list_snapshots()}

    @app.get("/api/jobs/{job_id}")
    def job_get(job_id: str) -> Any:
        job = manager.get(job_id)
        if job is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return job.snapshot(queue_position=manager.queue_position(job_id))

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str) -> Any:
        ok = manager.cancel(job_id)
        return {"ok": ok} if ok else JSONResponse({"error": "not cancellable"}, status_code=409)

    @app.post("/api/jobs/{job_id}/stop")
    def job_stop(job_id: str) -> Any:
        """End a live session; stopping is its normal ending (final status: done)."""
        ok = manager.stop_live(job_id)
        return {"ok": ok} if ok else JSONResponse({"error": "not a live session"}, status_code=409)

    @app.post("/api/jobs/{job_id}/delete")
    def job_delete(job_id: str) -> Any:
        ok = manager.delete(job_id)
        return {"ok": ok} if ok else JSONResponse({"error": "not deletable"}, status_code=409)

    @app.get("/api/jobs/{job_id}/annotated.mp4")
    def job_video(job_id: str) -> Any:
        job = manager.get(job_id)
        path = job.job_dir / "annotated.mp4" if job else None
        if path is None or not path.is_file():
            return JSONResponse({"error": "no annotated video"}, status_code=404)
        return FileResponse(path, media_type="video/mp4")  # Range-capable: <video> can seek

    @app.get("/api/jobs/{job_id}/thumb.jpg")
    def job_thumb(job_id: str) -> Any:
        job = manager.get(job_id)
        path = job.job_dir / "thumb.jpg" if job else None
        if path is None or not path.is_file():
            return JSONResponse({"error": "no thumbnail"}, status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    @app.websocket("/ws/jobs/{job_id}")
    async def ws_job(ws: WebSocket, job_id: str) -> None:
        await ws.accept()
        job = manager.get(job_id)
        if job is None:
            await ws.close(code=4404)
            return
        try:
            while True:
                snap = job.snapshot(queue_position=manager.queue_position(job_id))
                await ws.send_json(snap)
                if snap["status"] not in _ACTIVE:
                    break
                await asyncio.sleep(WS_PUSH_INTERVAL_S)
            await ws.close()
        except WebSocketDisconnect:
            return

    @app.websocket("/ws/jobs/{job_id}/frames")
    async def ws_frames(ws: WebSocket, job_id: str) -> None:
        """Browser -> server camera frames (binary JPEG per message). The bytes go into
        the session's 1-deep mailbox and never anywhere else."""
        await ws.accept()
        job = manager.get(job_id)
        if job is None or not job.live:
            await ws.close(code=4404)
            return
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if data is None:
                    continue  # text frames carry no camera data; ignore them
                if not manager.push_live_frame(job_id, data):
                    break
            await ws.close()
        except WebSocketDisconnect:
            return

    return app


class _UploadTooLarge(Exception):
    pass


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vividly - Model Demo</title>
<style>
  :root {
    --bg: #ffffff; --ink: #000000;
    --serif: "Tiempos Text", Charter, Georgia, "Times New Roman", serif;
  }
  * { box-sizing: border-box; margin: 0; border-radius: 0 !important; }
  body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--serif); }
  code, .mono { font-family: var(--serif); font-size: 12.5px; }
  header {
    display: flex; align-items: center; gap: 14px; padding: 14px 36px;
    border-bottom: 1px solid var(--ink);
  }
  header img.mark { width: 34px; height: 34px; display: block; }
  header h1 { font: 600 21px/1 var(--serif); letter-spacing: -0.01em; }
  header .sep { width: 1px; height: 22px; background: var(--ink); }
  header .app { font: 600 11px/1 var(--serif); text-transform: uppercase;
                letter-spacing: .16em; }
  .pill { font: 600 11px/1.4 var(--serif); border: 1px solid var(--ink); padding: 3px 10px; }
  main { max-width: 1520px; margin: 0 auto; padding: 28px 36px; }
  .card { border: 1px solid var(--ink); padding: 24px; margin-bottom: 20px;
          background: var(--bg); }
  .card h2 { font: 700 12px/1 var(--serif); text-transform: uppercase;
             letter-spacing: .14em; margin-bottom: 16px; }
  .sec { font: 700 10px/1 var(--serif); text-transform: uppercase; letter-spacing: .16em;
         margin: 22px 0 10px; }
  .drop { border: 1.5px dashed var(--ink); padding: 30px; text-align: center;
          cursor: pointer; }
  .drop.armed { border: 2px solid var(--ink); }
  .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
         gap: 16px; }
  .field { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  label { font: 600 10.5px/1.4 var(--serif); text-transform: uppercase;
          letter-spacing: .12em; }
  select, input[type=number], input[type=text] {
    background: var(--bg); color: var(--ink); border: 1px solid var(--ink);
    padding: 8px 10px; font: 13px/1.4 var(--serif); width: 100%;
  }
  .zonebox { border: 1px solid var(--ink); background: #000; max-width: 640px; }
  #zonecanvas { display: block; width: 100%; cursor: crosshair; touch-action: none; }
  .zonectl { display: flex; gap: 14px; align-items: flex-end; margin-top: 10px;
             flex-wrap: wrap; }
  ::placeholder { color: var(--ink); opacity: 1; font-style: italic; }
  input[type=range] { -webkit-appearance: none; appearance: none; width: 100%;
    height: 3px; background: var(--ink); margin: 11px 0; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none;
    width: 16px; height: 16px; background: var(--ink); border-radius: 0; }
  input[type=range]::-moz-range-thumb { width: 16px; height: 16px;
    background: var(--ink); border: 0; border-radius: 0; }
  button {
    background: var(--ink); color: #ffffff; border: 1px solid var(--ink);
    padding: 9px 18px; font: 700 11.5px/1.4 var(--serif); letter-spacing: .06em;
    text-transform: uppercase; cursor: pointer; white-space: nowrap;
  }
  button.primary { background: var(--ink); border-color: var(--ink);
                   padding: 10px 30px; }
  button.ghost { background: var(--bg); color: var(--ink); font-weight: 600; }
  button:disabled { opacity: .3; cursor: default; }
  .actions { display: flex; gap: 10px; margin-top: 24px; align-items: center; }
  .modelnote { font: 12px/1.5 var(--serif); margin-top: 8px; }
  .badge { font: 700 10px/1.4 var(--serif); padding: 1px 7px; letter-spacing: .06em;
           border: 1px solid var(--ink); }
  .badge.prototype { background: var(--ink); color: var(--bg); }
  .badge.vividly { color: var(--ink); border-color: var(--ink); }
  .badge.upload { background: var(--bg); color: var(--ink); border-style: dashed; }
  .runrow {
    display: flex; align-items: center; gap: 16px; padding: 12px 6px;
    border-top: 1px solid var(--ink); cursor: pointer;
  }
  .runrow:hover { background: rgba(0, 0, 0, .04); }
  .runrow img { width: 74px; height: 46px; object-fit: cover;
                background: #000; }
  .runrow .grow { flex: 1; min-width: 0; }
  .runrow .title { font: 600 14px/1.5 var(--serif); }
  .runrow .sub { font: 12px/1.5 var(--serif); }
  .chip { font: 700 10.5px/1.4 var(--serif); padding: 3px 9px; letter-spacing: .06em;
          text-transform: uppercase; border: 1px solid var(--ink); }
  .chip.done { color: var(--ink); background: var(--bg); }
  .chip.running, .chip.normalizing, .chip.queued { color: #ffffff;
      background: var(--ink); border-color: var(--ink); }
  .chip.failed, .chip.interrupted { color: var(--bg); background: var(--ink); }
  .chip.cancelled { color: var(--ink); background: var(--bg); }
  /* job view */
  .jobgrid { display: grid; grid-template-columns: 1fr 340px; gap: 18px; }
  @media (max-width: 900px) { .jobgrid { grid-template-columns: 1fr; } }
  .viewer { background: #000; border: 1px solid var(--ink); overflow: hidden;
            text-align: center; }
  .viewer img, .viewer video { max-width: 100%; max-height: 66vh; display: block;
                               margin: 0 auto; }
  .viewer .empty { padding: 60px 20px; color: #ffffff; font: 13px/1.5 var(--serif); }
  .statusstrip { display: flex; align-items: center; gap: 14px; margin-bottom: 14px;
                 flex-wrap: wrap; }
  .bar { flex: 1; min-width: 160px; height: 6px; background: var(--bg);
         border: 1px solid var(--ink); overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; background: var(--ink);
           transition: width .3s; }
  .timeline { position: relative; height: 18px; background: var(--bg);
              border: 1px solid var(--ink); margin-top: 10px; }
  .tick { position: absolute; top: 0; width: 3px; height: 100%;
          background: var(--ink); cursor: pointer; }
  table { width: 100%; border-collapse: collapse; font: 12px/1.5 var(--serif); }
  td, th { text-align: right; padding: 4px 6px; border-top: 1px solid var(--ink); }
  td:first-child, th:first-child { text-align: left; }
  th { font-weight: 700; }
  .alertrow { padding: 8px 6px; border-top: 1px solid var(--ink); font-size: 12px;
              cursor: pointer; }
  .alertrow:hover { background: rgba(0, 0, 0, .04); }
  .alertrow .rid { font: 600 13.5px/1.5 var(--serif); }
  .addonnx { align-self: flex-start; padding: 4px 10px; font-size: 10px; margin-top: 6px; }
  .cmpgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 900px) { .cmpgrid { grid-template-columns: 1fr; } }
  .cmpgrid .viewer img, .cmpgrid .viewer video { max-height: 48vh; }
  .hidden { display: none !important; }
  .err { font: 650 12px/1.5 var(--serif); margin-top: 8px; white-space: pre-wrap; }
  .note { font: 12px/1.5 var(--serif); }
  a { color: inherit; text-decoration: underline; }
</style>
</head>
<body>
<header>
  <img class="mark" src="__LOGO__" alt="" onerror="this.style.display='none'">
  <h1>Vividly</h1>
  <span class="sep"></span>
  <span class="app">Model demo</span>
  <span class="pill hidden" id="ffmpegwarn">ffmpeg missing</span>
</header>

<main id="view-home">
  <section class="card">
    <h2>New run</h2>
    <div class="drop" id="drop">
      <div id="droptext">Drop a video, or click to choose
        <div class="note">MP4 &middot; MOV &middot; M4V &middot; AVI &middot; MKV
          &middot; WEBM</div>
      </div>
      <input type="file" id="file" class="hidden"
             accept=".mp4,.mov,.m4v,.avi,.mkv,.webm,video/*">
    </div>
    <div id="zonewrap" class="hidden">
      <div class="sec">Exit zone</div>
      <div class="note">Optional. Click and drag on the preview to mark the door region
        and arm Egress without discharge
        (<span class="mono">egress_without_discharge_v0</span>): a person last seen
        inside the zone who stays gone becomes a candidate event. Under demo calibration
        it fires about 20 seconds after the person is last seen, so the clip should keep
        rolling past the exit. Leave the preview unmarked to run without it.</div>
      <div class="zonebox" style="margin-top:10px">
        <canvas id="zonecanvas" width="0" height="0"></canvas>
      </div>
      <div class="zonectl">
        <div class="field" style="max-width:220px">
          <label>Zone name</label>
          <input type="text" id="zonename" value="exit" maxlength="40">
        </div>
        <button class="ghost" id="zoneclear" disabled>Clear zone</button>
        <span class="note mono" id="zoneinfo"></span>
      </div>
    </div>
    <div class="sec">Models</div>
    <div class="row">
      <div class="field">
        <label>Detector</label>
        <select id="det"></select>
        <div class="modelnote" id="detnote"></div>
        <button class="ghost addonnx" data-kind="detector">+ add .onnx</button>
      </div>
      <div class="field">
        <label>Pose</label>
        <select id="pose"></select>
        <div class="modelnote" id="posenote"></div>
        <button class="ghost addonnx" data-kind="pose">+ add .onnx</button>
      </div>
      <input type="file" id="onnxfile" class="hidden" accept=".onnx">
    </div>
    <div class="sec">Settings</div>
    <div class="row">
      <div class="field">
        <label>Score threshold <span class="mono" id="thrval">0.40</span></label>
        <input type="range" id="thr" min="0.05" max="0.8" step="0.05" value="0.4">
      </div>
      <div class="field">
        <label>Compare against</label>
        <select id="cmpdet"></select>
      </div>
      <div class="field">
        <label>Rules</label>
        <select id="rules">
          <option value="demo">Demo calibration</option>
          <option value="production">Production</option>
          <option value="hide">Hide alerts</option>
        </select>
      </div>
      <div class="field">
        <label>First N seconds</label>
        <input type="number" id="maxs" min="1" max="3600" placeholder="whole clip">
      </div>
    </div>
    <div class="actions">
      <button id="run" class="primary" disabled>Run</button>
      <button class="ghost" id="sample">Sample clip</button>
      <button class="ghost" id="livebtn">Live camera</button>
    </div>
    <div class="err hidden" id="newerr"></div>
    <div class="note hidden" id="upnote"></div>
  </section>
  <section class="card">
    <h2>Runs</h2>
    <div id="runs"><div class="note">No runs yet.</div></div>
  </section>
</main>

<main id="view-job" class="hidden">
  <section class="card">
    <div class="statusstrip">
      <button class="ghost" onclick="location.hash=''">&#8592; All runs</button>
      <span class="chip" id="jstatus">-</span>
      <span class="note" id="jmeta"></span>
      <div class="bar"><i id="jbar"></i></div>
      <span class="note mono" id="jprog"></span>
      <button class="ghost hidden" id="jcancel">Cancel</button>
    </div>
    <div class="jobgrid">
      <div>
        <div class="viewer">
          <img id="jframe" class="hidden" alt="live frame">
          <video id="jvideo" class="hidden" controls playsinline></video>
          <div id="jempty" class="empty">waiting for frames...</div>
        </div>
        <div class="timeline hidden" id="jtimeline"></div>
        <div class="note" id="jdownload" style="margin-top:8px"></div>
      </div>
      <div>
        <div class="card" style="margin-bottom:12px">
          <h2>Models</h2>
          <div id="jmodels"></div>
        </div>
        <div class="card" style="margin-bottom:12px">
          <h2>Latency (this run)</h2>
          <div id="jlatency" class="note">pending</div>
        </div>
        <div class="card">
          <h2>Candidate events</h2>
          <div id="jalerts" class="note">none</div>
        </div>
      </div>
    </div>
    <div class="err hidden" id="jerr"></div>
  </section>
</main>

<main id="view-cmp" class="hidden">
  <section class="card">
    <div class="statusstrip">
      <button class="ghost" onclick="location.hash=''">&#8592; All runs</button>
      <span class="note" id="cmpmeta"></span>
      <button class="ghost hidden" id="cmpplay">Play both</button>
    </div>
    <div class="cmpgrid">
      <div id="cmpA"></div>
      <div id="cmpB"></div>
    </div>
  </section>
</main>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const state = { models: null, uploadId: null, uploadName: null, ws: null, job: null,
                pollTimer: null };

// ---------- catalog ----------
function optLabel(e) {
  let label = e.display_name + "  \\u00b7 " + e.sha8;
  if (e.badge === "prototype") label += "  \\u00b7 PROTOTYPE";
  if (e.disabled) label += "  (unavailable)";
  return label;
}
function fillPicker(sel, entries, prev) {
  sel.innerHTML = "";
  for (const e of entries) {
    const o = document.createElement("option");
    o.value = e.id; o.textContent = optLabel(e); o.disabled = !!e.disabled;
    sel.appendChild(o);
  }
  if (prev && entries.some(e => e.id === prev && !e.disabled)) sel.value = prev;
  else { const first = entries.find(e => !e.disabled); if (first) sel.value = first.id; }
}
function fillCmpPicker(sel, entries, prev) {
  sel.innerHTML = "";
  const none = document.createElement("option");
  none.value = ""; none.textContent = "No comparison";
  sel.appendChild(none);
  for (const e of entries) {
    const o = document.createElement("option");
    o.value = e.id; o.textContent = optLabel(e); o.disabled = !!e.disabled;
    sel.appendChild(o);
  }
  if (prev && entries.some(e => e.id === prev && !e.disabled)) sel.value = prev;
}
function noteFor(e) {
  if (!e) return "";
  const b = `<span class="badge ${e.badge}">${e.badge.toUpperCase()}</span>`;
  return `${b} <span class="mono">${e.input_size[0]}\\u00d7${e.input_size[1]}</span>`;
}
function entryById(id) {
  if (!state.models) return null;
  return [...state.models.detectors, ...state.models.poses].find(e => e.id === id) || null;
}
function refreshNotes() {
  $("detnote").innerHTML = noteFor(entryById($("det").value));
  $("posenote").innerHTML = noteFor(entryById($("pose").value));
}
async function loadModels() {
  try {
    const r = await fetch("/api/models"); if (!r.ok) return;
    state.models = await r.json();
    fillPicker($("det"), state.models.detectors, $("det").value);
    fillPicker($("pose"), state.models.poses, $("pose").value);
    fillCmpPicker($("cmpdet"), state.models.detectors, $("cmpdet").value);
    refreshNotes();
  } catch (e) { /* offline moment: keep the last catalog */ }
}
$("det").addEventListener("change", refreshNotes);
$("pose").addEventListener("change", refreshNotes);
setInterval(loadModels, 10000);
window.addEventListener("focus", loadModels);

// ---------- upload ----------
const drop = $("drop"), fileInput = $("file");
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", (ev) => { ev.preventDefault(); drop.classList.add("armed"); });
drop.addEventListener("dragleave", () => drop.classList.remove("armed"));
drop.addEventListener("drop", (ev) => {
  ev.preventDefault(); drop.classList.remove("armed");
  if (ev.dataTransfer.files.length) beginUpload(ev.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) beginUpload(fileInput.files[0]);
});
async function beginUpload(file) {
  hide("newerr"); state.uploadId = null; $("run").disabled = true; resetZone();
  show("upnote"); $("upnote").textContent = `Uploading ${file.name} ...`;
  try {
    const r = await fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`,
                          { method: "POST", body: file });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error || r.status);
    state.uploadId = body.upload_id; state.uploadName = body.filename;
    $("upnote").textContent =
      `Ready: ${body.filename} (${(body.bytes / 1048576).toFixed(1)} MB)`;
    $("droptext").firstChild.textContent = body.filename;
    $("run").disabled = false;
    loadZonePreview(body.upload_id);
  } catch (err) {
    hide("upnote"); showErr("newerr", "Upload failed: " + err.message);
  }
}

// ---------- exit zone drawing ----------
// The canvas shows a frame of the normalized video (the server's preview grab), so the
// committed rectangle is already in the exact pixel space the pipeline processes; the
// canvas backing store is that frame's resolution and CSS only scales the display.
const zone = { img: null, rect: null, drag: null, uploadId: null };
const zcv = $("zonecanvas");
function zoneName() { return ($("zonename").value.trim() || "exit").slice(0, 40); }
function resetZone() {
  zone.img = null; zone.rect = null; zone.drag = null; zone.uploadId = null;
  hide("zonewrap");
}
function loadZonePreview(uploadId) {
  resetZone();
  zone.uploadId = uploadId;
  const img = new Image();
  img.onload = () => {
    if (zone.uploadId !== uploadId) return;  // superseded by a newer upload
    zone.img = img;
    zcv.width = img.naturalWidth; zcv.height = img.naturalHeight;
    show("zonewrap"); drawZone(); zoneCtl();
  };
  img.onerror = () => { /* no preview frame: the zone tool simply stays hidden */ };
  img.src = `/api/uploads/${uploadId}/preview.jpg`;
}
function orderedRect(a, b) {
  return [Math.round(Math.min(a[0], b[0])), Math.round(Math.min(a[1], b[1])),
          Math.round(Math.max(a[0], b[0])), Math.round(Math.max(a[1], b[1]))];
}
function drawZone(liveRect) {
  if (!zone.img) return;
  const ctx = zcv.getContext("2d");
  ctx.drawImage(zone.img, 0, 0);
  const r = liveRect || zone.rect;
  if (!r) return;
  const [x1, y1, x2, y2] = r;
  // monochrome overlay rule: white dashed outline over a black halo, nothing else
  ctx.setLineDash([]);
  ctx.lineWidth = Math.max(3, zcv.width / 320);
  ctx.strokeStyle = "#000000";
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.lineWidth = Math.max(1.5, zcv.width / 800);
  ctx.strokeStyle = "#ffffff";
  ctx.setLineDash([10, 7]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
  const fs = Math.max(13, Math.round(zcv.width / 55));
  ctx.font = fs + 'px "Tiempos Text", Charter, Georgia, serif';
  const label = zoneName();
  const tw = ctx.measureText(label).width;
  const pad = Math.round(fs / 2.6);
  const ly = Math.max(0, y1 - fs - pad * 2);
  ctx.fillStyle = "#000000";
  ctx.fillRect(x1, ly, tw + pad * 2, fs + pad * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, x1 + pad, ly + fs + Math.round(pad / 2) - 2);
}
function zoneCtl() {
  $("zoneclear").disabled = !zone.rect;
  $("zoneinfo").textContent = zone.rect
    ? `${zone.rect[2] - zone.rect[0]}\\u00d7${zone.rect[3] - zone.rect[1]} px` +
      ` at (${zone.rect[0]}, ${zone.rect[1]})`
    : "no zone drawn";
}
function zonePoint(ev) {
  const b = zcv.getBoundingClientRect();
  const x = (ev.clientX - b.left) * zcv.width / b.width;
  const y = (ev.clientY - b.top) * zcv.height / b.height;
  return [Math.min(Math.max(x, 0), zcv.width), Math.min(Math.max(y, 0), zcv.height)];
}
zcv.addEventListener("pointerdown", (ev) => {
  if (!zone.img) return;
  ev.preventDefault();
  zcv.setPointerCapture(ev.pointerId);
  zone.drag = zonePoint(ev);
});
zcv.addEventListener("pointermove", (ev) => {
  if (!zone.drag) return;
  drawZone(orderedRect(zone.drag, zonePoint(ev)));
});
zcv.addEventListener("pointerup", (ev) => {
  if (!zone.drag) return;
  const r = orderedRect(zone.drag, zonePoint(ev));
  zone.drag = null;
  if (r[2] - r[0] >= 4 && r[3] - r[1] >= 4) zone.rect = r;  // a click is not a zone
  drawZone(); zoneCtl();
});
zcv.addEventListener("pointercancel", () => { zone.drag = null; drawZone(); });
$("zoneclear").addEventListener("click", () => { zone.rect = null; drawZone(); zoneCtl(); });
$("zonename").addEventListener("input", () => drawZone());

// ---------- drop-in .onnx models ----------
let onnxKind = null;
for (const btn of document.querySelectorAll(".addonnx")) {
  btn.addEventListener("click", () => { onnxKind = btn.dataset.kind; $("onnxfile").click(); });
}
$("onnxfile").addEventListener("change", async () => {
  const file = $("onnxfile").files[0];
  $("onnxfile").value = "";
  if (!file || !onnxKind) return;
  hide("newerr"); show("upnote");
  $("upnote").textContent = `Adding ${file.name} as an unverified ${onnxKind} ...`;
  try {
    const r = await fetch(`/api/models/upload?kind=${onnxKind}` +
                          `&filename=${encodeURIComponent(file.name)}`,
                          { method: "POST", body: file });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error || r.status);
    await loadModels();
    const sel = onnxKind === "detector" ? $("det") : $("pose");
    sel.value = body.id; refreshNotes();
    $("upnote").textContent = `Added ${file.name} (unverified upload, internal only)`;
  } catch (err) { hide("upnote"); showErr("newerr", "Model upload failed: " + err.message); }
});

// ---------- create job ----------
$("thr").addEventListener("input", () => {
  $("thrval").textContent = Number($("thr").value).toFixed(2);
});
async function createJob(payload) {
  hide("newerr");
  payload.detector_id = $("det").value;
  payload.pose_id = $("pose").value;
  payload.score_thr = Number($("thr").value);
  payload.rules_profile = $("rules").value;
  const maxs = $("maxs").value;
  if (maxs) payload.max_s = Number(maxs);
  const cmpSel = $("cmpdet").value;
  if (!payload.live && cmpSel && cmpSel !== payload.detector_id) {
    payload.compare_detector_id = cmpSel;
  }
  if (payload.upload_id && zone.rect) {
    payload.zones = [{ name: zoneName(), rect: zone.rect }];
  }
  try {
    const r = await fetch("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error || r.status);
    state.uploadId = null; $("run").disabled = true; resetZone();
    location.hash = body.compare_job_id
      ? `#cmp/${body.job_id}/${body.compare_job_id}` : "#job/" + body.job_id;
  } catch (err) { showErr("newerr", "Could not start: " + err.message); }
}
$("run").addEventListener("click", () => {
  if (state.uploadId) createJob({ upload_id: state.uploadId });
});
$("sample").addEventListener("click", () => createJob({ sample: true }));
$("livebtn").addEventListener("click", () => createJob({ live: true }));

// ---------- live camera capture ----------
// Frames go browser -> /ws/jobs/<id>/frames and exist server-side for one pipeline
// tick; nothing from the camera is ever written to disk, because the project never
// records its own footage.
const cam = { jobId: null, stream: null, ws: null, timer: null, video: null, canvas: null };
async function startCamera(jobId) {
  if (cam.jobId === jobId) return;
  stopCamera();
  cam.jobId = jobId;
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia(
      { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (err) {
    cam.jobId = null;
    showErr("jerr", "Camera unavailable: " + err.message);
    fetch(`/api/jobs/${jobId}/stop`, { method: "POST" });
    return;
  }
  cam.video = document.createElement("video");
  cam.video.muted = true; cam.video.playsInline = true;
  cam.video.srcObject = cam.stream;
  await cam.video.play().catch(() => {});
  cam.canvas = document.createElement("canvas");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  cam.ws = new WebSocket(`${proto}:\\/\\/${location.host}/ws/jobs/${jobId}/frames`);
  cam.ws.binaryType = "arraybuffer";
  cam.timer = setInterval(() => {
    const v = cam.video, ws = cam.ws;
    if (!ws || ws.readyState !== 1 || ws.bufferedAmount > (1 << 20)) return;
    if (!v || v.readyState < 2 || !v.videoWidth) return;
    const scale = Math.min(1, 960 / Math.max(v.videoWidth, v.videoHeight));
    cam.canvas.width = Math.round(v.videoWidth * scale / 2) * 2;
    cam.canvas.height = Math.round(v.videoHeight * scale / 2) * 2;
    cam.canvas.getContext("2d").drawImage(v, 0, 0, cam.canvas.width, cam.canvas.height);
    cam.canvas.toBlob(b => {
      if (b && cam.ws === ws && ws.readyState === 1) {
        b.arrayBuffer().then(buf => {
          if (cam.ws === ws && ws.readyState === 1) ws.send(buf);
        });
      }
    }, "image/jpeg", 0.7);
  }, 100);
}
function stopCamera() {
  clearInterval(cam.timer); cam.timer = null;
  if (cam.ws) { try { cam.ws.close(); } catch (e) { /* already closed */ } cam.ws = null; }
  if (cam.stream) { cam.stream.getTracks().forEach(t => t.stop()); cam.stream = null; }
  cam.video = null; cam.canvas = null; cam.jobId = null;
}

// ---------- runs list ----------
function chip(status) { return `<span class="chip ${status}">${status}</span>`; }
function runRow(j) {
  const det = j.models.detector || {}, pose = j.models.pose || {};
  const alerts = (j.alerts || []).length;
  const thumb = j.annotated_ready || j.status === "done"
    ? `<img src="/api/jobs/${j.job_id}/thumb.jpg" onerror="this.style.visibility='hidden'">`
    : `<img style="visibility:hidden">`;
  return `<div class="runrow" onclick="location.hash='#job/${j.job_id}'">
    ${thumb}
    <div class="grow">
      <div class="title">${j.source_label || j.job_id}</div>
      <div class="sub">${det.display_name || "?"} <span class="mono">${det.sha8 || ""}</span>
        + ${pose.display_name || "?"} <span class="mono">${pose.sha8 || ""}</span></div>
    </div>
    <span class="note">${alerts ? alerts + " event" + (alerts > 1 ? "s" : "") : ""}</span>
    ${chip(j.status)}
    <button class="ghost" onclick="event.stopPropagation();deleteJob('${j.job_id}')">
      Delete</button>
  </div>`;
}
async function refreshRuns() {
  try {
    const r = await fetch("/api/jobs"); if (!r.ok) return;
    const body = await r.json();
    const jobs = body.jobs || [];
    $("runs").innerHTML = jobs.length
      ? jobs.map(runRow).join("")
      : '<div class="note">No runs yet.</div>';
    const active = jobs.some(j => ["queued", "normalizing", "running"].includes(j.status));
    clearTimeout(state.pollTimer);
    if (location.hash === "" || location.hash === "#home") {
      state.pollTimer = setTimeout(refreshRuns, active ? 2000 : 10000);
    }
  } catch (e) { /* retry on next cycle */ }
}
async function deleteJob(id) {
  if (!confirm("Delete this run and its files?")) return;
  await fetch(`/api/jobs/${id}/delete`, { method: "POST" });
  refreshRuns();
}

// ---------- job view ----------
// The three event models (EVENT_MODEL_NAMES in src/vividly/demo/jobs.py); ids stay alongside.
const EVENT_NAMES = {
  fall_then_still_v0: "Collapse to floor",
  seizure_candidate_v0: "Rhythmic convulsion",
  egress_without_discharge_v0: "Egress without discharge",
};
function eventName(ruleId) { return EVENT_NAMES[ruleId] || ruleId; }
function fmtS(s) { return s == null ? "?" : (s >= 60 ? Math.round(s / 60) + " min" : Math.round(s) + " s"); }
function renderJob(j) {
  state.job = j;
  $("jstatus").textContent = j.status; $("jstatus").className = "chip " + j.status;
  const v = j.video || {};
  $("jmeta").textContent = j.live
    ? "live camera" + (v.width ? `  ${v.width}\\u00d7${v.height}` : "") +
      (j.queue_position != null && j.queue_position >= 0
        ? `  \\u00b7 queued behind ${j.queue_position}` : "")
    : `${j.source_label || ""}  ${v.width || "?"}\\u00d7${v.height || "?"}` +
      ` @ ${v.fps || "?"} fps` + (j.queue_position != null && j.queue_position >= 0
        ? `  \\u00b7 queued behind ${j.queue_position}` : "");
  $("jbar").style.width = j.progress != null ? (j.progress * 100).toFixed(1) + "%" : "0";
  let prog;
  if (j.live) {
    prog = `LIVE ${fmtS(j.t_source || 0)} \\u00b7 ${j.frame_index || 0} frames`;
    if (j.proc_fps) prog += ` \\u00b7 ${j.proc_fps.toFixed(1)} fps`;
  } else {
    prog = `${j.frame_index || 0}/${j.n_total || "?"}`;
    if (j.proc_fps) prog += ` \\u00b7 ${j.proc_fps.toFixed(1)} fps`;
    if (j.eta_s != null && j.status === "running") prog += ` \\u00b7 eta ${fmtS(j.eta_s)}`;
  }
  $("jprog").textContent = prog;
  const active = ["queued", "normalizing", "running"].includes(j.status);
  $("jcancel").classList.toggle("hidden", !active);
  $("jcancel").textContent = j.live ? "Stop session" : "Cancel";
  if (j.live && active) { startCamera(j.job_id); } else { stopCamera(); }

  if (j.status === "running" && j.frame_b64) {
    $("jframe").src = "data:image/jpeg;base64," + j.frame_b64;
    show("jframe"); hide("jvideo"); hide("jempty");
  } else if (!active) { attachVideo(j); }
  else if (j.live) { $("jempty").textContent = "starting camera and models..."; }

  // models rail
  const rows = [];
  for (const kind of ["detector", "pose"]) {
    const m = j.models[kind]; if (!m) continue;
    const badge = `<span class="badge ${m.badge}">${(m.badge || "").toUpperCase()}</span>`;
    const proto = m.badge === "prototype"
      ? '<div class="note">prototype weights, internal demonstration only</div>' : "";
    rows.push(`<div style="margin-bottom:10px"><div>${m.display_name} ${badge}</div>
      <div class="note mono">${m.sha8}</div>${proto}</div>`);
  }
  if (j.compare_peer) {
    rows.push(`<div style="margin-bottom:8px"><a href="#cmp/${j.job_id}/${j.compare_peer}">` +
      `side-by-side view</a></div>`);
  }
  rows.push(`<div class="note">rules: ${j.rules_profile}` +
    (j.rules_profile === "demo" ? " (demo calibration, not the clinical operating point)" : "") +
    ` \\u00b7 score_thr ${j.score_thr}</div>`);
  const zones = j.zones || [];
  if (zones.length) {
    // zone names are server-validated to [A-Za-z0-9 ._-], safe for innerHTML
    rows.push(`<div class="note">exit zone${zones.length > 1 ? "s" : ""}: ` +
      zones.map(z => z.name).join(", ") + `</div>`);
  }
  $("jmodels").innerHTML = rows.join("");

  // latency rail
  if (j.latency_p50_ms) {
    const stages = ["source", "detect", "pose", "track", "rules", "render"];
    const p50 = j.latency_p50_ms, p95 = j.latency_p95_ms || {};
    $("jlatency").innerHTML = `<table><tr><th>stage</th><th>p50 ms</th><th>p95 ms</th></tr>` +
      stages.map(s => `<tr><td>${s}</td><td>${fmt1(p50[s])}</td><td>${fmt1(p95[s])}</td></tr>`)
        .join("") + "</table>";
  } else if (j.status === "running") {
    $("jlatency").textContent = "measuring...";
  }

  // alerts rail + timeline
  const alerts = j.alerts || [];
  if (alerts.length) {
    $("jalerts").innerHTML = alerts.map((a, i) => {
      const zoneNote = a.details && a.details.exit_zone
        ? ` \\u00b7 zone ${a.details.exit_zone}` : "";
      return `<div class="alertrow" onclick="seek(${a.onset_ts})">
         <span class="rid">${eventName(a.rule_id)}</span>
         <div class="note">track ${a.track_id} \\u00b7 onset ${a.onset_ts.toFixed(1)} s${zoneNote}
           \\u00b7 <span class="mono">${a.rule_id}</span></div>
       </div>`;
    }).join("");
  } else {
    $("jalerts").innerHTML = j.rules_profile === "production"
      ? '<div class="note">0 events; production windows need minutes of footage</div>'
      : '<div class="note">none</div>';
  }
  renderTimeline(j);

  if (j.error) { show("jerr"); $("jerr").textContent = j.error; } else hide("jerr");
}
function fmt1(x) { return x == null ? "-" : Number(x).toFixed(1); }
function attachVideo(j) {
  hide("jframe");
  if (j.annotated_ready) {
    const url = `/api/jobs/${j.job_id}/annotated.mp4`;
    const vid = $("jvideo");
    if (!vid.src || !vid.src.includes(j.job_id)) vid.src = url;
    show("jvideo"); hide("jempty");
    $("jdownload").innerHTML = `<a href="${url}" download="vividly_${j.job_id}.mp4">` +
      `Download annotated video</a>`;
  } else {
    hide("jvideo"); show("jempty");
    $("jempty").textContent = j.status === "failed" ? "run failed"
      : (j.live ? "live session ended" : "no output");
  }
}
function renderTimeline(j) {
  const v = j.video || {}; const dur = v.duration_s;
  const alerts = j.alerts || [];
  if (!dur || !alerts.length) { hide("jtimeline"); return; }
  show("jtimeline");
  $("jtimeline").innerHTML = alerts.map(a =>
    `<i class="tick" style="left:${Math.min(99.5, a.onset_ts / dur * 100)}%"
        title="${eventName(a.rule_id)} @ ${a.onset_ts.toFixed(1)}s"
        onclick="seek(${a.onset_ts})"></i>`
  ).join("");
}
function seek(t) {
  const vid = $("jvideo");
  if (!vid.classList.contains("hidden")) { vid.currentTime = Math.max(0, t - 2); vid.play(); }
}
$("jcancel").addEventListener("click", async () => {
  if (!state.job) return;
  const action = state.job.live ? "stop" : "cancel";
  await fetch(`/api/jobs/${state.job.job_id}/${action}`, { method: "POST" });
});

function openJob(id) {
  closeWs();
  $("jframe").classList.add("hidden"); $("jvideo").classList.add("hidden");
  $("jvideo").removeAttribute("src"); show("jempty");
  $("jempty").textContent = "connecting...";
  $("jdownload").innerHTML = "";
  connectWs(id, 0);
}
function connectWs(id, attempt) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/jobs/${id}`);
  state.ws = ws;
  ws.onmessage = (ev) => renderJob(JSON.parse(ev.data));
  ws.onclose = () => {
    if (state.ws !== ws) return;
    const j = state.job;
    if (j && ["queued", "normalizing", "running"].includes(j.status)) {
      setTimeout(() => { if (state.ws === ws) connectWs(id, attempt + 1); },
                 Math.min(5000, 500 * (attempt + 1)));
    }
  };
  ws.onerror = () => ws.close();
}
function closeWs() { const ws = state.ws; state.ws = null; if (ws) ws.close(); }

// ---------- compare view ----------
const cmp = { a: null, b: null, timer: null };
function cmpSideHtml(sid) {
  return `<div class="note" id="${sid}-title" style="margin-bottom:6px"></div>
    <div class="viewer">
      <img id="${sid}-frame" class="hidden" alt="live frame">
      <video id="${sid}-video" class="hidden" playsinline muted></video>
      <div id="${sid}-empty" class="empty">waiting...</div>
    </div>
    <div class="note" id="${sid}-stats" style="margin-top:6px"></div>`;
}
function openCompare(a, b) {
  closeWs(); clearInterval(cmp.timer);
  cmp.a = a; cmp.b = b;
  $("cmpA").innerHTML = cmpSideHtml("cmpA");
  $("cmpB").innerHTML = cmpSideHtml("cmpB");
  hide("cmpplay"); $("cmpplay").textContent = "Play both";
  const va = $("cmpA-video"), vb = $("cmpB-video");
  va.addEventListener("timeupdate", () => {
    if (!va.paused && Math.abs(va.currentTime - vb.currentTime) > 0.12) {
      vb.currentTime = va.currentTime;
    }
  });
  va.addEventListener("ended", () => { vb.pause(); $("cmpplay").textContent = "Play both"; });
  cmp.timer = setInterval(pollCompare, 1000);
  pollCompare();
}
async function pollCompare() {
  let allDone = true;
  for (const [sid, jid] of [["cmpA", cmp.a], ["cmpB", cmp.b]]) {
    try {
      const r = await fetch("/api/jobs/" + jid);
      if (!r.ok) continue;
      const j = await r.json();
      renderCmpSide(sid, j);
      if (["queued", "normalizing", "running"].includes(j.status)) allDone = false;
    } catch (e) { allDone = false; }
  }
  if (allDone) clearInterval(cmp.timer);
}
function renderCmpSide(sid, j) {
  const det = j.models.detector || {};
  $(sid + "-title").innerHTML =
    `<b>${det.display_name || "?"}</b> <span class="mono">${det.sha8 || ""}</span> ` +
    chip(j.status);
  const p50 = j.latency_p50_ms
    ? ` \u00b7 detect p50 ${fmt1(j.latency_p50_ms.detect)} ms` : "";
  const alerts = (j.alerts || []).length;
  let stats = `${alerts} candidate event${alerts === 1 ? "" : "s"}${p50}`;
  if (j.status === "running" && j.proc_fps) stats += ` \u00b7 ${j.proc_fps.toFixed(1)} fps`;
  $(sid + "-stats").textContent = stats;
  if (sid === "cmpA") {
    const v = j.video || {};
    $("cmpmeta").textContent =
      `${j.source_label || ""}  ${v.width || "?"}\u00d7${v.height || "?"}`;
  }
  const img = $(sid + "-frame"), vid = $(sid + "-video"), empty = $(sid + "-empty");
  const active = ["queued", "normalizing", "running"].includes(j.status);
  if (j.status === "running" && j.frame_b64) {
    img.src = "data:image/jpeg;base64," + j.frame_b64;
    img.classList.remove("hidden"); vid.classList.add("hidden"); empty.classList.add("hidden");
  } else if (!active) {
    img.classList.add("hidden");
    if (j.annotated_ready) {
      if (!vid.src || !vid.src.includes(j.job_id)) {
        vid.src = `/api/jobs/${j.job_id}/annotated.mp4`;
      }
      vid.classList.remove("hidden"); empty.classList.add("hidden");
      const other = sid === "cmpA" ? $("cmpB-video") : $("cmpA-video");
      if (!other.classList.contains("hidden")) show("cmpplay");
    } else {
      empty.textContent = j.status; empty.classList.remove("hidden");
      vid.classList.add("hidden");
    }
  } else {
    empty.textContent = j.status +
      (j.queue_position != null ? ` (position ${j.queue_position + 1})` : "");
    empty.classList.remove("hidden");
  }
}
$("cmpplay").addEventListener("click", () => {
  const a = $("cmpA-video"), b = $("cmpB-video");
  if (a.paused) {
    b.currentTime = a.currentTime;
    a.play().catch(() => {}); b.play().catch(() => {});
    $("cmpplay").textContent = "Pause";
  } else { a.pause(); b.pause(); $("cmpplay").textContent = "Play both"; }
});

// ---------- routing ----------
function show(id) { $(id).classList.remove("hidden"); }
function hide(id) { $(id).classList.add("hidden"); }
function showErr(id, msg) { show(id); $(id).textContent = msg; }
function route() {
  const c = location.hash.match(/^#cmp\\/([0-9a-f]+)\\/([0-9a-f]+)$/);
  const m = location.hash.match(/^#job\\/([0-9a-f]+)$/);
  if (c) {
    hide("view-home"); hide("view-job"); show("view-cmp");
    clearTimeout(state.pollTimer); stopCamera();
    openCompare(c[1], c[2]);
  } else if (m) {
    hide("view-home"); show("view-job"); hide("view-cmp");
    clearTimeout(state.pollTimer); clearInterval(cmp.timer);
    openJob(m[1]);
  } else {
    show("view-home"); hide("view-job"); hide("view-cmp");
    closeWs(); clearInterval(cmp.timer); stopCamera(); refreshRuns();
  }
}
window.addEventListener("hashchange", route);

// ---------- boot ----------
(async function boot() {
  try {
    const r = await fetch("/api/health"); const h = await r.json();
    if (!h.ffmpeg) { show("ffmpegwarn"); $("ffmpegwarn").title = h.ffmpeg_error || ""; }
  } catch (e) { /* page still works for browsing past runs */ }
  await loadModels();
  route();
})();
</script>
</body>
</html>
"""


def _logo_data_uri() -> str:
    """The committed brand mark, inlined so the page keeps its zero-network-assets rail."""
    mark = Path(__file__).resolve().parents[3] / "assets" / "brand" / "vividly-square-black-96.png"
    try:
        return "data:image/png;base64," + base64.b64encode(mark.read_bytes()).decode("ascii")
    except OSError:
        return ""


PAGE_HTML = _PAGE_TEMPLATE.replace("__LOGO__", _logo_data_uri())
