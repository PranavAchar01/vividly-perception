# Live camera bring-up

The path from a physical camera to the perception stack, written against a Reolink RLN16-410
network video recorder with power-over-Ethernet cameras. The same shape, camera as an RTSP
sensor feeding on-premises compute, is what a deployed appliance uses. No source video is
recorded: `scripts/live_camera.py` writes annotated frames only, to the directory given by
`--out`.

## Topology

The recorder runs an internal power-over-Ethernet network for its cameras, separate from the
house network:

```
Reolink camera --PoE--> RLN16-410 recorder --(one PoE port)--> USB-C Ethernet --> Mac
   172.16.25.x                172.16.25.1                        en9
```

- The segment is `172.16.25.0/24`: the recorder is `.1` and each camera takes its own address
  on the segment. The camera of record is at `.10`, confirmed in the 2026-08-27 section below,
  and that is the address every example here uses.
- There is no DHCP on that segment, so the Mac's adapter self-assigns a link-local address and
  cannot route. Give it a static alias on that subnet. The change is non-destructive and is
  undone by unplugging:

  ```bash
  sudo ifconfig en9 inet 172.16.25.50 netmask 255.255.255.0 alias
  ```

## Credentials

Keep the stream credentials out of the repository. `scripts/live_camera.py` reads one variable,
`REOLINK_RTSP_URL`, and the account and password travel inside that URL, so the file to write is
the composed URL and nothing else:

```bash
umask 077 && cat > ~/.reolink.env <<'EOF'
# PASS is the camera's own admin password, not the recorder's; see the 2026-08-27 section.
REOLINK_RTSP_URL='rtsp://admin:PASS@172.16.25.10:554/h264Preview_01_sub'
EOF
```

Source that file in the shell that runs the script, `set -a; . ~/.reolink.env; set +a`, or pass
the same URL to `--url`. Those are the only two ways the credential reaches the script.

The recorder locks an account for roughly 30 minutes after a handful of failed logins, and it
returns the same login-failed message it returns for a wrong password, so a lockout is easy to
mistake for a bad credential.

## Streams

RTSP ships disabled on recent Reolink firmware; RTMP on port 1935 is usually enabled. The camera
answers `api.cgi` over HTTPS with relaxed transport security. The URL forms carry credentials
inline and are never committed:

- RTSP, after enabling: `rtsp://admin:PASS@172.16.25.10:554/h264Preview_01_main`
- RTMP, working out of the box:
  `rtmp://172.16.25.10:1935/bcs/channel0_main.bcs?channel=0&stream=0&user=admin&password=PASS`

The stream paths above are the general Reolink forms. The two that were exercised against the
camera of record are listed in the 2026-08-27 section below.

## Run perception on the live feed

```bash
REOLINK_RTSP_URL='rtsp://admin:PASS@172.16.25.10:554/h264Preview_01_sub' \
  uv run python scripts/live_camera.py --frames 20 --out outputs/live
```

`scripts/live_camera.py` drives `vividly.sources.LiveRTSPSource` through the same detector and
pose factory the replay pipeline uses, on ONNX and CPU, writing annotated frames. Models default
to the smoke configuration's evaluation-only zoo assets; point `--config` at
`configs/pipeline/replay_demo.yaml`, whose top-level `detector` and `pose` blocks name the lane
graphs under `final_models/`, to run the models of record instead. Continuous mode is
`--frames 0`, which runs until interrupted; pair it with `--rolling` so the annotated output
overwrites a single `latest.jpg` and each processed frame appends one line to `detections.log`,
and with `--stride N` to process every Nth source frame. At 25 fps, `--stride 5` is about five
inferences per second. In continuous mode the source is reopened with backoff if the stream
drops, so a camera blip does not end the run.

## The Clip Library profile

The 2026-09-09 owner decision in `POLICY.md` (section 2) admits one use of the live feed
beyond bring-up: `configs/pipeline/clip_library.yaml` reads the camera through
`source: {rtsp_env: REOLINK_RTSP_URL, camera_id: ...}` (the URL stays in the environment and
is never written into a config), stamps every frame with the wall clock, blacks out the
configured exclusion masks before detection, runs the two allowlisted rule heads and writes
one spool record per firing into the Clip Library's data directory. Nothing from the rig
enters any corpus, the subject is the founder alone, and every Clip the rig produces is a
watermarked demonstration in a demo store. Only the `clip_library` and `device_track`
profile families may open the live source; `scripts/run_pipeline.py` refuses it elsewhere.
Exactly one pipeline runs on the rig at a time (`VIVIDLY_MODE`).

## Notes

- `LiveRTSPSource` (`src/vividly/sources/rtsp.py`) is the working body of the `RTSPSource` stub.
  It is unbounded and forward-only with no seek, reconnects on transient drops, and never logs
  the credentialed URL. It is reachable only from an explicit, credential-bearing invocation,
  never from a default pipeline run.
- Because the camera lives on the recorder's private segment, that segment is the isolated
  camera network the rig plan calls for: the edge node sits on it through a second network
  interface, exactly as this machine does through `en9`.

## Confirmed working, 2026-08-27, RLN16-410 with RLC-810A

First live bring-up against real hardware. This rig is a live inference bring-up, not a
recording setup: nothing it sees is recorded into any corpus, no frame from it enters any
manifest or any training or evaluation set, and `scripts/live_camera.py` writes annotated
frames only. That is the reading [../POLICY.md](../POLICY.md), rule 4 supports, where the
recorded interpretation is that a live demonstration is not a dataset, while self-recorded
footage and a labelling loop over it stay barred. [PIPELINE.md](PIPELINE.md) section 2 states
the same bar and should be read with this paragraph.

Findings that differ from the topology above:

- **Target camera.** Reolink RLC-810A (4K), `Camera1`, at `172.16.25.10` on the recorder's
  segment (`172.16.25.0/24`). The Mac joins that segment through a USB-Ethernet adapter with a
  static alias: `sudo ifconfig en9 inet 172.16.25.50 netmask 255.255.255.0 alias`.
- **Credentials.** A power-over-Ethernet camera the recorder auto-adds carries its own
  account, `admin`, and the recorder's own admin password does not apply to it. The password
  that reaches this camera is not recorded in this document or anywhere else in the
  repository: it reaches `scripts/live_camera.py` only inside the stream URL, from `--url` or
  the `REOLINK_RTSP_URL` environment variable, and is never committed, under the
  no-secrets-in-code rule of `POLICY.md` section 7. That variable is the only one the script
  reads, and `.env.example` carries it as a commented placeholder, in the URL form the
  credentials section above shows.
- **Enable RTSP first**, since it ships disabled: `api.cgi` `SetNetPort` with `rtspEnable=1`.
  Working URLs:
  - sub stream (H.264, 640x360): `rtsp://admin:PASS@172.16.25.10:554/h264Preview_01_sub`
  - main stream (H.265, 3840x2160): `rtsp://admin:PASS@172.16.25.10:554/h265Preview_01_main`
- **Verified.** `scripts/live_camera.py` ran detection and pose on ONNX and CPU against the live
  feed at 22 to 68 ms per frame, on both the sub stream and the 4K main stream, detecting people
  per frame. That is real time on CPU for a single stream.

```bash
# sub-stream, fast enough for real time
REOLINK_RTSP_URL='rtsp://admin:PASS@172.16.25.10:554/h264Preview_01_sub' \
  uv run python scripts/live_camera.py --frames 10 --out outputs/live
```
