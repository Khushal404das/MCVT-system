# MCVT — Multi-Camera Vehicle Tracking System

Agent-based vision mixer from the SRS: every vehicle in every camera is detected
(YOLOv8), tracked inside each camera (DeepSORT), re-identified across cameras
(ResNet50-IBN trained on VeRi-776), and given one global id (`VH-001`, ...).
Click a vehicle in the web UI and the system **follows it across all cameras**:
the camera switching engine (DBN + hysteresis + 5-frame buffer + fallback) keeps
the best view on screen. **Request full video** stitches every clip of that
vehicle in time order (FFmpeg, overlaps removed) and emails it (SMTP).

```
 cameras (files / RTSP)                                     browsers
        │                                                       ▲
        ▼                                                       │ ws:// (JSON + JPEG frames)
 ┌───────────────────────────── Python AI engine ─────────┐  ┌──┴──────────────────────┐
 │ camera readers → YOLOv8 → DeepSORT (per camera)         │  │ PHP WebSocket hub        │
 │   → ResNet-IBN ReID → global ids (feature DB)           │──▶ (web/bin/ws-server.php)  │
 │   → camera scoring + DBN switching (per followed car)   │◀── focus commands           │
 │   → journey worker: FFmpeg stitch + SMTP email          │  └─────────────────────────┘
 └──────────────┬──────────────────────────────────────────┘  ┌─────────────────────────┐
                └──────── SQLite (storage/mcvt.sqlite) ───────▶ PHP web app + REST API   │
                                                               │ login, vehicles, details,│
                                                               │ journey requests, media  │
                                                               └─────────────────────────┘
```

The deep-learning pipeline is Python (PyTorch/Ultralytics, as the SRS requires);
the web application, REST API and WebSocket server are PHP (no Composer packages).

## Folder layout

| Path | What it is |
|---|---|
| `config.yaml` | the single configuration file (thresholds, cameras, ports, SMTP host …) |
| `.env` | secrets: `ENGINE_TOKEN`, `SMTP_USER`, `SMTP_PASSWORD` (see `.env.example`) |
| `schema.sql` | database schema shared by Python and PHP |
| `engine/mcvt/` | detector, tracker, ReID model + extractor, matcher, switching, stitcher, email, pipeline |
| `engine/models/` | `yolov8n/s/m.pt`, `veri_sbs_R50-ibn.pth` |
| `engine/tests/` | pytest suite — one test per case in `test_cases.pdf` |
| `web/lib/` | `WsServer.php` (RFC 6455 hub), `App.php` (config/db/auth), `Yaml.php` |
| `web/public/` | dashboard (`index.php`), `login.php`, `api/*.php`, `media.php`, assets |
| `storage/` | database, thumbnails, journey videos, analysis streams, logs (outside the web root) |

## Requirements

* Python 3.9+ and `pip install -r requirements.txt` (an NVIDIA GPU + CUDA torch is recommended)
* PHP 8.1+ with `pdo_sqlite` (XAMPP's PHP works)
* FFmpeg — the `imageio-ffmpeg` package ships one, or put `ffmpeg` on PATH

## First run

```bash
pip install -r requirements.txt
```
```bash
php web/bin/create-user.php admin admin
```
```bash
python engine/prepare_streams.py
```
```bash
start.bat
```

Then open <http://127.0.0.1:8080> and sign in. `start.bat` opens three windows:
the WebSocket hub, the web server and the AI engine. To start them by hand:

```bash
php web/bin/ws-server.php
```
```bash
php web/bin/serve.php
```
```bash
python engine/run_engine.py
```

`prepare_streams.py` is a one-time step for recorded files: it writes a 960 px /
10 FPS copy of each video (the equivalent of an IP camera's sub-stream). The AI
and the live view use those copies, which keeps a CPU-only laptop responsive.
Journey videos are still cut from the full-quality originals. Turn it off with
`cameras.analysis.enabled: false`.

## Cameras — nothing is hard-coded

* Every file matching `cameras.discover.pattern` in `cameras.discover.directory`
  becomes a camera; its id comes from the file name (`camera_c21.mp4 → C21`).
* Add RTSP cameras under `cameras.sources` (optionally with `analysis_source`
  set to the camera's sub-stream URL).
* Resolution, FPS and the resolution weight `w_r` are read from the streams;
  override `name` / `resolution_weight` per camera in `cameras.overrides`.

## Using the dashboard

* **Live view**: the active camera with boxes and global ids; tiles below show all cameras.
* **Live vehicles**: every vehicle currently in view: type, colour, parked/moving, cameras.
* **Click a vehicle** (in the list or its box on the video) to open its details and
  **follow it across all cameras**. A banner announces every automatic camera switch.
* **Request full video → Send** (3 clicks from the live view) builds the journey
  video and emails it. Leave the address empty to only build it for download.
* **All vehicles** searches history; **Requests** lists journey jobs with download links.

## Following a vehicle across cameras (hand-off)

The same car looks quite different from another camera's angle, so ReID similarity
between cameras is lower than within one camera. The matcher therefore uses a
camera-link (spatio-temporal) prior, an idea from the AIC19 Track 1 reference:

* A vehicle that **left** a camera less than `reid.handoff_window_seconds` ago can be
  matched in a **different** camera at `reid.handoff_threshold` (instead of 0.75).
* The vehicle you are tracking gets the lowest threshold, `reid.follow_threshold`.
* A vehicle that is still visible somewhere always needs the strict threshold.
* Camera-to-camera transitions (and typical travel times) are learned as they happen.
  While the tracked car is between cameras, the UI shows the cameras it is expected in
  next. When it reappears, the view switches to that camera automatically.

## Improving detection by training on your cameras

No hand-labelled data is needed: a large teacher model labels frames from your own
cameras, and the fast YOLOv8n is fine-tuned on them.

```bash
python engine/training/make_dataset.py --per-camera 120
```
```bash
python engine/training/train_detector.py --epochs 30
```
```bash
python engine/training/evaluate_detector.py engine/models/yolov8n.pt engine/models/yolov8n_mcvt.pt
```

Then set `detection.model: engine/models/yolov8n_mcvt.pt` in `config.yaml`.
`engine/training/calibrate_handoff.py` measures how similar the same car looks in
different cameras, which helps you tune the hand-off thresholds.

## Email

Put the SMTP account in `.env`:

```
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-app-password
```

The host, port and TLS mode are set under `email` in `config.yaml`. A wrong
password is reported in the log panel and on the request; it never stops tracking.

## Tests

```bash
cd engine && python -m pytest tests -v
```

All 20 cases in `test_cases.pdf` are covered: TC-VD-01..05, TC-CS-01..05,
TC-RI-01..05 and TC-VE-01..05. Detection and ReID tests use frames from the
dataset. Email tests run against a local fake SMTP server, so no real mail is sent.
TC-RI-05 (≤ 30 ms per crop) is marked *expected to fail* on CPU-only machines,
because the SRS ties that figure to a GPU.

## Requirement map

| SRS | Where |
|---|---|
| FR-VD-01..05 | `engine/mcvt/detector.py` (batched over all cameras, class + confidence filter) |
| FR-TR-01..04 | `engine/mcvt/tracker.py` (DeepSORT, Kalman prediction, max_age) |
| FR-RI-01..05 | `reid/resnet_ibn.py`, `reid/extractor.py`, `matcher.py` (cosine ≥ 0.75, persistent gallery) |
| FR-CS-01..06 | `switching.py` (score, DBN belief, hysteresis, frame buffer, overlap, fallback) |
| FR-UI-01..04 | `web/public/index.php`, `assets/app.js` |
| FR-VS-01..05 | `stitcher.py`, `jobs.py` (clip index = `sightings` table) |
| FR-ED-01..04 | `emailer.py` |
| NFR-S-01..03 | secrets in `.env`; login on every page/API/WebSocket; media served only via `media.php` |
| NFR-M-02 | `config.yaml` (read by both Python and PHP) |

## Ideas reused from the reference repositories

* **AIC19 Track 2 (UWIPL)**: video-based ReID with temporal pooling. A track's
  identity is the average of several crop embeddings. Colour/type metadata is
  shown as vehicle attributes.
* **AIC19 Track 1 / TNT**: tracklet-level association and NMS of duplicates.
  Here, a global id that later proves to be an existing vehicle is merged into it.
  Two simultaneous tracks in one camera can never share an id.

## Performance notes

On a CPU-only laptop the system runs at a few analysed frames per second per
camera; it is functional but slower than the SRS figures, which assume a GPU.
With CUDA, `tracking.appearance: auto` switches DeepSORT to ResNet-IBN
embeddings, and all models run on the GPU. Motion-gated scheduling
(`pipeline.motion_gating`) analyses idle cameras less often.

## Moving vehicles only, and accurate boxes

* YOLOv8 detects every vehicle. A vehicle is reported only while it is **moving**:
  each camera reader keeps a background model on every video frame, and a detection
  counts as moving when enough of the pixels in its box change
  (`detection.motion_box_fraction`). On these cameras, moving cars score 0.11–0.16 and
  parked cars 0.00. A vehicle that stops for `detection.stopped_after_seconds` is
  hidden again but keeps its ID.
* The AI analyses each camera a few times per second on a CPU. To keep boxes exactly on
  the cars, the live view is shown `pipeline.display_delay_seconds` (2.5 s) behind real
  time. Every frame then lies between two AI results, and its boxes are interpolated
  between them. Set the delay to 0 for no delay; boxes then follow by optical flow and
  are less exact.
* **One continuous box per vehicle.** At 1–2 AI checks per second a car driving towards a
  camera moves and grows too much between checks for DeepSORT's box overlap, so it gets a
  new track. A new track that appears where a moving vehicle's lost track should be now
  (same direction and speed, similar size and colour; `tracking.link_*`) continues that
  vehicle: same ID, same box, ReID samples kept. The live view bridges missed detections
  up to `pipeline.overlay_max_gap_seconds` and runs the box along the car's motion for
  `pipeline.overlay_edge_seconds` before the first and after the last detection.
  YOLOv8 uses class-agnostic NMS (`detection.agnostic_nms`), so a vehicle scored as both
  "car" and "truck" no longer gets two boxes.

### Following one vehicle across cameras

* Click a moving vehicle: its global ID is locked. While it is in view it is sampled by ReID
  on every AI check (whole vehicle only, top priority, `reid.follow_samples`), so the next
  camera is matched against a rich set of views. It stays boxed if it parks.
* Cameras that watch the same place (`reid.overlapping_cameras`, here C11 + C21 at the gate)
  can show it at the same time; there it is matched at the hand-off threshold and carries
  the same global ID in both.
* When it leaves every camera the dashboard shows "searching" with the learned next cameras.
  Each new moving vehicle is checked against it (logged as `Checked VH-x on C61 against
  followed VH-y: similarity …`). A match switches the view to that camera.
* A followed vehicle can match below `follow_threshold` (down to `follow_min_similarity`)
  only when the candidate is closer to it than to any other vehicle by `follow_margin`.
* Limit: on these cameras the same car seen from another camera scores 0.44–0.68 with the
  VeRi ResNet-IBN model, while a different white car scored 0.53, so a hand-off is found in
  roughly half of the passes of the test car (C21 → C61, ~42 s). A ReID model fine-tuned on
  these cameras, or a GPU (more samples per car), would raise this.

### Complete journey and replay

Click a vehicle's box (or its card in the gallery) to follow it. While it is followed:
- The main view switches live to its best camera, and when it leaves every camera, to the next camera it shows up on.
- **Complete journey** in the side panel lists every camera visit in order: entry and exit times,
  how long it stayed, and the time between cameras. The current camera is marked LIVE.
- **▶ Replay journey** (or a click on one visit) plays the recorded video of each camera it passed,
  in order. The view switches camera along its path, with the vehicle's box drawn on top (1×/2×/4×,
  previous/next camera, "Back to live").

The engine stores each visit's analysed video file and the vehicle's boxes in it (`sightings.media`,
`sightings.boxes`). `media.php` cuts each visit into a short clip with FFmpeg and caches it in
`storage/journey_clips`. The PHP built-in server answers one request at a time, so it only ever sends
small files.

### Global IDs from features + colour + type

1. A new moving vehicle's crop goes through ResNet-IBN, which gives a 2048-value appearance vector
   (shape, body style, i.e. the model of the car), plus its **colour** and **type** (the detector's class).
2. The vector is compared with every vehicle stored in the feature database (`embeddings` table).
   The best match above the threshold gives it that vehicle's ID; otherwise it gets a new `VH-…` ID.
   The vector, colour and type are stored in the database (`embeddings`, and `vehicles.color` / `vehicles.vclass`).
3. A stored vehicle whose colour or type clearly differs cannot give its ID, even when the features
   are close: white vs black, red vs silver, car vs motorcycle. Neighbouring shades
   (white/silver, silver/grey, grey/black) and car/truck still match, since cameras and the detector mix them up.
   The log then says e.g. `Track on C61 looked like VH-12 (similarity 0.66) but is not it: colour white vs black.`

Colour is named after removing each camera's tint (C31/C41 are strongly blue). The engine learns
that tint from the vehicles it sees and saves it in `storage/color_cast.json`. Settings are in
`reid.attributes` in config.yaml. Measured on 341 tracklets: colour blocks 0–4% of true
same-car matches and 14–24% of wrong look-alike matches.

### Adapting ReID to these cameras

Fine-tuning the ResNet50-IBN backbone is not practical on a CPU, so the frozen network's
features are adapted with a learned projection trained on the project's own footage:

```
python engine/training/make_reid_dataset.py --fps 2.5 --minutes 25       # ~30 min, engine stopped
python engine/training/train_reid_adapter.py --dims 512 --reg 0.3 --final  # ~5 min
```

1. `make_reid_dataset.py`: YOLOv8 + the background model + a box tracker cut one pass over
   every camera into moving-vehicle tracklets (one tracklet = one identity in one camera)
   and save up to 4 whole-vehicle crops each (`storage/training/reid`).
2. `train_reid_adapter.py`: links identities across cameras automatically: tracklets of
   overlapping cameras at the same moment, plus mutual-best pairs that leave one camera and
   enter another within 2 minutes. It fits an LDA projection (2048 → 512-d) that keeps
   what separates vehicles and drops what changes between views. It holds out the later
   40% of the video for testing. Match thresholds are converted to the new similarity
   scale (same share of different-vehicle pairs accepted) and stored with the adapter.
3. `reid.adapter` in config.yaml switches it on (`""` = plain features).

Measured (341 tracklets, held-out part of the video):

| Features | C11↔C21 same-car retrieval rank-1 / mAP | real hand-offs C21→C41/C61 (6 pairs checked by eye), mean rank |
|---|---|---|
| plain ResNet-IBN | 0.74 / 0.85 | 14.0 |
| + adapter (LDA512) | 0.92 / 0.94 | 12.0 |

Removing each camera's mean feature helped C11↔C21 but hurt the real hand-offs, so it is
not used. The hand-off gain is small: only 10 cross-camera pairs could be mined from timing.
More footage, or a few hand-confirmed hand-offs, would improve it.

### CPU budget (4-core laptop)

The models share 4 cores with video decoding, so the engine controls who gets them:

| Setting | Why |
|---|---|
| `detection.cpu_threads: 4`, `reid.cpu_threads: 1` | ResNet-IBN (~1 s/crop) running beside YOLOv8 with all threads made YOLOv8 take ~900 ms/frame; now ~170–290 ms |
| `reid.worker_batch: 1` | a moving car without an ID is always embedded next (an 8-crop batch made it wait 8–10 s) |
| `reid.parked_samples`, `verify_samples`, `samples_per_visit` | ReID time goes to moving vehicles, not parked ones |
| `detection.motion_mask_fps: 6` | the background model on every frame of 6 cameras cost 1.3 cores |
| `pipeline.motion_blob_fraction` | a camera is analysed when a car-sized blob moves (or a moving vehicle is on it), not because of parked cars the tracker lost |
| `reid.load_gallery_on_start: false` | replayed videos start with an empty gallery; reloading earlier runs (same cars under many IDs) caused false matches |

The engine logs its speed every minute (`Speed: YOLOv8 … ms/frame, ReID … ms/crop, AI checks/s …`).

## Known limitations

* Journey clips need a recording of the camera. For video files the file is the
  recording. RTSP cameras would need a recorder (not included yet).
* The dataset has no night footage; TC-VD-04 darkens a daytime frame instead.
