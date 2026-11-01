# Multi-Camera Vehicle Tracking & Re-Identification System (MCVT)

**Final Year Project (FYP)**  
*Advanced Computer Vision & Intelligent Transportation Systems*  
*Architecture inspired by CVPR AI City Challenge Track 1 (MCMT) & Track 2 (Vehicle ReID)*

---

## 🌟 Overview

The **MCVT Vision** system is an end-to-end multi-camera traffic surveillance platform that detects, tracks, extracts visual appearance signatures, and re-identifies vehicles across non-overlapping surveillance cameras without any hardcoded tracking data.

When a user tracks a vehicle on any camera (e.g. `c11`), the system continuously compares its spatial and appearance features across all other cameras (`c21`, `c31`, `c41`, `c51`, `c61`). Once the vehicle appears in another camera's field of view, the system matches the global profile and **automatically switches the surveillance view** to follow that vehicle in real time.

---

## 🚀 Key Features

1. **Synchronized 6-Camera Live Surveillance Dashboard**
   - 6 camera streams from realistic traffic intersection video feeds.
   - 6-camera grid view + focused single-camera stream with smooth transitions.
   - Interactive camera switching with click-to-focus.

2. **Real-time Vehicle Detection & Cropping**
   - YOLOv8 vehicle detection (cars, trucks, buses, motorcycles).
   - High-performance adaptive motion detector fallback with morphological filtering.
   - Sub-pixel bounding box coordinate extraction and auto-cropping.

3. **Multi-Modal Visual Feature Extraction (CVPR Track 2 Approach)**
   - **HSV Color Histograms:** Captures dominant body paint colors under lighting variations.
   - **Spatial Color Grids:** Captures 2D color layout (roof vs hood vs body panels).
   - **Aspect Ratio & Geometry:** Distinguishes vehicle types (sedan, SUV, truck, bus).
   - **Edge & Texture Descriptors:** Captures grill, window, and wheel textures.
   - **Deep Feature Representation:** Projected into a compact, L2-normalized 256-dimensional embedding vector.

4. **Cross-Camera Re-Identification (ReID) Gallery (CVPR Track 1 Approach)**
   - Global gallery of vehicle identities (`Global ID: G1, G2, ...`).
   - Cosine similarity matching between incoming detections and stored profiles.
   - Running average feature update to adapt to gradual angle and lighting changes.

5. **Target Locking & Automated Camera Following**
   - One-click target locking on any detected vehicle (`🎯 Track Target`).
   - Cross-camera matching triggers instant notification and automatic UI camera switching.
   - Full camera transition history logged (`CAM C11 → CAM C21 → CAM C31`).

6. **Executive Dark-Mode Dashboard (PHP + Vanilla CSS + Vanilla JS)**
   - Glassmorphism design system with modern gradients and micro-animations.
   - Live telemetry: Unique Vehicles, Active Vehicles, Cross-Cam ReIDs, and New Signatures.
   - Real-time activity log and non-intrusive toast notification system.

---

## 🏗️ System Architecture

```
[Dataset Video Feeds]
 (c11, c21, c31, c41, c51, c61)
           │
           ▼
[Multi-Camera Video Processor] ── (Multi-Threaded Frame Decoder)
           │
           ├─► [Vehicle Detector (YOLOv8 / Adaptive MOG2)]
           │            │
           │            ▼ (Vehicle Crops)
           ├─► [Feature Extractor (256-d Multi-Modal Appearance Vector)]
           │            │
           │            ▼
           ├─► [Single-Camera Trackers (IoU + Visual Feature Affinity)]
           │            │
           │            ▼ (Active Tracklets)
           └─► [Cross-Camera ReID Engine (Cosine Similarity Gallery)]
                        │
                        ├─► Auto-Switch Event Trigger
                        ▼
           [Flask API Server (Port 5000)]
             • MJPEG Video Streams (/stream/<cam_id>, /thumbnail/<cam_id>)
             • REST Endpoints (/api/cameras, /api/vehicles, /api/track)
                        ▲
                        │ (REST & MJPEG Fetch)
           [PHP Surveillance Dashboard (Port 8080)]
             • index.php / index.html
             • Custom Dark-Mode Design System (style.css)
             • Interactive Controller & Auto-Switch Handler (app.js)
```

---

## 📁 Directory Structure

```
FYP_PROJECT-main/
├── dataset/                        # Traffic video files (camera_c11.mp4, etc.)
├── run.sh                          # Linux startup script
├── run.bat                         # Windows startup script
└── mcvt-system/
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # System documentation
    ├── backend/
    │   ├── app.py                  # Flask REST API & streaming server
    │   ├── video_processor.py      # Multi-camera orchestrator & streaming
    │   ├── vehicle_detector.py     # YOLOv8 + adaptive fallback detector
    │   ├── feature_extractor.py    # Multi-modal feature extractor
    │   ├── tracker.py              # Single-camera tracking
    │   └── cross_camera_reid.py    # Cross-camera ReID gallery & auto-switch
    ├── frontend/
    │   ├── index.php               # Main PHP surveillance dashboard
    │   ├── index.html              # Static companion dashboard
    │   └── assets/
    │       ├── css/style.css       # Premium dark-mode glassmorphic styles
    │       └── js/app.js           # Frontend client controller & auto-switcher
    └── output/                     # Saved frames and feature artifacts
```

---

## 🚦 How to Run

### Option A: Using the Launcher Script (Recommended)

#### On Linux / Ubuntu:
```bash
./run.sh
```

#### On Windows:
Double-click `run.bat` or run:
```cmd
run.bat
```

The script will automatically detect PHP and start:
1. **PHP Frontend:** `http://localhost:8080/index.php`
2. **AI Backend:** `http://localhost:5000`
3. (If PHP is not yet installed, you can access the dashboard directly at `http://localhost:5000/ui`).

---

### Option B: Manual Startup

#### 1. Start the Python AI Backend:
```bash
cd mcvt-system/backend
python3 app.py
```

#### 2. Start the PHP Frontend:
```bash
cd mcvt-system/frontend
php -S localhost:8080
```
Then open your browser to **`http://localhost:8080/index.php`**.

---

## 🛠️ Installing PHP (If Not Installed)

On Ubuntu/Debian Linux:
```bash
sudo apt update
sudo apt install php php-cli
```

---

## 🔬 Mathematical Formulation of ReID Matching

Given a probe vehicle detection crop $I_p$ in Camera $C_i$ with normalized feature vector $\mathbf{f}_p \in \mathbb{R}^{256}$ and a gallery of existing vehicle identities $\mathcal{G} = \{ (g, \mathbf{f}_g) \}_{g=1}^N$:

1. **Cosine Similarity Computation:**
   $$S(\mathbf{f}_p, \mathbf{f}_g) = \frac{\mathbf{f}_p \cdot \mathbf{f}_g}{\|\mathbf{f}_p\|_2 \|\mathbf{f}_g\|_2}$$

2. **Identity Assignment:**
   $$g^* = \arg\max_{g \in \mathcal{G}} S(\mathbf{f}_p, \mathbf{f}_g)$$

   - If $S(\mathbf{f}_p, \mathbf{f}_{g^*}) \ge \tau$ (where threshold $\tau = 0.60$):
     - Assign detection to Global Identity $g^*$.
     - Update gallery template with momentum $\alpha = 0.85$:
       $$\mathbf{f}_{g^*} \leftarrow \alpha \mathbf{f}_{g^*} + (1 - \alpha) \mathbf{f}_p$$
     - If $g^* == \text{TargetID}$ and $C_i \neq \text{CurrentView}$, trigger **Camera Auto-Switch** to $C_i$.
   - If $S(\mathbf{f}_p, \mathbf{f}_{g^*}) < \tau$:
     - Register as new Global Identity $g_{\text{new}}$ with profile $\mathbf{f}_p$.

---

## 👥 Credits
- Developed for University Final Year Project (FYP).
- Theoretical basis: CVPR 2019 AI City Challenge Tracks 1 & 2.
