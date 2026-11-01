<?php
/**
 * MCVT Vision – Multi-Camera Vehicle Tracking & Re-Identification System
 * Final Year Project (FYP) Dashboard
 * Frontend: PHP 8.2  |  Backend: Python Flask + WebSocket (port 5001)
 */

$api_base = "http://localhost:5000";
$ws_base  = "ws://localhost:5001";

$page_title = "MCVT Vision – Multi-Camera Vehicle Tracking & Re-Identification";

$cameras = [
    'c11' => ['name' => 'Camera 11', 'location' => 'North Intersection'],
    'c21' => ['name' => 'Camera 21', 'location' => 'Main Boulevard East'],
    'c31' => ['name' => 'Camera 31', 'location' => 'Highway Overpass'],
    'c41' => ['name' => 'Camera 41', 'location' => 'South Toll Gate'],
    'c51' => ['name' => 'Camera 51', 'location' => 'West Ramp Inbound'],
    'c61' => ['name' => 'Camera 61', 'location' => 'Commercial Crossing'],
];
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?php echo htmlspecialchars($page_title); ?></title>

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

    <!-- Design System -->
    <link rel="stylesheet" href="assets/css/style.css">

    <!-- Expose PHP config to JavaScript -->
    <script>
        const PHP_CONFIG = {
            api_base: "<?php echo $api_base; ?>",
            ws_base:  "<?php echo $ws_base; ?>",
            cameras:  <?php echo json_encode(array_keys($cameras)); ?>
        };
    </script>
</head>
<body>

    <!-- Loading Screen -->
    <div id="loading-overlay" class="loading-overlay">
        <div class="loading-spinner"></div>
        <div id="loading-text" class="loading-text">Initializing AI Vision Core & Cross-Camera Network...</div>
        <div class="loading-subtitle">YOLOv8 Detection • Multi-Feature Extraction • Cosine ReID Gallery</div>
    </div>

    <!-- Camera Auto-Switch Effect Overlay -->
    <div id="switch-overlay" class="camera-switch-overlay"></div>

    <!-- Toast Notifications -->
    <div id="toast-container" class="toast-container"></div>

    <!-- App Shell -->
    <div class="app-layout">

        <!-- Header -->
        <header class="app-header">
            <div class="header-left">
                <div class="logo">
                    <div class="logo-icon">🛰️</div>
                    <span class="logo-text">MCVT Vision</span>
                    <span class="logo-badge">FYP AI ReID</span>
                </div>
            </div>

            <div class="header-center">
                <div style="display:flex;gap:6px;background:var(--bg-card);padding:4px;border-radius:var(--radius-sm);border:1px solid var(--border-color);">
                    <button id="btn-grid-view" class="btn btn-sm btn-primary">
                        <span>▦</span> Grid View (6 Cams)
                    </button>
                    <button id="btn-single-view" class="btn btn-sm">
                        <span>▣</span> Focus Stream
                    </button>
                </div>

                <div style="display:flex;align-items:center;gap:8px;margin-left:14px;background:var(--bg-card);padding:5px 12px;border-radius:20px;border:1px solid var(--border-color);">
                    <label class="toggle-switch">
                        <input type="checkbox" id="auto-switch-toggle">
                        <span class="toggle-slider"></span>
                    </label>
                    <span style="font-size:12px;font-weight:600;color:var(--text-secondary);user-select:none;">
                        Cross-Camera Auto-Switch
                    </span>
                </div>
            </div>

            <div class="header-right">
                <div class="status-indicator">
                    <span class="status-dot online"></span>
                    <span style="font-weight:500;">AI Engine Online</span>
                </div>
                <div style="font-size:11px;color:var(--text-muted);font-family:var(--font-mono);">
                    CVPR MCMT PIPELINE
                </div>
            </div>
        </header>

        <!-- Main Dashboard -->
        <main class="app-main">

            <!-- Camera Panel -->
            <section id="camera-panel" class="camera-panel grid-view">

                <!-- 6-Camera Grid -->
                <div id="camera-grid" class="camera-grid">
                    <?php foreach ($cameras as $cid => $cinfo): ?>
                    <div class="camera-cell" id="cam-cell-<?php echo $cid; ?>"
                         onclick="UI.switchToCamera('<?php echo $cid; ?>')">

                        <!-- Canvas renders WebSocket binary JPEG frames -->
                        <canvas class="camera-cell-stream"
                                id="grid-canvas-<?php echo $cid; ?>"
                                width="640" height="360"></canvas>

                        <div class="camera-cell-live">LIVE</div>
                        <div class="camera-cell-overlay">
                            <div>
                                <span class="camera-cell-name"><?php echo strtoupper($cid); ?></span>
                                <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">
                                    <?php echo htmlspecialchars($cinfo['location']); ?>
                                </div>
                            </div>
                            <span class="camera-cell-badge" id="cam-badge-<?php echo $cid; ?>">0 vehicles</span>
                        </div>
                    </div>
                    <?php endforeach; ?>
                </div>

                <!-- Single Focus Camera View -->
                <div class="main-camera">
                    <div class="main-camera-header">
                        <div class="main-camera-title">
                            <span>SURVEILLANCE FEED:</span>
                            <span id="main-camera-id" class="cam-id">CAM C11</span>
                            <span style="font-size:11px;color:var(--text-muted);font-weight:400;margin-left:8px;">
                                [Multi-Feature Spatial ReID Active]
                            </span>
                        </div>
                        <div class="main-camera-actions">
                            <button id="btn-back-grid" class="btn-back-grid">
                                <span>◂</span> Return to Camera Grid
                            </button>
                        </div>
                    </div>

                    <div class="main-camera-stream">
                        <div id="stream-loading"
                             style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.7);z-index:5;border-radius:8px;">
                            <div style="text-align:center;color:#94a3b8;">
                                <div style="font-size:24px;margin-bottom:8px;">📡</div>
                                <div style="font-size:12px;">Connecting to WebSocket stream...</div>
                            </div>
                        </div>
                        <!-- Canvas receives full-res WebSocket binary frames -->
                        <canvas id="main-camera-canvas"
                                width="960" height="540"
                                style="width:100%;height:100%;object-fit:contain;border-radius:8px;background:#000;">
                        </canvas>
                    </div>

                    <div class="main-camera-footer">
                        <div>
                            <span>⚡ WebSocket Binary Stream • YOLO Detection Bounding Boxes • ReID Tracking IDs</span>
                        </div>
                        <div style="font-family:var(--font-mono);color:var(--accent-primary);">
                            AUTOSWITCH: <span style="color:var(--accent-success)">ENGAGED</span>
                        </div>
                    </div>
                </div>

            </section>

            <!-- Sidebar -->
            <aside class="app-sidebar">

                <!-- Active Target Tracking -->
                <div class="sidebar-section">
                    <div class="sidebar-section-header">
                        <span class="sidebar-section-title">Active Vehicle Target</span>
                        <span style="font-size:12px;color:var(--accent-primary);">Cross-Cam Lock</span>
                    </div>
                    <div class="sidebar-section-content">
                        <div id="tracking-content">
                            <div class="no-tracking">
                                <div class="no-tracking-icon">🎯</div>
                                <p>Click any vehicle card below to track it across cameras.<br>
                                The system will automatically follow and switch camera views.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Vehicle Gallery -->
                <div class="sidebar-section" style="flex:1;display:flex;flex-direction:column;overflow:hidden;">
                    <div class="sidebar-section-header">
                        <span class="sidebar-section-title">Vehicle Gallery (Global ID)</span>
                        <span style="font-size:11px;color:var(--text-muted);font-family:var(--font-mono);">Live Feed</span>
                    </div>
                    <div class="sidebar-section-content" style="flex:1;overflow-y:auto;padding-bottom:8px;">
                        <div id="vehicle-list" class="vehicle-list">
                            <div class="no-tracking">
                                <div class="no-tracking-icon">🚗</div>
                                <p>Scanning video frames for vehicles...</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ReID Analytics -->
                <div class="sidebar-section">
                    <div class="sidebar-section-header">
                        <span class="sidebar-section-title">ReID Performance Analytics</span>
                    </div>
                    <div class="sidebar-section-content">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div id="stat-total-vehicles" class="stat-value">0</div>
                                <div class="stat-label">Unique Vehicles</div>
                            </div>
                            <div class="stat-card">
                                <div id="stat-active-vehicles" class="stat-value">0</div>
                                <div class="stat-label">In Live View</div>
                            </div>
                            <div class="stat-card">
                                <div id="stat-cross-matches" class="stat-value">0</div>
                                <div class="stat-label">Cross-Cam ReIDs</div>
                            </div>
                            <div class="stat-card">
                                <div id="stat-new-vehicles" class="stat-value">0</div>
                                <div class="stat-label">New Signatures</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Event Log -->
                <div class="sidebar-section">
                    <div class="sidebar-section-header">
                        <span class="sidebar-section-title">ReID Event Log</span>
                        <span style="font-size:10px;color:var(--accent-success);">● LIVE</span>
                    </div>
                    <div class="sidebar-section-content">
                        <div id="events-log" class="events-log">
                            <div class="event-item auto_switch">
                                <span class="event-time"><?php echo date('H:i:s'); ?></span>
                                <span class="event-text">System initialized. Awaiting video pipeline startup.</span>
                            </div>
                        </div>
                    </div>
                </div>

            </aside>
        </main>
    </div>

    <!-- Frontend Controller -->
    <script src="assets/js/app.js"></script>
</body>
</html>
