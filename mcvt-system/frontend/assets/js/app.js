/**
 * MCVT Vision – Frontend Controller (PHP + WebSocket Edition)
 * Streaming: ws://localhost:5001  (binary JPEG push → <canvas>)
 * REST API:  http://localhost:5000 (JSON: vehicles, tracking, stats)
 */

// ── Config injected by PHP ──────────────────────────────────────
const API_BASE = (typeof PHP_CONFIG !== 'undefined') ? PHP_CONFIG.api_base : 'http://localhost:5000';
const WS_BASE  = (typeof PHP_CONFIG !== 'undefined') ? PHP_CONFIG.ws_base  : 'ws://localhost:5001';
const CAM_IDS  = (typeof PHP_CONFIG !== 'undefined') ? PHP_CONFIG.cameras  : ['c11','c21','c31','c41','c51','c61'];

// ── Application State ────────────────────────────────────────────
const AppState = {
    cameras:            {},
    activeCamera:       null,
    viewMode:           'grid',   // 'grid' | 'single'
    vehicles:           [],
    trackedVehicle:     null,
    autoSwitch:         false,
    events:             [],
    lastEventTimestamp: 0,
    isSystemReady:      false,
    pollInterval:       null,
    eventPollInterval:  null,
    _gridWs:            {},       // { camId: { ws, running, stop } }
    _singleWs:          null,     // { ws, running, stop }
};

// ── REST API Client ──────────────────────────────────────────────
const API = {
    async get(endpoint, timeoutMs = 5000) {
        try {
            const ctrl = new AbortController();
            const t = setTimeout(() => ctrl.abort(), timeoutMs);
            const res = await fetch(`${API_BASE}${endpoint}`, { signal: ctrl.signal });
            clearTimeout(t);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error(`API GET ${endpoint}:`, err.message);
            return null;
        }
    },

    async post(endpoint, data = {}) {
        try {
            const res = await fetch(`${API_BASE}${endpoint}`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(data),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error(`API POST ${endpoint}:`, err.message);
            return null;
        }
    },
};

// ── WebSocket Stream Helper ──────────────────────────────────────
/**
 * Opens a WebSocket to ws://localhost:5001/<route>/<camId>
 * On each binary message, decode Blob → ImageBitmap → draw on canvas.
 * Auto-reconnects every 500ms on unexpected close.
 *
 * @param {string} route    'stream' (full-res) | 'thumb' (grid thumbnail)
 * @param {string} camId    Camera ID e.g. 'c11'
 * @param {HTMLCanvasElement} canvas  Target canvas element
 * @returns {{ stop: Function }}     Call stop() to close cleanly
 */
function openCameraWs(route, camId, canvas) {
    const ctx = canvas.getContext('2d');
    let running   = true;
    let rendering = false;
    let ws        = null;

    const connect = () => {
        if (!running) return;
        ws = new WebSocket(`${WS_BASE}/${route}/${camId}`);
        ws.binaryType = 'blob';

        ws.onmessage = async (evt) => {
            if (!running || rendering) return;
            rendering = true;
            try {
                const bitmap = await createImageBitmap(evt.data);
                if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
                    canvas.width  = bitmap.width;
                    canvas.height = bitmap.height;
                }
                ctx.drawImage(bitmap, 0, 0);
                bitmap.close();
            } catch (_) {}
            rendering = false;
        };

        ws.onerror = () => {};

        ws.onclose = () => {
            // Auto-reconnect after 500ms unless deliberately stopped
            if (running) setTimeout(connect, 500);
        };
    };

    connect();

    return {
        stop() {
            running = false;
            if (ws) { ws.onclose = null; ws.close(); ws = null; }
            // Clear canvas to black on disconnect
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    };
}

// ── UI Controller ────────────────────────────────────────────────
const UI = {

    // ── Init ──────────────────────────────────────────────────────
    init() {
        document.getElementById('btn-grid-view')
            ?.addEventListener('click', () => this.setViewMode('grid'));
        document.getElementById('btn-single-view')
            ?.addEventListener('click', () => this.setViewMode('single'));
        document.getElementById('btn-back-grid')
            ?.addEventListener('click', () => this.setViewMode('grid'));

        document.getElementById('auto-switch-toggle')
            ?.addEventListener('change', (e) => this.toggleAutoSwitch(e.target.checked));

        document.getElementById('btn-stop-tracking')
            ?.addEventListener('click', () => this.stopTracking());

        this.waitForSystem();
    },

    // ── Wait for backend ready ────────────────────────────────────
    async waitForSystem() {
        const overlay  = document.getElementById('loading-overlay');
        const loadText = document.getElementById('loading-text');

        let attempts = 0;
        while (true) {
            try {
                const data = await API.get('/api/health', 3000);
                if (data && data.ready) break;
            } catch (_) {}
            attempts++;
            if (loadText) {
                loadText.textContent = `Connecting to AI engine... (attempt ${attempts})`;
            }
            await new Promise(r => setTimeout(r, 2000));
        }

        AppState.isSystemReady = true;
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => overlay.remove(), 600);
        }

        await this.loadCameras();
        this.startPolling();
        await this.loadVehicles();
        await this.updateStats();
    },

    // ── Camera Setup ──────────────────────────────────────────────
    async loadCameras() {
        const data = await API.get('/api/cameras');
        if (!data) return;

        AppState.cameras      = data.cameras || {};
        AppState.activeCamera = data.active_camera || CAM_IDS[0];

        // Grid is rendered statically by PHP; just start WebSocket streams
        if (AppState.viewMode === 'grid') {
            this._startGridStreams(CAM_IDS);
        }
    },

    // ── Grid WebSocket Streams ────────────────────────────────────
    /**
     * Opens one WebSocket per camera in the 6-camera grid.
     * Each WS pushes binary JPEG frames which are drawn to the camera's <canvas>.
     */
    _startGridStreams(cameraIds) {
        if (!AppState._gridWs) AppState._gridWs = {};

        cameraIds.forEach((camId, idx) => {
            if (AppState._gridWs[camId]?.running) return;

            const canvas = document.getElementById(`grid-canvas-${camId}`);
            if (!canvas) return;

            // Stagger connections 60ms apart to avoid burst
            const handle = setTimeout(() => {
                const stream = openCameraWs('thumb', camId, canvas);
                AppState._gridWs[camId] = {
                    running: true,
                    stop: () => {
                        AppState._gridWs[camId].running = false;
                        stream.stop();
                    }
                };
            }, idx * 60);

            // Store placeholder so _stopGridStreams can cancel the timeout too
            AppState._gridWs[camId] = {
                running: false,
                stop: () => clearTimeout(handle)
            };
        });
    },

    _stopGridStreams() {
        if (!AppState._gridWs) return;
        Object.values(AppState._gridWs).forEach(s => s?.stop?.());
        AppState._gridWs = {};
    },

    // ── Single Camera WebSocket Stream ────────────────────────────
    _startSingleStream(cameraId) {
        this._stopSingleStream();

        const canvas = document.getElementById('main-camera-canvas');
        const loader = document.getElementById('stream-loading');
        if (!canvas) return;

        if (loader) loader.style.display = 'flex';

        // Track first frame to hide loader
        const origConnect = openCameraWs;
        const ctx = canvas.getContext('2d');
        let firstFrame = true;
        let running = true;
        let rendering = false;
        let ws = null;

        const connect = () => {
            if (!running) return;
            ws = new WebSocket(`${WS_BASE}/stream/${cameraId}`);
            ws.binaryType = 'blob';

            ws.onmessage = async (evt) => {
                if (!running || rendering) return;
                rendering = true;
                try {
                    const bitmap = await createImageBitmap(evt.data);
                    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
                        canvas.width  = bitmap.width;
                        canvas.height = bitmap.height;
                    }
                    ctx.drawImage(bitmap, 0, 0);
                    bitmap.close();
                    if (firstFrame) {
                        firstFrame = false;
                        if (loader) loader.style.display = 'none';
                    }
                } catch (_) {}
                rendering = false;
            };

            ws.onerror = () => {};
            ws.onclose = () => { if (running) setTimeout(connect, 500); };
        };

        connect();

        AppState._singleWs = {
            stop() {
                running = false;
                if (ws) { ws.onclose = null; ws.close(); ws = null; }
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (loader) loader.style.display = 'flex';
            }
        };
    },

    _stopSingleStream() {
        AppState._singleWs?.stop();
        AppState._singleWs = null;
    },

    // ── View Mode (grid / single) ─────────────────────────────────
    setViewMode(mode) {
        AppState.viewMode = mode;
        const panel = document.getElementById('camera-panel');
        if (!panel) return;

        panel.classList.remove('grid-view', 'single-view');
        panel.classList.add(`${mode}-view`);

        document.getElementById('btn-grid-view')
            ?.classList.toggle('btn-primary', mode === 'grid');
        document.getElementById('btn-single-view')
            ?.classList.toggle('btn-primary', mode === 'single');

        if (mode === 'single' && AppState.activeCamera) {
            this._stopGridStreams();
            this._startSingleStream(AppState.activeCamera);

            const title = document.getElementById('main-camera-id');
            if (title) title.textContent = `CAM ${AppState.activeCamera.toUpperCase()}`;

        } else if (mode === 'grid') {
            this._stopSingleStream();
            setTimeout(() => this._startGridStreams(CAM_IDS), 60);
        }
    },

    // ── Switch Camera ─────────────────────────────────────────────
    switchToCamera(cameraId) {
        AppState.activeCamera = cameraId;

        document.querySelectorAll('.camera-cell').forEach(c => c.classList.remove('active'));
        document.getElementById(`cam-cell-${cameraId}`)?.classList.add('active');

        if (AppState.viewMode === 'single') {
            // Just swap the stream — no view-mode change
            this._startSingleStream(cameraId);
            const title = document.getElementById('main-camera-id');
            if (title) title.textContent = `CAM ${cameraId.toUpperCase()}`;
        } else {
            this.setViewMode('single');
        }

        API.post('/api/camera/switch', { camera_id: cameraId });
        this.showSwitchAnimation();
        this.addEvent('camera_switch', `Switched to Camera ${cameraId.toUpperCase()}`);
    },

    // ── Vehicle Tracking ──────────────────────────────────────────
    async loadVehicles() {
        const data = await API.get('/api/vehicles');
        if (!data) return;
        AppState.vehicles = data.vehicles || [];
        this.renderVehicleList();
        this.updateCameraBadges();
    },

    renderVehicleList() {
        const list = document.getElementById('vehicle-list');
        if (!list) return;

        if (!AppState.vehicles.length) {
            list.innerHTML = `
                <div class="no-tracking">
                    <div class="no-tracking-icon">🚗</div>
                    <p>No vehicles detected yet.<br>Waiting for detections...</p>
                </div>`;
            return;
        }

        // Sort by global_id ascending — lowest ID = earliest spotted vehicle
        const sorted = [...AppState.vehicles].sort((a, b) => a.global_id - b.global_id);
        list.innerHTML = sorted.slice(0, 50).map(v => {
            const isTracked = AppState.trackedVehicle?.global_id === v.global_id;
            const hex = this.getColorHex(v.color_name);
            // Show all cameras this physical vehicle has been seen in
            const allCams = (v.cameras_visited || [v.current_camera]).filter(Boolean);
            const camBadges = allCams.map(c =>
                `<span class="cam-badge ${c === v.current_camera ? 'active' : ''}">${c.toUpperCase()}</span>`
            ).join('');
            return `
                <div class="vehicle-card ${isTracked ? 'tracked' : ''}"
                     onclick="UI.trackVehicle(${v.global_id})"
                     title="G${v.global_id} — ${v.color_name} ${v.class_name} seen in: ${allCams.join(', ')}">
                    <div class="vehicle-color-dot" style="background:${hex}"></div>
                    <div class="vehicle-info">
                        <div class="vehicle-id">G${v.global_id}
                            <span class="gid-unique-badge" title="Unique global ID — same across all cameras">🔑</span>
                        </div>
                        <div class="vehicle-meta">
                            <span>${v.class_name || '—'}</span><span>•</span>
                            <span>${v.color_name || '—'}</span>
                        </div>
                        <div class="vehicle-cameras">
                            ${camBadges}
                        </div>
                    </div>
                    <div class="vehicle-actions">
                        ${isTracked
                            ? `<button class="btn btn-sm btn-danger"
                                       onclick="event.stopPropagation();UI.stopTracking()">✕</button>`
                            : `<button class="btn btn-sm btn-success"
                                       onclick="event.stopPropagation();UI.trackVehicle(${v.global_id})">🎯</button>`
                        }
                    </div>
                </div>`;
        }).join('');
    },

    async trackVehicle(globalId) {
        const result = await API.post('/api/track', { global_id: globalId });
        if (!result?.success) return;

        const toggle = document.getElementById('auto-switch-toggle');
        if (toggle) toggle.checked = true;
        AppState.autoSwitch = true;

        this.showToast(`Tracking Vehicle G${globalId} — Auto-switch enabled`, 'success');
        this.addEvent('track_start', `Started tracking Vehicle G${globalId}`);
        this.checkTrackingStatus();
    },

    async stopTracking() {
        await API.post('/api/track/stop');
        AppState.trackedVehicle = null;
        this.renderTrackingPanel();
        this.renderVehicleList();
        this.showToast('Vehicle tracking stopped', 'info');
        this.addEvent('track_stop', 'Tracking stopped');
    },

    async checkTrackingStatus() {
        const data = await API.get('/api/track/status');
        if (!data) return;

        if (data.tracking) {
            AppState.trackedVehicle = data.vehicle;

            if (data.vehicle.current_camera !== AppState.activeCamera && AppState.autoSwitch) {
                this.switchToCamera(data.vehicle.current_camera);
                this.showToast(
                    `🎯 Vehicle G${data.vehicle.global_id} spotted in Camera ${data.vehicle.current_camera.toUpperCase()}!`,
                    'success'
                );
            }

            document.querySelectorAll('.camera-cell').forEach(c => c.classList.remove('has-tracked-vehicle'));
            if (data.vehicle.current_camera) {
                document.getElementById(`cam-cell-${data.vehicle.current_camera}`)
                    ?.classList.add('has-tracked-vehicle');
            }
        } else {
            if (AppState.trackedVehicle) {
                const prevGid = AppState.trackedVehicle.global_id;
                AppState.trackedVehicle = null;
                AppState.autoSwitch = false;
                const toggle = document.getElementById('auto-switch-toggle');
                if (toggle) toggle.checked = false;
                this.showToast(`🏁 Vehicle G${prevGid} journey completed. Auto-switch stopped.`, 'info');
                document.querySelectorAll('.camera-cell').forEach(c => c.classList.remove('has-tracked-vehicle'));
            }
            AppState.trackedVehicle = null;
        }

        this.renderTrackingPanel();
    },

    renderTrackingPanel() {
        const panel = document.getElementById('tracking-content');
        if (!panel) return;

        if (!AppState.trackedVehicle) {
            panel.innerHTML = `
                <div class="no-tracking">
                    <div class="no-tracking-icon">🎯</div>
                    <p>Click a vehicle to track it across cameras.<br>
                    The system will auto-switch to follow it.</p>
                </div>`;
            return;
        }

        const v = AppState.trackedVehicle;
        const hex = this.getColorHex(v.color_name);
        const camPath = (v.cameras_visited || [])
            .map(c => `<span class="camera-path-item">CAM ${c.toUpperCase()}</span>`)
            .join('<span class="camera-path-arrow">→</span>');

        panel.innerHTML = `
            <div class="tracking-vehicle-info">
                <div class="tracking-label">⚡ ACTIVELY TRACKING</div>
                <div class="tracking-gid">Vehicle G${v.global_id}</div>
                <div class="tracking-details">
                    <div class="tracking-detail-item">
                        <div class="label">Type</div><div class="value">${v.class_name}</div>
                    </div>
                    <div class="tracking-detail-item">
                        <div class="label">Color</div>
                        <div class="value" style="display:flex;align-items:center;gap:6px">
                            <span style="width:12px;height:12px;border-radius:3px;background:${hex};display:inline-block"></span>
                            ${v.color_name}
                        </div>
                    </div>
                    <div class="tracking-detail-item">
                        <div class="label">Current Camera</div>
                        <div class="value" style="color:var(--accent-primary)">CAM ${v.current_camera?.toUpperCase()}</div>
                    </div>
                    <div class="tracking-detail-item">
                        <div class="label">Age</div><div class="value">${v.age}s</div>
                    </div>
                </div>
                ${(v.cameras_visited || []).length > 0 ? `
                    <div style="margin-top:12px">
                        <div class="label" style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">Camera Path</div>
                        <div class="camera-path">${camPath}</div>
                    </div>` : ''}
            </div>
            <button class="btn btn-danger" onclick="UI.stopTracking()" style="width:100%;margin-top:8px">
                ✕ Stop Tracking
            </button>`;
    },

    async toggleAutoSwitch(enabled) {
        AppState.autoSwitch = enabled;
        await API.post('/api/auto-switch', { enabled });
        this.showToast(enabled ? 'Auto-switch enabled' : 'Auto-switch disabled', 'info');
    },

    updateCameraBadges() {
        const byCamera = {};
        AppState.vehicles.forEach(v => {
            if (v.current_camera) byCamera[v.current_camera] = (byCamera[v.current_camera] || 0) + 1;
        });
        CAM_IDS.forEach(id => {
            const badge = document.getElementById(`cam-badge-${id}`);
            if (badge) {
                const n = byCamera[id] || 0;
                badge.textContent = `${n} vehicle${n !== 1 ? 's' : ''}`;
            }
        });
    },

    // ── Stats ─────────────────────────────────────────────────────
    async updateStats() {
        const data = await API.get('/api/reid/stats');
        if (!data) return;
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('stat-total-vehicles',  data.total_vehicles      || 0);
        set('stat-active-vehicles', data.active_vehicles     || 0);
        set('stat-cross-matches',   data.cross_camera_matches || 0);
        set('stat-new-vehicles',    data.new_vehicles        || 0);
    },

    // ── Events ────────────────────────────────────────────────────
    async loadEvents() {
        const data = await API.get(`/api/events?since=${AppState.lastEventTimestamp}`);
        if (!data?.events?.length) return;
        AppState.lastEventTimestamp = data.events[data.events.length - 1].timestamp;
        data.events.forEach(e => AppState.events.unshift(e));
        AppState.events = AppState.events.slice(0, 50);
        this.renderEvents();
    },

    addEvent(type, message) {
        AppState.events.unshift({
            type,
            time_str:  new Date().toLocaleTimeString(),
            data:      { message },
            timestamp: Date.now() / 1000,
        });
        AppState.events = AppState.events.slice(0, 50);
        this.renderEvents();
    },

    renderEvents() {
        const log = document.getElementById('events-log');
        if (!log) return;
        log.innerHTML = AppState.events.slice(0, 15).map(e => {
            const msg = e.data?.message || e.data?.reason || e.type;
            return `<div class="event-item ${e.type}">
                        <span class="event-time">${e.time_str}</span>
                        <span class="event-text">${msg}</span>
                    </div>`;
        }).join('');
    },

    // ── REST Polling (vehicles / tracking / stats / events) ───────
    startPolling() {
        AppState.pollInterval = setInterval(async () => {
            if (!AppState.isSystemReady) return;
            await this.loadVehicles();
            await this.checkTrackingStatus();
            await this.updateStats();
        }, 1500);

        // Real-time auto-switch: checks every 400ms when a target car is being actively tracked
        setInterval(async () => {
            if (!AppState.isSystemReady || !AppState.trackedVehicle || !AppState.autoSwitch) return;
            await this.checkTrackingStatus();
        }, 400);

        AppState.eventPollInterval = setInterval(async () => {
            if (!AppState.isSystemReady) return;
            await this.loadEvents();
        }, 2000);
    },

    // ── Helpers ───────────────────────────────────────────────────
    getColorHex(colorName) {
        const map = {
            red: '#ef4444', blue: '#3b82f6', green: '#22c55e',
            white: '#f1f5f9', black: '#1e293b', silver: '#94a3b8',
            yellow: '#eab308', orange: '#f97316', purple: '#a855f7',
            unknown: '#64748b',
        };
        return map[colorName] || map.unknown;
    },

    showSwitchAnimation() {
        const overlay = document.getElementById('switch-overlay');
        if (!overlay) return;
        overlay.classList.add('active');
        setTimeout(() => overlay.classList.remove('active'), 600);
    },

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        const icons = { success: '✅', info: 'ℹ️', warning: '⚠️' };
        toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(50px)';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    },
};

// ── Boot ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => UI.init());
