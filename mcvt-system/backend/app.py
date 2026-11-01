"""
Flask API Server for Multi-Camera Vehicle Tracking System
Provides REST API endpoints and MJPEG video streams for the PHP frontend.
"""

import os
import io
import sys
import time
import json
import threading
import asyncio
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    websockets = None
import torch
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(1)
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

# Force UTF-8 output to avoid UnicodeEncodeError on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Use a mutable container so background thread writes are visible to Flask request threads
_state = {'processor': None}

def get_processor():
    """Thread-safe accessor for the global processor instance."""
    return _state['processor']

# ============================================================
# Configuration
# ============================================================

# Camera video file mapping
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset')

CAMERA_CONFIGS = {
    'c11': os.path.join(DATASET_DIR, 'camera_c11.mp4'),
    'c21': os.path.join(DATASET_DIR, 'camera_c21.mp4'),
    'c31': os.path.join(DATASET_DIR, 'camera_c31.mp4'),
    'c41': os.path.join(DATASET_DIR, 'camera_c41.mp4'),
    'c51': os.path.join(DATASET_DIR, 'camera_c51.mp4'),
    'c61': os.path.join(DATASET_DIR, 'camera_c61.mp4'),
}

PROCESS_FPS = 160  # Processing framerate (160 FPS streaming)

# ============================================================
# Video Streaming Endpoints (MJPEG)
# ============================================================

def generate_mjpeg(camera_id):
    """Generator function for MJPEG streaming."""
    while True:
        processor = get_processor()
        if processor is None:
            time.sleep(0.5)
            continue
        
        jpeg = processor.get_frame_jpeg(camera_id, quality=75)
        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        
        time.sleep(1.0 / 160)  # 160 FPS streaming


def generate_thumbnail(camera_id):
    """Generator function for thumbnail MJPEG streaming."""
    while True:
        processor = get_processor()
        if processor is None:
            time.sleep(1)
            continue
        
        jpeg = processor.get_thumbnail_jpeg(camera_id, quality=75)
        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        
        time.sleep(1.0 / 160)  # 160 FPS for smooth thumbnails


@app.route('/stream/<camera_id>')
def video_stream(camera_id):
    """MJPEG stream for a single camera (full resolution)."""
    response = Response(
        generate_mjpeg(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/thumbnail/<camera_id>')
def thumbnail_stream(camera_id):
    """MJPEG stream for camera thumbnail."""
    response = Response(
        generate_thumbnail(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/snapshot/<camera_id>')
def snapshot(camera_id):
    """Get a single JPEG snapshot from a camera."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503

    jpeg = processor.get_frame_jpeg(camera_id, quality=90)
    if jpeg is None:
        return jsonify({'error': f'Camera {camera_id} not found'}), 404

    return Response(jpeg, mimetype='image/jpeg')


@app.route('/thumb/<camera_id>')
def thumb_snapshot(camera_id):
    """Single-frame thumbnail JPEG for grid polling (short-lived, no persistent connection)."""
    processor = get_processor()
    if processor is None:
        return Response(status=503)

    jpeg = processor.get_thumbnail_jpeg(camera_id, size=(640, 360), quality=72)
    if jpeg is None:
        return Response(status=404)

    response = Response(jpeg, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response




# ============================================================
# WebSocket Streaming Server (Port 5001)
# Push binary JPEG frames directly to client with zero HTTP overhead
# ============================================================

WS_PORT = 5001

async def ws_camera_handler(websocket):
    """
    WebSocket streaming handler:
    Routes:
      - /stream/<camera_id> -> full-resolution HD stream
      - /thumb/<camera_id>  -> thumbnail stream for grid view
    Pushes cached binary JPEG frames as soon as each new frame is rendered.
    """
    try:
        # Compatible with all websockets library versions
        path = getattr(websocket, 'path', None)
        if path is None and hasattr(websocket, 'request'):
            path = getattr(websocket.request, 'path', '')
        if not path:
            path = ''
        path = path.strip('/')
        parts = path.split('/')
        if not parts:
            return
        
        mode = parts[0] if len(parts) > 1 else 'stream'
        camera_id = parts[-1].lower()
        last_version = -1
        
        while True:
            proc = get_processor()
            if proc is not None and camera_id in proc.cameras:
                cam = proc.cameras[camera_id]
                cur_ver = cam.frame_version
                if cur_ver != last_version:
                    last_version = cur_ver
                    with cam._frame_lock:
                        jpeg = cam.current_thumb_bytes if mode == 'thumb' else cam.current_full_bytes
                    if jpeg:
                        await websocket.send(jpeg)
            
            await asyncio.sleep(0.008)  # ~120 Hz reactive check, zero redundant encoding
            
    except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
        pass
    except Exception:
        pass


def run_websocket_server():
    """Runs asyncio WebSocket server on WS_PORT in a dedicated background daemon thread."""
    if not HAS_WEBSOCKETS or websockets is None:
        print("[Server] Note: websockets library not available, WebSocket streaming disabled.")
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def serve():
        async with websockets.serve(ws_camera_handler, "0.0.0.0", WS_PORT):
            print(f"[Server] WebSocket streaming server active on ws://0.0.0.0:{WS_PORT}")
            await asyncio.Future()
            
    try:
        loop.run_until_complete(serve())
    except Exception as e:
        print(f"[Server] WebSocket server error: {e}")


# ============================================================
# Camera Control Endpoints
# ============================================================

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get list of all cameras and their status."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503
    
    status = processor.get_system_status()
    return jsonify({
        'cameras': status['cameras'],
        'active_camera': status['active_camera'],
        'camera_ids': list(status['cameras'].keys())
    })


@app.route('/api/camera/switch', methods=['POST'])
def switch_camera():
    """Switch the active camera view."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503
    
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if camera_id and processor.switch_camera(camera_id):
        return jsonify({'success': True, 'active_camera': camera_id})
    
    return jsonify({'error': f'Invalid camera: {camera_id}'}), 400


# ============================================================
# Vehicle Tracking Endpoints
# ============================================================

@app.route('/api/vehicles', methods=['GET'])
def get_vehicles():
    """Get all tracked vehicles across cameras."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503
    
    camera_id = request.args.get('camera_id')
    
    if camera_id:
        vehicles = processor.get_vehicles_in_camera(camera_id)
    else:
        vehicles = processor.reid_engine.get_all_vehicles()
    
    return jsonify({'vehicles': vehicles})


@app.route('/api/vehicle/<int:global_id>', methods=['GET'])
def get_vehicle(global_id):
    """Get details of a specific vehicle."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503
    
    entry = processor.reid_engine.gallery.get_entry(global_id)
    if entry:
        return jsonify({'vehicle': entry.to_dict()})
    
    return jsonify({'error': f'Vehicle G{global_id} not found'}), 404


@app.route('/api/track', methods=['POST'])
def track_vehicle():
    """Start tracking a vehicle across cameras."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503
    
    data = request.get_json()
    global_id = data.get('global_id')
    
    if global_id is None:
        return jsonify({'error': 'global_id required'}), 400
    
    processor.track_vehicle(int(global_id))
    processor.enable_auto_switch(True)
    
    return jsonify({
        'success': True,
        'tracking': global_id,
        'message': f'Now tracking vehicle G{global_id}'
    })


@app.route('/api/track/stop', methods=['POST'])
def stop_tracking():
    """Stop tracking the current vehicle."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503
    
    processor.stop_tracking()
    return jsonify({'success': True, 'message': 'Tracking stopped'})


@app.route('/api/track/status', methods=['GET'])
def track_status():
    """Get the status of the currently tracked vehicle."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503
    
    info = processor.reid_engine.get_tracked_vehicle_info()
    if info:
        return jsonify({
            'tracking': True,
            'vehicle': info,
            'active_camera': processor.active_camera,
            'auto_switch': processor.auto_switch_enabled
        })
    
    return jsonify({'tracking': False})


@app.route('/api/auto-switch', methods=['POST'])
def toggle_auto_switch():
    """Enable/disable automatic camera switching."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503
    
    data = request.get_json()
    enabled = data.get('enabled', True)
    processor.enable_auto_switch(enabled)
    
    return jsonify({'success': True, 'auto_switch': enabled})


# ============================================================
# System Status & Events
# ============================================================

@app.route('/api/status', methods=['GET'])
def system_status():
    """Get full system status."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized', 'ready': False}), 503
    
    status = processor.get_system_status()
    status['ready'] = True
    return jsonify(status)


@app.route('/api/events', methods=['GET'])
def get_events():
    """Get recent events (camera switches, re-identifications)."""
    processor = get_processor()
    if processor is None:
        return jsonify({'events': []})
    
    since = request.args.get('since', type=float)
    events = processor.get_events(since)
    return jsonify({'events': events})


@app.route('/api/reid/stats', methods=['GET'])
def reid_stats():
    """Get re-identification statistics."""
    processor = get_processor()
    if processor is None:
        return jsonify({'error': 'System not initialized'}), 503
    
    stats = processor.reid_engine.get_stats()
    return jsonify(stats)


# ============================================================
# Health Check
# ============================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'ready': get_processor() is not None,
        'timestamp': time.time()
    })


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve frontend static assets."""
    return send_from_directory(os.path.join(FRONTEND_DIR, 'assets'), filename)


@app.route('/ui')
@app.route('/dashboard')
def serve_dashboard():
    """Serve web dashboard UI directly."""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/')
def index():
    """API index and status."""
    return jsonify({
        'service': 'Multi-Camera Vehicle Tracking API',
        'version': '1.0.0',
        'dashboard_ui': '/ui',
        'endpoints': {
            'streams': '/stream/<camera_id>',
            'thumbnails': '/thumbnail/<camera_id>',
            'cameras': '/api/cameras',
            'vehicles': '/api/vehicles',
            'track': '/api/track',
            'status': '/api/status',
            'events': '/api/events'
        }
    })


# ============================================================
# Initialization & Startup
# ============================================================

def init_processor():
    """Initialize the video processor in a background thread."""
    try:
        # Verify video files exist
        valid_configs = {}
        for cam_id, path in CAMERA_CONFIGS.items():
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                valid_configs[cam_id] = abs_path
                print(f"  [OK] Camera {cam_id}: {abs_path}")
            else:
                print(f"  [X]  Camera {cam_id}: NOT FOUND at {abs_path}")
        
        if not valid_configs:
            print("\n[ERROR] No valid camera feeds found!")
            print(f"Expected videos in: {os.path.abspath(DATASET_DIR)}")
            return
        
        from video_processor import MultiCameraProcessor
        proc = MultiCameraProcessor(valid_configs, process_fps=PROCESS_FPS)
        proc.start()
        _state['processor'] = proc  # Store in mutable dict for thread visibility
        print("\n[Server] Video processing started! processor is ready.")
    except Exception as e:
        import traceback
        print(f"\n[FATAL] init_processor failed: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Multi-Camera Vehicle Tracking System - API Server")
    print("=" * 60)
    print(f"\nDataset directory: {os.path.abspath(DATASET_DIR)}")
    print(f"Processing FPS: {PROCESS_FPS}")
    print("\nChecking camera feeds...")
    
    # Start processor in background thread
    init_thread = threading.Thread(target=init_processor, daemon=True)
    init_thread.start()
    
    # Start WebSocket streaming server (Port 5001)
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()
    
    print("\n[Server] Starting Flask API on http://localhost:5000")
    print("[Server] PHP frontend should connect to this server")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
