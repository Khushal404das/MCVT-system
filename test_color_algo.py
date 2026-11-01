import os, cv2, numpy as np
import sys
sys.path.insert(0, r"c:\Users\vision Mixer\Desktop\FYP_PROJECT-main\mcvt-system\backend")
# pyrefly: ignore[missing-import]
from vehicle_detector import VehicleDetector

def classify_vehicle_color(crop):
    """
    Robust vehicle color classification designed for outdoor surveillance.
    Filters out windshield/sky reflection and distinguishes neutral 
    tones (white, silver, grey, black) from chromatic colors.
    """
    if crop is None or crop.size == 0:
        return 'unknown'
        
    h, w = crop.shape[:2]
    if h < 15 or w < 15:
        return 'unknown'

    # Focus on the lower 60% of the vehicle (doors, bumpers, hood/trunk)
    # avoiding the upper 35% where windows and sky reflections dominate.
    body = crop[int(h * 0.35):int(h * 0.92), int(w * 0.12):int(w * 0.88)]
    if body.size == 0:
        body = crop

    # Convert to float BGR
    b = body[:, :, 0].astype(np.float32)
    g = body[:, :, 1].astype(np.float32)
    r = body[:, :, 2].astype(np.float32)

    # Convert to HSV
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # Calculate statistics
    mean_v = np.mean(v_chan)
    median_v = np.median(v_chan)
    mean_s = np.mean(s_chan)
    median_s = np.median(s_chan)

    # Chroma: difference between max and min color channels
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    chroma = max_c - min_c
    mean_chroma = np.mean(chroma)
    median_chroma = np.median(chroma)

    # Effective brightness (perceptual luminance)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    mean_lum = np.mean(luminance)

    # 1. Neutral / Achromatic detection:
    # In outdoor daylight, camera white-balance imparts a slight blue cast
    # (Hue 105-120) with saturation up to 60 even on neutral surfaces.
    # True chromatic colors have high chroma (> 35) or high saturation (> 75).
    is_neutral = False
    
    # Very low chroma is always neutral
    if median_chroma < 26 and mean_chroma < 30:
        is_neutral = True
    elif median_s < 45 and median_chroma < 32:
        is_neutral = True
    # For blue-tinted daylight (Hue around 105-125), require much higher chroma for real blue
    elif 95 <= np.median(h_chan) <= 130:
        # Daylight skylight cast: real blue cars have strong blue dominance:
        # b - r > 40 and b - g > 25 and median_chroma > 36
        b_dominance = np.mean(b - np.maximum(r, g))
        if b_dominance < 22 or median_chroma < 36 or median_s < 75:
            is_neutral = True

    if is_neutral:
        if mean_lum < 75 or median_v < 72:
            return 'black'
        elif mean_lum > 155 or median_v > 160:
            return 'white'
        elif mean_lum > 115:
            return 'silver'
        else:
            return 'silver'  # Silver / grey

    # 2. Chromatic colors (strong color presence)
    # Mask out shadows (very low V) and specular highlights (very low S)
    valid_mask = (v_chan > 40) & (s_chan > 40)
    if np.sum(valid_mask) > 10:
        valid_hues = h_chan[valid_mask]
        dominant_h = np.median(valid_hues)
    else:
        dominant_h = np.median(h_chan)

    if dominant_h < 10 or dominant_h > 170:
        return 'red'
    elif dominant_h < 25:
        return 'orange'
    elif dominant_h < 35:
        return 'yellow'
    elif dominant_h < 85:
        return 'green'
    elif dominant_h < 132:
        return 'blue'
    elif dominant_h < 165:
        return 'purple'

    return 'unknown'

# Test on key frames
det = VehicleDetector(model_path='yolov8n.pt', confidence=0.2)
tests = [
    ('dataset/camera_c11.mp4', 500, 'Expected: silver/white'),
    ('dataset/camera_c11.mp4', 3000, 'Expected: silver/white (Mehran)'),
    ('dataset/camera_c11.mp4', 3500, 'Expected: black (SUV)'),
    ('dataset/camera_c11.mp4', 3532, 'Expected: black (SUV)'),
    ('dataset/camera_c21.mp4', 2702, 'Expected: black (SUV)'),
    ('dataset/camera_c21.mp4', 3000, 'Expected: black (SUV)'),
    ('dataset/camera_c21.mp4', 3500, 'Expected: black (SUV)'),
    ('dataset/camera_c41.mp4', 500, 'Expected: white/silver & black'),
    ('dataset/camera_c51.mp4', 1500, 'Expected: orange / green / red'),
    ('dataset/camera_c61.mp4', 500, 'Expected: black'),
]

for cam_rel, fidx, exp in tests:
    cap = cv2.VideoCapture(os.path.join(r"c:\Users\vision Mixer\Desktop\FYP_PROJECT-main", cam_rel))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.resize(frame, (960, 540))
    res = det.detect_and_crop(frame)
    cam_name = os.path.basename(cam_rel)
    print(f"\n--- {cam_name} frame {fidx} ({exp}) ---")
    for d, crop in res:
        col = classify_vehicle_color(crop)
        print(f"  bbox={d['bbox']} cls={d['class_name']} => detected_color={col}")
