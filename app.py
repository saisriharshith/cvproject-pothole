#!/usr/bin/env python3
"""
PyResearch Pothole Detection System - Advanced Computer Vision Dashboard
Supports:
  1. Deep Learning YOLOv12 Detection (best.pt + Supervision Annotator)
  2. Classical Mid-Level Computer Vision Pipeline (Unit 1: Gaussian -> Canny -> Morphology -> Contours)
  3. Real-Time Spatial Gradient Edge Map Inspection
  4. Robust Centroid Object Tracking for Physically Accurate "Total Detected" Counts
  5. Live Telemetry: FPS, Frame Latency, Detection Counters, Road Hazard Alerts
"""

import os
import sys
import time
import math
import threading
import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify, request

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from ultralytics import YOLO
    import ultralytics.nn.modules.block as block

    # Backward-compatibility monkeypatch for legacy AAttn architectures in modern Ultralytics
    _orig_aattn_forward = block.AAttn.forward
    def _compat_aattn_forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'qk') and hasattr(self, 'v') and not hasattr(self, 'qkv'):
            self.qkv = lambda inp: torch.cat([self.qk(inp), self.v(inp)], dim=1)
        return _orig_aattn_forward(self, x)

    block.AAttn.forward = _compat_aattn_forward
except ImportError:
    YOLO = None

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
DEMO_VIDEO = os.path.join(BASE_DIR, "demo.mp4")

def deduplicate_boxes(boxes, confs=None, iou_thresh=0.25, ios_thresh=0.45):
    """
    Suppresses duplicate overlapping or nested bounding boxes in the SAME frame.
    Prevents a single physical pothole from having multiple simultaneous bounding boxes.
    """
    if len(boxes) <= 1:
        return boxes, confs if confs is not None else [1.0] * len(boxes)
    
    if confs is None:
        confs = [1.0] * len(boxes)
        
    indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    kept_boxes = []
    kept_confs = []
    
    for idx in indices:
        b = boxes[idx]
        b_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        duplicate = False
        
        for kb in kept_boxes:
            xA = max(b[0], kb[0])
            yA = max(b[1], kb[1])
            xB = min(b[2], kb[2])
            yB = min(b[3], kb[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            
            if inter > 0:
                kb_area = max(1.0, (kb[2] - kb[0]) * (kb[3] - kb[1]))
                iou = inter / float(b_area + kb_area - inter + 1e-6)
                ios = inter / float(min(b_area, kb_area) + 1e-6)
                if iou > iou_thresh or ios > ios_thresh:
                    duplicate = True
                    break
        if not duplicate:
            kept_boxes.append(b)
            kept_confs.append(confs[idx])
            
    return kept_boxes, kept_confs

def compute_ground_plane_dimensions(box, h_cam=120.0, pitch_deg=14.0, f_y=500.0, v0=180.0):
    """
    Computes real-world physical metric dimensions (Width, Length, Distance, Area)
    from a 2D bounding box using Monocular Ground-Plane Inverse Perspective Mapping (IPM).
    
    Parameters:
      box: [x1, y1, x2, y2] in pixel coordinates
      h_cam: Camera mounting height in cm (default 120cm / 1.2m)
      pitch_deg: Camera pitch down angle in degrees (default 14 deg)
      f_y: Camera focal length in pixels (default 500px for 640x360 dashcam)
      v0: Principal optical center y-coordinate (default 180px)
    """
    x1, y1, x2, y2 = box
    w_px = max(1.0, float(x2 - x1))
    h_px = max(1.0, float(y2 - y1))
    y_base = max(float(y1), float(y2))
    
    # Angle of incidence for ground contact point
    alpha = math.atan((y_base - v0) / f_y)
    phi = max(math.radians(8.0), math.radians(pitch_deg) + alpha)
    Z_cm = h_cam / math.tan(phi)
    
    # Transverse real-world width across road
    w_cm = w_px * (Z_cm / f_y)
    
    # Longitudinal real-world length along road surface (corrected for foreshortening)
    sin_factor = max(0.22, math.sin(phi))
    l_cm = h_px * (Z_cm / f_y) / sin_factor
    
    # Physical road defect constraint (potholes rarely exceed 2.2x width)
    l_cm = min(l_cm, w_cm * 2.2)
    
    # Approximate elliptical defect area
    area_m2 = (math.pi / 4.0) * (w_cm / 100.0) * (l_cm / 100.0)
    
    # ASTM D6433 Pothole Severity Index
    max_dim = max(w_cm, l_cm)
    if max_dim < 30.0:
        sev = "MINOR"
        bgr_color = (60, 200, 60)      # Emerald green
        hex_color = "#2ea043"
    elif max_dim <= 60.0:
        sev = "MODERATE"
        bgr_color = (0, 165, 255)     # Amber orange
        hex_color = "#d29922"
    else:
        sev = "SEVERE"
        bgr_color = (50, 50, 240)      # Vivid red
        hex_color = "#f85149"
        
    return {
        "width_cm": round(w_cm, 1),
        "length_cm": round(l_cm, 1),
        "area_m2": round(area_m2, 2),
        "distance_m": round(Z_cm / 100.0, 2),
        "severity": sev,
        "bgr_color": bgr_color,
        "hex_color": hex_color
    }

def draw_pothole_annotation(annotated, box, tid, score, dim):
    """
    Renders an engineering-grade HUD annotation badge including:
    - Color-coded severity bounding box (Red: Severe, Amber: Moderate, Green: Minor)
    - Corner brackets for tactical HUD feel
    - Top header badge: #{tid} | {W}x{L}cm [{SEVERITY}]
    - Bottom sub-badge: Dist: {Z}m | Area: {A}m²
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    color = dim["bgr_color"]
    w_cm = dim["width_cm"]
    l_cm = dim["length_cm"]
    dist_m = dim["distance_m"]
    area_m2 = dim["area_m2"]
    sev = dim["severity"]
    
    # 1. Bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    
    # Corner brackets
    bracket_len = min(14, (x2 - x1) // 4, (y2 - y1) // 4)
    if bracket_len > 4:
        # Top-left
        cv2.line(annotated, (x1, y1), (x1 + bracket_len, y1), color, 3)
        cv2.line(annotated, (x1, y1), (x1, y1 + bracket_len), color, 3)
        # Top-right
        cv2.line(annotated, (x2, y1), (x2 - bracket_len, y1), color, 3)
        cv2.line(annotated, (x2, y1), (x2, y1 + bracket_len), color, 3)
        # Bottom-left
        cv2.line(annotated, (x1, y2), (x1 + bracket_len, y2), color, 3)
        cv2.line(annotated, (x1, y2), (x1, y2 - bracket_len), color, 3)
        # Bottom-right
        cv2.line(annotated, (x2, y2), (x2 - bracket_len, y2), color, 3)
        cv2.line(annotated, (x2, y2), (x2, y2 - bracket_len), color, 3)

    # 2. Top Header Badge
    score_txt = f" ({score:.2f})" if score is not None else ""
    header_text = f"#{tid}{score_txt} | {w_cm:.0f}x{l_cm:.0f}cm [{sev}]"
    (tw, th), baseline = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    badge_y1 = max(0, y1 - th - 8)
    badge_y2 = y1
    cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
    cv2.putText(annotated, header_text, (x1 + 5, badge_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Bottom Sub-Badge: Dist & Area
    sub_text = f"Dist: {dist_m:.1f}m | {area_m2:.2f}m2"
    (sw, sh), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    sub_y1 = y2
    sub_y2 = min(annotated.shape[0], y2 + sh + 8)
    cv2.rectangle(annotated, (x1, sub_y1), (x1 + sw + 10, sub_y2), (20, 20, 20), -1)
    cv2.rectangle(annotated, (x1, sub_y1), (x1 + sw + 10, sub_y2), color, 1)
    cv2.putText(annotated, sub_text, (x1 + 5, sub_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

class RobustPotholeTracker:
    """
    Robust spatio-temporal tracker for road potholes:
    - Eliminates intra-frame duplicates via IoU / IoS suppression.
    - Matches detections across frames via dual IoU overlap and centroid proximity.
    - Calibrates and temporal-smooths real-world metric dimensions (Length, Width, Area).
    - Requires temporal stability (min_hits >= 3) before confirming a unique count.
    """
    def __init__(self, max_dist=130, max_missing=22, min_hits=3):
        self.next_id = 1
        self.max_dist = max_dist
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.tracks = {}  # tid -> {'box': b, 'center': c, 'hits': n, 'missing': n, 'counted': bool, 'dim': dict}
        self.confirmed_unique_count = 0
        self.lock = threading.Lock()

    def update(self, raw_boxes, raw_confs=None):
        with self.lock:
            # 1. Deduplicate boxes in the same frame
            boxes, confs = deduplicate_boxes(raw_boxes, raw_confs)
            
            matched_tracks = set()
            box_to_id = [None] * len(boxes)

            # 2. Match current detections against active tracks using IoU & distance
            for b_idx, b in enumerate(boxes):
                bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
                best_id = None
                best_score = float('inf')

                for tid, tr in self.tracks.items():
                    if tid in matched_tracks:
                        continue
                    tc = tr['center']
                    dist = math.hypot(bc[0] - tc[0], bc[1] - tc[1])

                    # Calculate bounding box IoU
                    xA = max(b[0], tr['box'][0])
                    yA = max(b[1], tr['box'][1])
                    xB = min(b[2], tr['box'][2])
                    yB = min(b[3], tr['box'][3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    b_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                    t_area = max(1.0, (tr['box'][2] - tr['box'][0]) * (tr['box'][3] - tr['box'][1]))
                    iou = inter / float(b_area + t_area - inter + 1e-6)

                    # Match condition: IoU overlap OR spatial centroid proximity
                    if iou > 0.12 or dist < self.max_dist:
                        score = dist - (iou * 120.0)
                        if score < best_score:
                            best_score = score
                            best_id = tid

                if best_id is not None:
                    tr = self.tracks[best_id]
                    tr['box'] = b
                    tr['center'] = bc
                    tr['hits'] += 1
                    tr['missing'] = 0
                    if tr['hits'] >= self.min_hits and not tr['counted']:
                        tr['counted'] = True
                        self.confirmed_unique_count += 1
                    matched_tracks.add(best_id)
                    box_to_id[b_idx] = best_id

            # 3. Register new potential tracks for unmatched detections
            for b_idx, b in enumerate(boxes):
                if box_to_id[b_idx] is None:
                    nid = self.next_id
                    self.next_id += 1
                    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
                    self.tracks[nid] = {
                        'box': b,
                        'center': bc,
                        'hits': 1,
                        'missing': 0,
                        'counted': False
                    }
                    box_to_id[b_idx] = nid

            # 4. Compute & smooth real-world metric dimensions
            active_dims = []
            for b_idx, b in enumerate(boxes):
                tid = box_to_id[b_idx]
                raw_dim = compute_ground_plane_dimensions(b)
                tr = self.tracks.get(tid)
                if tr and 'dim' in tr:
                    # Exponential Moving Average (EMA, alpha=0.35) for temporal stabilization
                    prev_w = tr['dim']['width_cm']
                    prev_l = tr['dim']['length_cm']
                    smooth_w = round(0.35 * raw_dim['width_cm'] + 0.65 * prev_w, 1)
                    smooth_l = round(0.35 * raw_dim['length_cm'] + 0.65 * prev_l, 1)
                    raw_dim['width_cm'] = smooth_w
                    raw_dim['length_cm'] = smooth_l
                    raw_dim['area_m2'] = round((math.pi / 4.0) * (smooth_w / 100.0) * (smooth_l / 100.0), 2)
                    
                    # Update ASTM Severity & color based on smoothed max dimension
                    max_d = max(smooth_w, smooth_l)
                    if max_d < 30.0:
                        raw_dim['severity'] = "MINOR"
                        raw_dim['bgr_color'] = (60, 200, 60)
                        raw_dim['hex_color'] = "#2ea043"
                    elif max_d <= 60.0:
                        raw_dim['severity'] = "MODERATE"
                        raw_dim['bgr_color'] = (0, 165, 255)
                        raw_dim['hex_color'] = "#d29922"
                    else:
                        raw_dim['severity'] = "SEVERE"
                        raw_dim['bgr_color'] = (50, 50, 240)
                        raw_dim['hex_color'] = "#f85149"

                if tr:
                    tr['dim'] = raw_dim
                active_dims.append(raw_dim)

            # 5. Age out old tracks that left the field of view
            to_del = []
            for tid, tr in self.tracks.items():
                if tid not in matched_tracks:
                    tr['missing'] += 1
                    if tr['missing'] > self.max_missing:
                        to_del.append(tid)
            for tid in to_del:
                del self.tracks[tid]

            return boxes, confs, box_to_id, self.confirmed_unique_count, active_dims

    def reset(self):
        with self.lock:
            self.next_id = 1
            self.tracks.clear()
            self.confirmed_unique_count = 0

# Global Telemetry & Configuration State
state = {
    "mode": "yolo",            # 'yolo', 'classical', 'edges'
    "source": "video",         # 'video', 'webcam'
    "detection_count": 0,
    "total_detections": 0,
    "fps": 0.0,
    "latency_ms": 0.0,
    "confidence_threshold": 0.40,
    "status": "MONITORING",
    "active_potholes": [],
    "max_dimension_cm": 0.0
}
state_lock = threading.Lock()

class DetectionEngine:
    """Unified Detection Engine combining Classical Mid-Level CV and Deep Learning YOLO."""
    
    def __init__(self):
        self.yolo_model = None
        self.tracker = RobustPotholeTracker(max_dist=130, max_missing=22, min_hits=3)
        
        # Initialize YOLO if model weights exist
        if os.path.exists(MODEL_PATH) and YOLO is not None:
            try:
                print(f"[ENGINE] Loading YOLO weights from: {MODEL_PATH}")
                self.yolo_model = YOLO(MODEL_PATH)
                print("[ENGINE] YOLO engine initialized successfully.")
            except Exception as e:
                print(f"[ENGINE WARN] YOLO load failed ({e}). Defaulting to Classical CV.")

        # Classical CV Structuring Element
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process_classical(self, frame):
        """Unit 1 Classical CV: Grayscale -> GaussianBlur -> Canny -> Morphology -> Contours"""
        t0 = time.perf_counter()
        
        # 1. Grayscale Dimensionality Reduction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Smoothing (Linear Filter to suppress high-frequency asphalt noise)
        blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2, sigmaY=1.2)
        
        # 3. Gradient-based Edge Mapping (Sobel Gradients + Canny Hysteresis)
        edges = cv2.Canny(blurred, 50, 150)
        
        # 4. Morphological Closing (Connects broken structural boundary edges)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # 5. Contour Grouping & Feature Extraction
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated = frame.copy()
        candidate_boxes = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area bounds to reject tiny gravel noise and massive horizon polygons
            if 800 < area < 30000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / max(1, h)
                # Exclude long linear features (lane markings have aspect ratio > 4 or < 0.25)
                if 0.4 < aspect_ratio < 2.8:
                    candidate_boxes.append([x, y, x + w, y + h])

        # Deduplicate and track unique physical potholes with metric dimensions
        dedup_boxes, _, track_ids, unique_total, active_dims = self.tracker.update(candidate_boxes)

        active_data = []
        for i, box in enumerate(dedup_boxes):
            tid = track_ids[i]
            dim = active_dims[i]
            draw_pothole_annotation(annotated, box, tid, None, dim)
            active_data.append({
                "id": tid,
                "width_cm": dim["width_cm"],
                "length_cm": dim["length_cm"],
                "area_m2": dim["area_m2"],
                "distance_m": dim["distance_m"],
                "severity": dim["severity"],
                "hex_color": dim["hex_color"]
            })

        latency = (time.perf_counter() - t0) * 1000.0
        return annotated, closed, len(dedup_boxes), unique_total, latency, active_data

    def process_yolo(self, frame):
        """Deep Learning YOLO Pipeline with Intra-Frame Deduplication & Spatio-Temporal Tracking"""
        if self.yolo_model is None:
            # Fallback to classical if YOLO not available
            annotated, closed, count, unique_total, latency, active_data = self.process_classical(frame)
            return annotated, count, unique_total, latency, active_data
            
        t0 = time.perf_counter()
        results = self.yolo_model(frame, conf=state["confidence_threshold"], verbose=False)[0]
        annotated = frame.copy()
        
        raw_boxes = []
        raw_confs = []
        if results.boxes is not None and len(results.boxes) > 0:
            raw_boxes = results.boxes.xyxy.cpu().numpy().tolist()
            raw_confs = results.boxes.conf.cpu().numpy().tolist()

        # Deduplicate overlapping boxes on same pothole & track across frames
        boxes, confs, track_ids, unique_total, active_dims = self.tracker.update(raw_boxes, raw_confs)

        active_data = []
        for i, box in enumerate(boxes):
            tid = track_ids[i]
            score = confs[i] if i < len(confs) else 0.0
            dim = active_dims[i]
            draw_pothole_annotation(annotated, box, tid, score, dim)
            active_data.append({
                "id": tid,
                "width_cm": dim["width_cm"],
                "length_cm": dim["length_cm"],
                "area_m2": dim["area_m2"],
                "distance_m": dim["distance_m"],
                "severity": dim["severity"],
                "hex_color": dim["hex_color"]
            })
            
        latency = (time.perf_counter() - t0) * 1000.0
        return annotated, len(boxes), unique_total, latency, active_data

engine = DetectionEngine()

def generate_video_stream():
    """Generates continuous MJPEG frames for the web dashboard."""
    global state
    
    cap = None
    curr_source = None
    frame_times = []
    
    while True:
        with state_lock:
            mode = state["mode"]
            source_type = state["source"]
            
        target_source = 0 if source_type == "webcam" else DEMO_VIDEO
        
        # Reset capture if source changed or uninitialized
        if cap is None or curr_source != target_source:
            if cap is not None:
                cap.release()
            curr_source = target_source
            cap = cv2.VideoCapture(curr_source)
            engine.tracker.reset()
            print(f"[STREAM] Connected to video source: {curr_source}")

        if not cap.isOpened():
            time.sleep(0.5)
            continue
            
        success, frame = cap.read()
        if not success:
            # Loop video file seamlessly
            if curr_source != 0 and os.path.exists(str(curr_source)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.05)
            continue
            
        # Processing based on selected mode
        active_potholes = []
        if mode == "classical":
            output_frame, _, count, unique_total, latency, active_potholes = engine.process_classical(frame)
        elif mode == "edges":
            _, edges, count, unique_total, latency, active_potholes = engine.process_classical(frame)
            output_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(output_frame, "CANNY GRADIENT EDGE MAP (Sobel Gx, Gy + Hysteresis)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else: # 'yolo'
            output_frame, count, unique_total, latency, active_potholes = engine.process_yolo(frame)
            
        # Compute real-time FPS
        now = time.time()
        frame_times.append(now)
        frame_times = [t for t in frame_times if now - t <= 1.0]
        fps = len(frame_times)
        
        # Update shared telemetry
        with state_lock:
            state["detection_count"] = count
            state["total_detections"] = unique_total
            state["fps"] = round(fps, 1)
            state["latency_ms"] = round(latency, 1)
            state["active_potholes"] = active_potholes
            if active_potholes:
                max_dim = max(max(p["width_cm"], p["length_cm"]) for p in active_potholes)
                state["max_dimension_cm"] = round(max_dim, 1)
                if any(p["severity"] == "SEVERE" for p in active_potholes):
                    state["status"] = "CRITICAL HAZARD DETECTED!"
                else:
                    state["status"] = "ROAD HAZARD DETECTED!"
            else:
                state["max_dimension_cm"] = 0.0
                state["status"] = "SURFACE CLEAR"

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    with state_lock:
        return jsonify({
            "detections": state["detection_count"],
            "total_detections": state["total_detections"],
            "fps": state["fps"],
            "latency_ms": state["latency_ms"],
            "mode": state["mode"],
            "source": state["source"],
            "status": state["status"],
            "confidence_threshold": state["confidence_threshold"],
            "active_potholes": state.get("active_potholes", []),
            "max_dimension_cm": state.get("max_dimension_cm", 0.0)
        })

@app.route('/api/set_config', methods=['POST'])
def set_config():
    data = request.get_json(force=True, silent=True) or {}
    with state_lock:
        if "mode" in data and data["mode"] in ["classical", "yolo", "edges"]:
            state["mode"] = data["mode"]
        if "source" in data and data["source"] in ["video", "webcam"]:
            if state["source"] != data["source"]:
                state["source"] = data["source"]
                engine.tracker.reset()
        if "confidence_threshold" in data:
            state["confidence_threshold"] = float(data["confidence_threshold"])
    return jsonify({"success": True, "state": state})

@app.route('/api/reset_stats', methods=['POST'])
def reset_stats():
    engine.tracker.reset()
    with state_lock:
        state["total_detections"] = 0
        state["detection_count"] = 0
        state["active_potholes"] = []
        state["max_dimension_cm"] = 0.0
    return jsonify({"success": True, "total_detections": 0})

if __name__ == "__main__":
    # Use port 5050 by default to prevent collision with macOS ControlCenter AirPlay on port 5000
    port = int(os.environ.get("PORT", 5050))
    print("\n" + "=" * 70)
    print(f"  ⚡ PYRESEARCH POTHOLE DETECTION DASHBOARD RUNNING")
    print(f"  Access the dashboard at: http://localhost:{port}")
    print(f"  Available Modes: Deep Learning YOLO | Classical CV | Canny Edge Map")
    print("=" * 70 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)