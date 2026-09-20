
import cv2
import numpy as np
import time
import psutil
import threading
from collections import deque
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, jsonify, request
from gpiozero import Buzzer
import logging


# 1. SETUP & CONFIG
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)

# CONFIGURABLE PARAMETERS
EAR_THRESHOLD = 0.20  
MAR_THRESHOLD = 0.5
ALARM_FRAMES = 15      
FATIGUE_WINDOW = 180   
YAWN_LIMIT = 3         
BLINK_LIMIT_BPM = 25   
PERCLOS_LIMIT = 15.0   

# GLOBALS
latest_jpeg = None
stream_active = False 
event_logs = deque(maxlen=50) 
last_status = "INITIALIZING"
picam = None 
ZOOM_FACTOR = 1.0 

telemetry = {
    "ear": 0.0, "mar": 0.0, "cpu": 0.0, "ram": 0.0, "temp": 0.0, "fps": 0,
    "yawn_count": 0, "blink_count": 0, "bpm": 0.0, "perclos": 0.0, 
    "status": "INITIALIZING", "uptime": 0, "streaming": False,
    "lens_pos": 10.0, "zoom": 1.0
}

yawn_timestamps = deque()
blink_timestamps = deque()
per_window = deque(maxlen=1200)

# 2. HAAR CASCADE CLASSIFIERS (improved)
import os
cascade_dir = os.path.join(os.path.dirname(__file__), 'cascades')
face_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_eye.xml'))

def calculate_ear_from_face(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    h, w = gray.shape
    eye_region = gray[:h//3, :]
    threshold = cv2.threshold(eye_region, 100, 255, cv2.THRESH_BINARY)[1]
    dark_pixels = np.sum(threshold == 0)
    total_pixels = eye_region.size
    return 1.0 - (dark_pixels / (total_pixels + 1))

def calculate_mar_from_face(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    h, w = gray.shape
    mouth_region = gray[int(h*0.6):, :]
    threshold = cv2.threshold(mouth_region, 100, 255, cv2.THRESH_BINARY_INV)[1]
    dark_pixels = np.sum(threshold > 100)
    total_pixels = mouth_region.size
    return dark_pixels / (total_pixels + 1)

def get_pi_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except: return 0.0

# 3. THE WEB DASHBOARD (Updated Focus Logic)
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Driver Dashboard v5.1 (Turbo)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #0b0e14; color: #e1e1e1; font-family: 'Segoe UI', sans-serif; text-align: center; margin: 0; }
        .header { padding: 15px; background: #161b22; border-bottom: 2px solid #30363d; display: flex; justify-content: space-around; align-items: center; }
        .live-stats { display: flex; justify-content: space-around; background: #0d1117; padding: 15px; border-bottom: 1px solid #30363d; }
        .stat-item { flex: 1; border-right: 1px solid #30363d; }
        .stat-val { display: block; font-size: 20px; color: #58a6ff; font-weight: bold; font-family: monospace; }
        .stat-label { font-size: 10px; color: #8b949e; text-transform: uppercase; }
        .status-bar { font-size: 22px; font-weight: bold; padding: 12px; background: #161b22; }
        .status-ok { color: #3fb950; } .status-warn { color: #d29922; background: #332200; } .status-crit { color: #f85149; background: #330000; }
        .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 20px; }
        .video-box { border: 2px solid #30363d; border-radius: 12px; width: 640px; height: 360px; overflow: hidden; background: #000; }
        .chart-box { background: #161b22; padding: 15px; border-radius: 12px; width: 500px; border: 1px solid #30363d; }
        .control-panel { background: #21262d; padding: 15px; border-radius: 12px; border: 1px solid #30363d; display: flex; gap: 20px; align-items: center; }
        .slider-group { text-align: left; font-size: 12px; }
        .btn-toggle { padding: 12px 24px; border-radius: 6px; border: none; font-weight: bold; cursor: pointer; background: #238636; color: white; }
        .btn-toggle.active { background: #da3633; }
        input[type=range] { width: 100px; vertical-align: middle; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI DRIVER MONITOR v5.1</h1>
        <div class="control-panel">
            <div class="slider-group">EAR: <span id="earVal">0.20</span><br><input type="range" id="earSlider" min="0.10" max="0.35" step="0.01" value="0.20" oninput="updateEar(this.value)"></div>
            <div class="slider-group">FOCUS: <span id="focusVal">10.0</span><br><input type="range" id="focusSlider" min="0.0" max="12.0" step="0.1" value="10.0" oninput="updateFocus(this.value)"></div>
            <div class="slider-group">ZOOM: <span id="zoomVal">1.0</span>x<br><input type="range" id="zoomSlider" min="1.0" max="3.0" step="0.1" value="1.0" oninput="updateZoom(this.value)"></div>
            <button id="streamBtn" class="btn-toggle" onclick="toggleStream()">ENABLE VIDEO</button>
        </div>
    </div>
    <div class="live-stats">
        <div class="stat-item"><span class="stat-val" id="curEAR">0.00</span><span class="stat-label">EAR</span></div>
        <div class="stat-item"><span class="stat-val" id="curMAR">0.00</span><span class="stat-label">MAR</span></div>
        <div class="stat-item"><span class="stat-val" id="curPER">0.0%</span><span class="stat-label">PERCLOS</span></div>
        <div class="stat-item"><span class="stat-val" id="curBPM">0.0</span><span class="stat-label">BPM</span></div>
        <div class="stat-item"><span class="stat-val" id="curFPS">0</span><span class="stat-label">FPS</span></div>
        <div class="stat-item"><span class="stat-val" id="curCPU">0%</span><span class="stat-label">CPU</span></div>
        <div class="stat-item"><span class="stat-val" id="curTMP">0C</span><span class="stat-label">Temp</span></div>
    </div>
    <div id="statusDiv" class="status-bar status-ok">SYSTEM STATUS: OK</div>
    <div class="container">
        <div class="video-box"><img id="streamImg" src="/video_feed" style="width: 100%;"></div>
        <div class="chart-box"><h3>Fatigue Trends</h3><canvas id="fatigueChart"></canvas></div>
    </div>
    <div class="container">
        <div class="chart-box"><h3>Biometrics (EAR/MAR)</h3><canvas id="driverChart"></canvas></div>
        <div class="chart-box"><h3>Behavior History</h3><canvas id="behaviorChart"></canvas></div>
        <div class="chart-box"><h3>System Performance</h3><canvas id="sysChart"></canvas></div>
    </div>
    <script>
        const ds = { cpu:[], temp:[], ram:[], fps:[], ear:[], mar:[], perclos:[], bpm:[], blink_count:[], yawn_count:[] };
        const labels = Array(120).fill('');
        Object.keys(ds).forEach(k => ds[k] = Array(120).fill(null));

        function updateEar(val) { document.getElementById('earVal').innerText = val; fetch(`/api/set_ear?val=${val}`); }
        function updateFocus(val) { document.getElementById('focusVal').innerText = val; fetch(`/api/set_focus?val=${val}`); }
        function updateZoom(val) { document.getElementById('zoomVal').innerText = val; fetch(`/api/set_zoom?val=${val}`); }
        function toggleStream() { fetch('/api/toggle_stream'); }

        const fChart = new Chart(document.getElementById('fatigueChart').getContext('2d'), { type:'line', data:{labels:labels, datasets:[{label:'PERCLOS %', borderColor:'#d29922', data:ds.perclos, yAxisID:'y', pointRadius:0},{label:'BPM', borderColor:'#bc8cff', data:ds.bpm, yAxisID:'y1', pointRadius:0}]}, options:{animation:false, scales:{y:{position:'left', suggestedMax:30},y1:{position:'right', suggestedMax:60, grid:{drawOnChartArea:false}}}} });
        const dChart = new Chart(document.getElementById('driverChart').getContext('2d'), { type:'line', data:{labels:labels, datasets:[{label:'EAR', borderColor:'#58a6ff', data:ds.ear, pointRadius:0},{label:'MAR', borderColor:'#3fb950', data:ds.mar, pointRadius:0}]}, options:{animation:false, scales:{y:{suggestedMax:0.6}}}} );
        const bChart = new Chart(document.getElementById('behaviorChart').getContext('2d'), { type:'line', data:{labels:labels, datasets:[{label:'Blinks', borderColor:'#58a6ff', data:ds.blink_count, fill:true, pointRadius:0, yAxisID:'y'},{label:'Yawns', borderColor:'#f85149', data:ds.yawn_count, pointRadius:0, yAxisID:'y1'}]}, options:{animation:false, scales:{y:{position:'left'}, y1:{position:'right', suggestedMax:10, grid:{drawOnChartArea:false}}}} });
        const sChart = new Chart(document.getElementById('sysChart').getContext('2d'), { type:'line', data:{labels:labels, datasets:[{label:'CPU %', borderColor:'#f85149', data:ds.cpu, pointRadius:0},{label:'FPS', borderColor:'#58a6ff', data:ds.fps, pointRadius:0},{label:'Temp C', borderColor:'#d29922', data:ds.temp, pointRadius:0}]}, options:{animation:false, scales:{y:{suggestedMax:100}}}} );

        setInterval(() => {
            fetch('/api/telemetry').then(r => r.json()).then(d => {
                document.getElementById('statusDiv').innerText = "SYSTEM STATUS: " + d.status;
                document.getElementById('statusDiv').className = "status-bar " + (d.status.includes('CRITICAL') ? 'status-crit' : (d.status.includes('WARNING') ? 'status-warn' : 'status-ok'));
                document.getElementById('curEAR').innerText = d.ear.toFixed(2); document.getElementById('curMAR').innerText = d.mar.toFixed(2);
                document.getElementById('curPER').innerText = d.perclos + "%"; document.getElementById('curBPM').innerText = d.bpm;
                document.getElementById('curFPS').innerText = d.fps; document.getElementById('curCPU').innerText = d.cpu + "%"; document.getElementById('curTMP').innerText = d.temp + "C";
                Object.keys(ds).forEach(k => { ds[k].shift(); ds[k].push(d[k]); });
                fChart.update(); dChart.update(); sChart.update(); bChart.update();
                const btn = document.getElementById('streamBtn'); btn.innerText = d.streaming ? "DISABLE VIDEO" : "ENABLE VIDEO"; btn.className = d.streaming ? "btn-toggle active" : "btn-toggle";
            });
        }, 1000);
    </script>
</body>
</html>
"""

# 4. FLASK ROUTES
@app.route('/')
def index(): return render_template_string(HTML_PAGE)

@app.route('/api/toggle_stream')
def toggle_stream():
    global stream_active
    stream_active = not stream_active
    return jsonify(streaming=stream_active)

@app.route('/api/set_ear')
def set_ear():
    global EAR_THRESHOLD
    try:
        EAR_THRESHOLD = float(request.args.get('val'))
        return jsonify(success=True)
    except: return jsonify(success=False)

@app.route('/api/set_focus')
def set_focus():
    global picam
    try:
        val = float(request.args.get('val'))
        if picam: picam.set_controls({"LensPosition": val})
        return jsonify(success=True)
    except: return jsonify(success=False)

@app.route('/api/set_zoom')
def set_zoom():
    global ZOOM_FACTOR
    try:
        ZOOM_FACTOR = float(request.args.get('val'))
        return jsonify(success=True)
    except: return jsonify(success=False)

@app.route('/api/telemetry')
def get_telemetry(): 
    res = telemetry.copy(); res["logs"] = list(event_logs); res["streaming"] = stream_active
    return jsonify(res)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if stream_active and latest_jpeg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + latest_jpeg + b'\r\n')
            else:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "STREAMING DISABLED", (160, 180), 1, 2, (100, 100, 100), 2)
                _, b = cv2.imencode('.jpg', blank); yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b.tobytes() + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 5. MAIN AI ENGINE
def main():
    global latest_jpeg, stream_active, telemetry, last_status, event_logs, EAR_THRESHOLD, picam, ZOOM_FACTOR
    
    buzzer = Buzzer(17); picam = Picamera2()
    
    # FORCING 60FPS HARDWARE CONFIG
    config = picam.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 360)},
        controls={"FrameRate": 60.0}
    )
    picam.configure(config); picam.start()
    
    # FORCING HIGH-SPEED EXPOSURE (33ms limit = 30fps min)
    # 16666us = 60fps, 33333us = 30fps
    picam.set_controls({
        "AfMode": 0, 
        "LensPosition": 10.0, 
        "FrameDurationLimits": (16666, 33333),
        "AeConstraintMode": 1 # 1 = Focus on Maintain Frame Rate
    })

    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()

    closed_eyes_counter = 0; blink_active = False; yawn_active = False; per_score = 0.0; current_bpm = 0.0
    start_time = time.time(); prev_time = time.time()

    while True:
        frame = picam.capture_array()
        if frame is None: continue
        now = time.time(); uptime = int(now - start_time); fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0; prev_time = now
        
        # DIGITAL ZOOM LOGIC
        h, w, _ = frame.shape
        if ZOOM_FACTOR > 1.0:
            nw, nh = int(w / ZOOM_FACTOR), int(h / ZOOM_FACTOR)
            x1, y1 = (w - nw) // 2, (h - nh) // 2
            frame = frame[y1:y1+nh, x1:x1+nw]
            frame = cv2.resize(frame, (w, h)) 
        
        frame = cv2.flip(frame, 1); h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(80, 80))
        cur_ear = 0.0; cur_mar = 0.0; status = "OK"

        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            face_roi = frame[y:y+fh, x:x+fw]
            cur_ear = calculate_ear_from_face(face_roi)
            cur_mar = calculate_mar_from_face(face_roi)

            per_window.append(1 if cur_ear < EAR_THRESHOLD else 0)
            per_score = round((sum(per_window)/len(per_window))*100, 1) if len(per_window)>0 else 0.0

            if cur_ear < EAR_THRESHOLD:
                closed_eyes_counter += 1
                if closed_eyes_counter >= 3 and not blink_active: blink_timestamps.append(now); blink_active = True
            else: blink_active = False; closed_eyes_counter = 0

            if cur_mar > MAR_THRESHOLD:
                if not yawn_active: yawn_timestamps.append(now); yawn_active = True
            else: yawn_active = False

            while blink_timestamps and now - blink_timestamps[0] > FATIGUE_WINDOW: blink_timestamps.popleft()
            while yawn_timestamps and now - yawn_timestamps[0] > FATIGUE_WINDOW: yawn_timestamps.popleft()
            current_bpm = round(len(blink_timestamps) / (FATIGUE_WINDOW / 60), 1)

            if closed_eyes_counter >= ALARM_FRAMES:
                status = "CRITICAL: SLEEPING"; buzzer.on()
            elif per_score >= PERCLOS_LIMIT or current_bpm >= BLINK_LIMIT_BPM or len(yawn_timestamps) >= YAWN_LIMIT:
                status = "WARNING: FATIGUE"; buzzer.on() if int(now * 3) % 2 == 0 else buzzer.off()
            else: buzzer.off()

            if stream_active:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                cv2.putText(bgr, f"EAR: {cur_ear:.2f} MAR: {cur_mar:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                _, buf = cv2.imencode('.jpg', bgr); latest_jpeg = buf.tobytes()
        else:
            status = "SEARCHING..."
            buzzer.off()
            if stream_active:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, "SEARCHING...", (150, 180), 1, 1.5, (0, 0, 255), 2)
                _, buf = cv2.imencode('.jpg', bgr); latest_jpeg = buf.tobytes()

        telemetry.update({"ear":cur_ear, "mar":cur_mar, "perclos":per_score, "bpm": current_bpm, "blink_count":len(blink_timestamps), "yawn_count":len(yawn_timestamps), "status":status, "cpu":psutil.cpu_percent(), "temp":get_pi_temp(), "fps":int(fps)})

if __name__ == '__main__':
    main()