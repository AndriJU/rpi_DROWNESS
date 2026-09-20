
import cv2
import numpy as np
import time
import psutil
import threading
from collections import deque
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, jsonify, request
from gpiozero import DigitalOutputDevice
import subprocess
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

# BUZZER (Matek DBuz5V on GPIO17, driven via external transistor)
BUZZER_PIN = 17
BEEP_PULSE_MS = 8.0    # tunable live; a common-emitter stage may need more
CRIT_INTERVAL = 0.12   # rapid clicking while eyes are actually closed
# Fatigue is advisory, not an emergency: a short burst on a long cooldown.
# Continuous alerting at the wheel startles and gets tuned out.
WARN_BURST = 2         # clicks per reminder
WARN_REPEAT_S = 420.0  # ~7 min between reminders
last_warn_time = 0.0
BUZZER_ACTIVE_HIGH = False  # transistor stage inverts: GPIO low = sound
alert_level = "OK"
test_beep_request = False
hold_request = False

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
    "lens_pos": 10.0, "zoom": 1.0,
    "quality": 0.0, "det_rate": 0.0, "size_score": 0.0, "exposure_score": 0.0,
    "eye_score": 0.0, "mouth_score": 0.0, "stability_score": 0.0,
    "brightness": 0.0, "face_size": 0.0, "quality_hint": "NO FACE"
}

yawn_timestamps = deque()
blink_timestamps = deque()
per_window = deque(maxlen=1200)
detect_window = deque(maxlen=100)

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

prev_face_center = None

def score_band(value, good_lo, good_hi, zero_lo, zero_hi):
    if good_lo <= value <= good_hi: return 100.0
    if value < good_lo:
        return max(0.0, 100.0 * (value - zero_lo) / (good_lo - zero_lo))
    return max(0.0, 100.0 * (zero_hi - value) / (zero_hi - good_hi))

def assess_quality(face_gray, face_box, frame_w, frame_h):
    global prev_face_center
    x, y, fw, fh = face_box
    gh = face_gray.shape[0]

    size_pct = 100.0 * (fw * fh) / (frame_w * frame_h)
    brightness = float(np.mean(face_gray))

    # Contrast in each band: a flat region means the EAR/MAR reading is noise.
    eye_score = min(100.0, float(np.std(face_gray[:gh // 3, :])) * 2.5)
    mouth_score = min(100.0, float(np.std(face_gray[int(gh * 0.6):, :])) * 2.5)

    center = (x + fw / 2.0, y + fh / 2.0)
    if prev_face_center is None:
        stability_score = 100.0
    else:
        drift = np.hypot(center[0] - prev_face_center[0], center[1] - prev_face_center[1])
        stability_score = max(0.0, 100.0 - (drift / max(fw, 1)) * 300.0)
    prev_face_center = center

    return {
        "size_score": score_band(size_pct, 8.0, 40.0, 1.5, 75.0),
        "exposure_score": score_band(brightness, 90.0, 170.0, 25.0, 240.0),
        "eye_score": eye_score, "mouth_score": mouth_score,
        "stability_score": stability_score,
        "brightness": brightness, "face_size": size_pct,
    }

def quality_hint(det_rate, q):
    if det_rate < 40: return "FACE RARELY FOUND - CHECK FRAMING"
    if q["face_size"] < 8: return "FACE TOO SMALL - INCREASE ZOOM"
    if q["face_size"] > 40: return "FACE TOO LARGE - REDUCE ZOOM"
    if q["brightness"] < 90: return "TOO DARK - ADD LIGHT"
    if q["brightness"] > 170: return "OVEREXPOSED - REDUCE LIGHT"
    if q["eye_score"] < 40: return "EYE REGION BLURRY - ADJUST FOCUS"
    if q["stability_score"] < 50: return "UNSTABLE - POSSIBLE FALSE DETECTION"
    if det_rate < 80: return "INTERMITTENT - HOLD STILL / FACE CAMERA"
    return "GOOD"

def set_buzzer(dev, on):
    # Polarity in software, so it can be flipped live without reopening the pin.
    dev.value = 1 if (on == BUZZER_ACTIVE_HIGH) else 0

def click(dev):
    set_buzzer(dev, True); time.sleep(BEEP_PULSE_MS / 1000.0); set_buzzer(dev, False)

def beeper_loop(dev):
    # Runs off the frame loop so pulse width is not quantised to the frame period.
    global test_beep_request, hold_request, last_warn_time
    while True:
        if hold_request:
            hold_request = False
            set_buzzer(dev, True); time.sleep(1.0); set_buzzer(dev, False)
            time.sleep(0.2); continue
        if test_beep_request:
            test_beep_request = False
            for _ in range(3):
                click(dev); time.sleep(0.15)
            continue
        level = alert_level
        if level == "CRITICAL":
            click(dev); time.sleep(CRIT_INTERVAL)
        elif level == "WARNING":
            if time.time() - last_warn_time >= WARN_REPEAT_S:
                last_warn_time = time.time()
                for i in range(WARN_BURST):
                    click(dev)
                    if i < WARN_BURST - 1: time.sleep(0.20)
            time.sleep(0.20)
        else:
            set_buzzer(dev, False); time.sleep(0.05)

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
        .hint-bar { font-size: 14px; padding: 8px; background: #0d1117; color: #8b949e; letter-spacing: 1px; border-bottom: 1px solid #30363d; }
        .hint-good { color: #3fb950; } .hint-bad { color: #d29922; }
        .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 20px; }
        .video-box { border: 2px solid #30363d; border-radius: 12px; width: 640px; height: 360px; overflow: hidden; background: #000; }
        .chart-box { background: #161b22; padding: 15px; border-radius: 12px; width: 500px; border: 1px solid #30363d; }
        .control-panel { background: #21262d; padding: 15px; border-radius: 12px; border: 1px solid #30363d; display: flex; flex-wrap: wrap; gap: 15px; align-items: center; justify-content: center; }
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
            <div class="slider-group">PULSE: <span id="pulseVal">8</span> ms<br><input type="range" id="pulseSlider" min="1" max="300" step="1" value="8" oninput="updatePulse(this.value)"></div>
            <button class="btn-toggle" style="background:#1f6feb" onclick="fetch('/api/test_beep')">TEST BEEP</button>
            <button class="btn-toggle" style="background:#8957e5" onclick="fetch('/api/hold_test')">HOLD 1s</button>
            <button id="polBtn" class="btn-toggle" style="background:#30363d" onclick="fetch('/api/toggle_polarity')">HIGH = ON</button>
            <button class="btn-toggle" style="background:#6e2018" onclick="doShutdown()">SHUT DOWN</button>
        </div>
    </div>
    <div class="live-stats">
        <div class="stat-item"><span class="stat-val" id="curQUAL">0</span><span class="stat-label">Quality</span></div>
        <div class="stat-item"><span class="stat-val" id="curDET">0%</span><span class="stat-label">Detect Rate</span></div>
        <div class="stat-item"><span class="stat-val" id="curEAR">0.00</span><span class="stat-label">EAR</span></div>
        <div class="stat-item"><span class="stat-val" id="curMAR">0.00</span><span class="stat-label">MAR</span></div>
        <div class="stat-item"><span class="stat-val" id="curPER">0.0%</span><span class="stat-label">PERCLOS</span></div>
        <div class="stat-item"><span class="stat-val" id="curBPM">0.0</span><span class="stat-label">BPM</span></div>
        <div class="stat-item"><span class="stat-val" id="curFPS">0</span><span class="stat-label">FPS</span></div>
        <div class="stat-item"><span class="stat-val" id="curCPU">0%</span><span class="stat-label">CPU</span></div>
        <div class="stat-item"><span class="stat-val" id="curTMP">0C</span><span class="stat-label">Temp</span></div>
    </div>
    <div id="shutdownOverlay" style="display:none; position:fixed; inset:0; background:#0b0e14; z-index:999; padding-top:18vh;">
        <h1 style="color:#f85149; letter-spacing:2px;">SHUTTING DOWN</h1>
        <p style="color:#e1e1e1; font-size:18px; max-width:520px; margin:20px auto; line-height:1.6;">
            Wait for the Pi's green activity LED to stop flashing, then wait 5 more seconds.<br><br>
            Only then is it safe to disconnect power.
        </p>
    </div>
    <div id="statusDiv" class="status-bar status-ok">SYSTEM STATUS: OK</div>
    <div id="hintDiv" class="hint-bar">SIGNAL: NO FACE</div>
    <div class="container">
        <div class="video-box"><img id="streamImg" src="/video_feed" style="width: 100%;"></div>
        <div class="chart-box"><h3>Fatigue Trends</h3><canvas id="fatigueChart"></canvas></div>
    </div>
    <div class="container">
        <div class="chart-box"><h3>Detection Quality</h3><canvas id="qualityChart"></canvas></div>
        <div class="chart-box"><h3>Biometrics (EAR/MAR)</h3><canvas id="driverChart"></canvas></div>
        <div class="chart-box"><h3>Behavior History</h3><canvas id="behaviorChart"></canvas></div>
        <div class="chart-box"><h3>System Performance</h3><canvas id="sysChart"></canvas></div>
    </div>
    <script>
        const ds = { cpu:[], temp:[], ram:[], fps:[], ear:[], mar:[], perclos:[], bpm:[], blink_count:[], yawn_count:[], quality:[], det_rate:[], eye_score:[], mouth_score:[] };
        const labels = Array(120).fill('');
        Object.keys(ds).forEach(k => ds[k] = Array(120).fill(null));

        function updateEar(val) { document.getElementById('earVal').innerText = val; fetch(`/api/set_ear?val=${val}`); }
        function updateFocus(val) { document.getElementById('focusVal').innerText = val; fetch(`/api/set_focus?val=${val}`); }
        function updateZoom(val) { document.getElementById('zoomVal').innerText = val; fetch(`/api/set_zoom?val=${val}`); }
        function toggleStream() { fetch('/api/toggle_stream'); }
        function updatePulse(val) { document.getElementById('pulseVal').innerText = val; fetch('/api/set_pulse?val=' + val); }
        function doShutdown() {
            if (!confirm('Shut down the Raspberry Pi? Monitoring stops and you will need physical access to power it back on.')) return;
            fetch('/api/shutdown?confirm=yes');
            clearInterval(poll);
            document.getElementById('shutdownOverlay').style.display = 'block';
        }

        const fChart = new Chart(document.getElementById('fatigueChart').getContext('2d'), { type:'line', data:{labels:labels, datasets:[{label:'PERCLOS %', borderColor:'#d29922', data:ds.perclos, yAxisID:'y', pointRadius:0},{label:'BPM', borderColor:'#bc8cff', data:ds.bpm, yAxisID:'y1', pointRadius:0}]}, options:{animation:false, scales:{y:{position:'left', suggestedMax:30},y1:{position:'right', suggestedMax:60, grid:{drawOnChartArea:false}}}} });
        const qChart = new Chart(document.getElementById('qualityChart').getContext('2d'), { type:'line', data:{labels:labels, datasets:[{label:'Quality', borderColor:'#3fb950', data:ds.quality, pointRadius:0, borderWidth:2},{label:'Detect %', borderColor:'#58a6ff', data:ds.det_rate, pointRadius:0},{label:'Eye Signal', borderColor:'#bc8cff', data:ds.eye_score, pointRadius:0},{label:'Mouth Signal', borderColor:'#d29922', data:ds.mouth_score, pointRadius:0}]}, options:{animation:false, scales:{y:{min:0, max:100}}}} );
        const dChart = new Chart(document.getElementById('driverChart').getContext('2d'),{ type:'line', data:{labels:labels, datasets:[{label:'EAR', borderColor:'#58a6ff', data:ds.ear, pointRadius:0},{label:'MAR', borderColor:'#3fb950', data:ds.mar, pointRadius:0}]}, options:{animation:false, scales:{y:{suggestedMax:0.6}}}} );
        const bChart = new Chart(document.getElementById('behaviorChart').getContext('2d'), { type:'line', data:{labels:labels, datasets:[{label:'Blinks', borderColor:'#58a6ff', data:ds.blink_count, fill:true, pointRadius:0, yAxisID:'y'},{label:'Yawns', borderColor:'#f85149', data:ds.yawn_count, pointRadius:0, yAxisID:'y1'}]}, options:{animation:false, scales:{y:{position:'left'}, y1:{position:'right', suggestedMax:10, grid:{drawOnChartArea:false}}}} });
        const sChart = new Chart(document.getElementById('sysChart').getContext('2d'), { type:'line', data:{labels:labels, datasets:[{label:'CPU %', borderColor:'#f85149', data:ds.cpu, pointRadius:0},{label:'FPS', borderColor:'#58a6ff', data:ds.fps, pointRadius:0},{label:'Temp C', borderColor:'#d29922', data:ds.temp, pointRadius:0}]}, options:{animation:false, scales:{y:{suggestedMax:100}}}} );

        const poll = setInterval(() => {
            fetch('/api/telemetry').then(r => r.json()).then(d => {
                document.getElementById('statusDiv').innerText = "SYSTEM STATUS: " + d.status;
                document.getElementById('statusDiv').className = "status-bar " + (d.status.includes('CRITICAL') ? 'status-crit' : (d.status.includes('WARNING') ? 'status-warn' : 'status-ok'));
                document.getElementById('curEAR').innerText = d.ear.toFixed(2); document.getElementById('curMAR').innerText = d.mar.toFixed(2);
                document.getElementById('curPER').innerText = d.perclos + "%"; document.getElementById('curBPM').innerText = d.bpm;
                document.getElementById('curFPS').innerText = d.fps; document.getElementById('curCPU').innerText = d.cpu + "%"; document.getElementById('curTMP').innerText = d.temp + "C";
                document.getElementById('curQUAL').innerText = d.quality; document.getElementById('curDET').innerText = d.det_rate + "%";
                document.getElementById('curQUAL').style.color = d.quality >= 70 ? '#3fb950' : (d.quality >= 40 ? '#d29922' : '#f85149');
                const hint = document.getElementById('hintDiv');
                hint.innerText = "SIGNAL: " + d.quality_hint;
                hint.className = "hint-bar " + (d.quality_hint === "GOOD" ? "hint-good" : "hint-bad");
                Object.keys(ds).forEach(k => { ds[k].shift(); ds[k].push(d[k]); });
                fChart.update(); dChart.update(); sChart.update(); bChart.update(); qChart.update();
                const btn = document.getElementById('streamBtn'); btn.innerText = d.streaming ? "DISABLE VIDEO" : "ENABLE VIDEO"; btn.className = d.streaming ? "btn-toggle active" : "btn-toggle";
                document.getElementById('polBtn').innerText = d.active_high ? "HIGH = ON" : "LOW = ON";
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

@app.route('/api/shutdown')
def shutdown():
    global alert_level
    if request.args.get('confirm') != 'yes':
        return jsonify(success=False, error="confirmation required"), 400
    alert_level = "OK"
    def halt():
        time.sleep(1.0)  # let the HTTP response reach the browser first
        subprocess.run(["sudo", "shutdown", "-h", "now"])
    threading.Thread(target=halt, daemon=True).start()
    return jsonify(success=True)

@app.route('/api/test_beep')
def test_beep():
    global test_beep_request
    test_beep_request = True
    return jsonify(success=True)

@app.route('/api/hold_test')
def hold_test():
    global hold_request
    hold_request = True
    return jsonify(success=True)

@app.route('/api/set_pulse')
def set_pulse():
    global BEEP_PULSE_MS
    try:
        BEEP_PULSE_MS = max(1.0, min(500.0, float(request.args.get('val'))))
        return jsonify(success=True, pulse=BEEP_PULSE_MS)
    except: return jsonify(success=False)

@app.route('/api/toggle_polarity')
def toggle_polarity():
    global BUZZER_ACTIVE_HIGH
    BUZZER_ACTIVE_HIGH = not BUZZER_ACTIVE_HIGH
    return jsonify(success=True, active_high=BUZZER_ACTIVE_HIGH)

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
    res["pulse_ms"] = BEEP_PULSE_MS; res["active_high"] = BUZZER_ACTIVE_HIGH
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
    global latest_jpeg, stream_active, telemetry, last_status, event_logs, EAR_THRESHOLD, picam, ZOOM_FACTOR, prev_face_center, alert_level
    
    buzzer = DigitalOutputDevice(BUZZER_PIN, initial_value=not BUZZER_ACTIVE_HIGH)
    picam = Picamera2()
    threading.Thread(target=beeper_loop, args=(buzzer,), daemon=True).start()
    
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
    q = {"size_score": 0.0, "exposure_score": 0.0, "eye_score": 0.0, "mouth_score": 0.0,
         "stability_score": 0.0, "brightness": 0.0, "face_size": 0.0}
    overall_quality = 0.0; hint = "NO FACE"; det_rate = 0.0

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

        detect_window.append(1 if len(faces) > 0 else 0)
        det_rate = round(100.0 * sum(detect_window) / len(detect_window), 1)

        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            face_gray = gray[y:y+fh, x:x+fw]
            cur_ear = calculate_ear_from_face(face_gray)
            cur_mar = calculate_mar_from_face(face_gray)

            q = assess_quality(face_gray, faces[0], w, h)
            overall_quality = round(
                det_rate * 0.40 + q["size_score"] * 0.15 + q["exposure_score"] * 0.15
                + q["eye_score"] * 0.15 + q["mouth_score"] * 0.075 + q["stability_score"] * 0.075, 1)
            hint = quality_hint(det_rate, q)

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
                status = "CRITICAL: SLEEPING"; alert_level = "CRITICAL"
            elif per_score >= PERCLOS_LIMIT or current_bpm >= BLINK_LIMIT_BPM or len(yawn_timestamps) >= YAWN_LIMIT:
                status = "WARNING: FATIGUE"; alert_level = "WARNING"
            else: alert_level = "OK"

            if stream_active:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                cv2.putText(bgr, f"EAR: {cur_ear:.2f} MAR: {cur_mar:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                _, buf = cv2.imencode('.jpg', bgr); latest_jpeg = buf.tobytes()
        else:
            status = "SEARCHING..."
            alert_level = "OK"
            prev_face_center = None
            overall_quality = round(
                det_rate * 0.40 + q["size_score"] * 0.15 + q["exposure_score"] * 0.15
                + q["eye_score"] * 0.15 + q["mouth_score"] * 0.075 + q["stability_score"] * 0.075, 1)
            hint = "NO FACE" if det_rate < 5 else quality_hint(det_rate, q)
            if stream_active:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, "SEARCHING...", (150, 180), 1, 1.5, (0, 0, 255), 2)
                _, buf = cv2.imencode('.jpg', bgr); latest_jpeg = buf.tobytes()

        telemetry.update({"ear":cur_ear, "mar":cur_mar, "perclos":per_score, "bpm": current_bpm, "blink_count":len(blink_timestamps), "yawn_count":len(yawn_timestamps), "status":status, "cpu":psutil.cpu_percent(), "temp":get_pi_temp(), "fps":int(fps),
                          "quality":overall_quality, "det_rate":det_rate, "quality_hint":hint,
                          "size_score":round(q["size_score"],1), "exposure_score":round(q["exposure_score"],1),
                          "eye_score":round(q["eye_score"],1), "mouth_score":round(q["mouth_score"],1),
                          "stability_score":round(q["stability_score"],1),
                          "brightness":round(q["brightness"],1), "face_size":round(q["face_size"],1)})

if __name__ == '__main__':
    main()