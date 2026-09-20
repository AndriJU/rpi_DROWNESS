#!/bin/bash
# Provisions a fresh Raspberry Pi OS Lite (Bookworm, 64-bit) to run the monitor.
# Every dependency comes from apt: the versions there are built against each
# other, which avoids the numpy/picamera2 ABI break that pip installs cause.
set -e

echo "==> Installing system packages"
sudo apt update
sudo apt install -y git python3-opencv python3-picamera2 python3-flask \
                    python3-gpiozero python3-psutil

echo "==> Verifying imports"
python3 - <<'EOF'
import cv2, numpy, psutil, flask, gpiozero
from picamera2 import Picamera2
print(f"    opencv {cv2.__version__} / numpy {numpy.__version__} - OK")
EOF

echo "==> Checking cascade files"
cd "$(dirname "$0")"
for f in haarcascade_frontalface_default.xml haarcascade_eye.xml; do
    if [ ! -s "cascades/$f" ]; then
        echo "    MISSING cascades/$f - re-clone the repo"; exit 1
    fi
done
echo "    cascades present - OK"

echo "==> Granting passwordless shutdown for the dashboard button"
if ! sudo -n true 2>/dev/null; then
    echo "$USER ALL=(ALL) NOPASSWD: /sbin/shutdown" | sudo tee /etc/sudoers.d/020_shutdown > /dev/null
    sudo chmod 0440 /etc/sudoers.d/020_shutdown
fi

echo
echo "Setup complete. Run with:"
echo "  python3 AI_driver_drowiness_detection_main_code.py"
echo "Then open http://$(hostname):5000/"
