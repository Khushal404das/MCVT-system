#!/usr/bin/env bash
# ==============================================================================
# Multi-Camera Vehicle Tracking & Re-Identification System (MCVT)
# Final Year Project (FYP) Launcher Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/mcvt-system"

echo "=================================================================="
echo "   MCVT VISION: Multi-Camera Vehicle Tracking & Re-ID System      "
echo "=================================================================="
echo ""

# Check python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is required but not installed."
    exit 1
fi

PHP_PID=""
cleanup() {
    echo ""
    echo "[MCVT] Shutting down services..."
    if [ -n "$PHP_PID" ]; then
        kill "$PHP_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Check PHP
if command -v php &> /dev/null; then
    echo "[✓] PHP detected: $(php -v | head -n 1)"
    echo "[*] Starting PHP Built-in Server on http://localhost:8080..."
    php -S 0.0.0.0:8080 -t "$SYSTEM_DIR/frontend" > /dev/null 2>&1 &
    PHP_PID=$!
    echo "[✓] PHP Dashboard running at: http://localhost:8080/index.php"
else
    echo "[!] PHP is not installed on this system."
    echo "    To run the native PHP frontend, install PHP using:"
    echo "      sudo apt-get install php php-cli"
    echo ""
    echo "    Alternatively, you can access the dashboard directly via the"
    echo "    Python server at: http://localhost:5000/ui"
fi

echo ""
echo "[*] Starting Python AI Vision Core (Flask + YOLOv8 + ReID Engine)..."
cd "$SYSTEM_DIR/backend"
python3 app.py

