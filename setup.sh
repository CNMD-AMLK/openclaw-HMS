#!/usr/bin/env bash
set -euo pipefail
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PLUGIN_DIR/data"
PYTHON="${PYTHON:-python3}"
SERVICE_NAME="hms-core"

echo "=== HMS v5 Plugin Setup ==="
echo "Plugin dir: $PLUGIN_DIR"

# Check Python
if ! command -v "$PYTHON" &>/dev/null; then
  echo "ERROR: $PYTHON not found. Install Python 3.10+ first."
  exit 1
fi
PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYVER"

# Install dependencies
echo "Installing Python dependencies..."
"$PYTHON" -m pip install --user -r "$PLUGIN_DIR/requirements.txt" 2>/dev/null || \
"$PYTHON" -m pip install -r "$PLUGIN_DIR/requirements.txt"

# Create data directory
mkdir -p "$DATA_DIR/insights" "$DATA_DIR/backups"

# Register systemd service
if command -v systemctl &>/dev/null && [ -d /etc/systemd/system ]; then
  echo "Registering systemd service: $SERVICE_NAME"
  SOCK_PATH="$DATA_DIR/hms.sock"

  sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null << EOF
[Unit]
Description=HMS Core v5 - Hierarchical Memory System
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PLUGIN_DIR
Environment=HMS_CONFIG={"dataDir":"\"$DATA_DIR\"","ollamaBaseUrl":"${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"}
Environment=HMS_OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON -u -m hms.daemon --mode rpc --socket $SOCK_PATH --data-dir "$DATA_DIR"
Restart=on-failure
RestartSec=3
TimeoutStopSec=10

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$DATA_DIR
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable "$SERVICE_NAME"
  sudo systemctl start "$SERVICE_NAME"
  echo "Service $SERVICE_NAME started."

  # Wait for socket
  echo -n "Waiting for socket..."
  for i in $(seq 1 30); do
    [ -S "$SOCK_PATH" ] && echo " ready!" && break
    echo -n "."
    sleep 1
  done
  if [ ! -S "$SOCK_PATH" ]; then
    echo " TIMEOUT"
    echo "Check: journalctl -u $SERVICE_NAME"
  fi
else
  echo "WARNING: systemd not available. HMS will run in Gateway-managed mode."
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Add to ~/.openclaw/openclaw.json:"
echo '     plugins.load.paths: ["~/.openclaw/extensions/hms-memory"]'
echo "  2. Restart gateway: openclaw gateway restart"
echo "  3. Verify: openclaw plugins list"
echo ""
echo "Service management:"
echo "  systemctl status $SERVICE_NAME"
echo "  journalctl -u $SERVICE_NAME -f"
echo "  systemctl restart $SERVICE_NAME"
