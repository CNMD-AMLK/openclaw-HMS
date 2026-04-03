#!/bin/bash
# ============================================================================
# HMS Setup Script — Hierarchical Memory Scaffold (LLM-driven)
#
# Usage: bash setup.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HMS_DIR="${SCRIPT_DIR}/hms"

echo "============================================"
echo "HMS Setup — LLM-driven Cognitive Memory"
echo "============================================"
echo ""

# ---- Step 1: Check Python version ----
echo "[1/8] Checking Python version..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3.10+ is required but not found."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "  ✓ Python $PYTHON_VERSION found"

# ---- Step 2: Detect OpenClaw Gateway ----
echo "[2/8] Detecting OpenClaw Gateway..."
DEFAULT_GATEWAY="http://127.0.0.1:18789"
GATEWAY_URL="${HMS_GATEWAY_URL:-${DEFAULT_GATEWAY}}"
GATEWAY_REACHABLE=false

if curl -s --max-time 3 "${GATEWAY_URL}/health" > /dev/null 2>&1; then
    echo "  ✓ OpenClaw Gateway found at ${GATEWAY_URL}"
    GATEWAY_REACHABLE=true
else
    echo "  ⚠ OpenClaw Gateway not reachable at ${GATEWAY_URL}"
    echo "    HMS will use heuristic fallbacks until Gateway is available."
fi

# ---- Step 3: Install dependencies ----
echo "[3/8] Installing Python dependencies..."
if $PYTHON_CMD -m pip install --quiet -r "${SCRIPT_DIR}/requirements.txt" 2>/dev/null; then
    echo "  ✓ Dependencies installed"
else
    echo "  ⚠ Dependency installation failed (may need --break-system-packages)"
fi

# ---- Step 4: Create directory structure ----
echo "[4/8] Creating directory structure..."
mkdir -p "${HMS_DIR}/cache"
mkdir -p "${HMS_DIR}/logs"
mkdir -p "${HMS_DIR}/hooks"
mkdir -p "${HMS_DIR}/prompts"
echo "  ✓ Directories created"

# ---- Step 5: Initialize cache files ----
echo "[5/8] Initializing cache files..."

init_json() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "{}" > "$file"
        echo "  ✓ Created $(basename $file)"
    else
        echo "  ✓ $(basename $file) already exists"
    fi
}

init_json "${HMS_DIR}/cache/beliefs.json"
init_json "${HMS_DIR}/cache/cognitive_fingerprint.json"
init_json "${HMS_DIR}/cache/topic_timelines.json"
init_json "${HMS_DIR}/cache/decay_state.json"

if [ ! -f "${HMS_DIR}/cache/compression_history.json" ]; then
    echo "[]" > "${HMS_DIR}/cache/compression_history.json"
    echo "  ✓ Created compression_history.json"
fi

if [ ! -f "${HMS_DIR}/cache/pending_processing.jsonl" ]; then
    touch "${HMS_DIR}/cache/pending_processing.jsonl"
    echo "  ✓ Created pending_processing.jsonl"
fi

# ---- Step 6: Setup .env configuration ----
echo "[6/8] Configuring environment..."

ENV_FILE="${HMS_DIR}/.env"
ENV_EXAMPLE="${SCRIPT_DIR}/.env.example"

if [ -f "${ENV_FILE}" ]; then
    echo "  ✓ .env file already exists"
elif [ -f "${ENV_EXAMPLE}" ]; then
    echo "  ℹ Copying .env.example to .env (please edit it)"
    cp "${ENV_EXAMPLE}" "${ENV_FILE}"
    echo "  ✓ .env created from .env.example"
    echo "  ⚠ Please edit ${ENV_FILE} to set HMS_GATEWAY_TOKEN"
else
    # Create minimal .env
    cat > "${ENV_FILE}" << 'ENVEOF'
HMS_GATEWAY_URL=http://127.0.0.1:18789
HMS_GATEWAY_TOKEN=
HMS_LLM_MODEL=openclaw
ENVEOF
    echo "  ✓ Created minimal .env file"
    echo "  ⚠ Please edit ${ENV_FILE} to set HMS_GATEWAY_TOKEN"
fi

# Load .env if it exists
if [ -f "${ENV_FILE}" ]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

# ---- Step 7: Run health check ----
echo "[7/8] Running health check..."
cd "${SCRIPT_DIR}"
HEALTH_OUTPUT=$($PYTHON_CMD -m hms health 2>&1 || echo "health_check_failed")
if echo "$HEALTH_OUTPUT" | grep -q "health_check_failed"; then
    echo "  ⚠ Health check failed — check Gateway connectivity"
elif echo "$HEALTH_OUTPUT" | grep -q "gateway_reachable.*true"; then
    echo "  ✓ Gateway reachable"
else
    echo "  ℹ Gateway not reachable (normal if OpenClaw is not running)"
fi

# ---- Step 8: Set permissions and print summary ----
echo "[8/8] Setting permissions..."
chmod +x "${HMS_DIR}/scripts/"*.py 2>/dev/null || true
chmod +x "${SCRIPT_DIR}/setup.sh"
echo "  ✓ Permissions set"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit ${ENV_FILE} if you haven't set HMS_GATEWAY_TOKEN"
echo "  2. Register with OpenClaw (see below)"
echo ""
echo "OpenClaw Integration:"
echo ""
echo "  # Every minute — process pending memories"
echo '  openclaw cron add --schedule "* * * * *" --command "python -m hms process_pending"'
echo ""
echo "  # Daily at 3 AM — consolidate memories"
echo '  openclaw cron add --schedule "0 3 * * *" --command "python -m hms consolidate"'
echo ""
echo "  # Weekly Sunday at 4 AM — forget weak memories"
echo '  openclaw cron add --schedule "0 4 * * 0" --command "python -m hms forget"'
echo ""
echo "Or use as a Python module in your skill/plugin:"
echo "  from hms.hooks import on_message_received, on_message_sent"
echo ""
echo "Done! 🧠"
