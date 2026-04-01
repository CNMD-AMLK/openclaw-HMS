#!/bin/bash
# ============================================================================
# HMS v3 Setup Script — Hierarchical Memory Scaffold (LLM-driven)
#
# Usage: bash setup.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HMS_DIR="${SCRIPT_DIR}/hms"

echo "============================================"
echo "HMS v3 Setup — LLM-driven Cognitive Memory"
echo "============================================"
echo ""

# ---- Step 1: Check Python version ----
echo "[1/6] Checking Python version..."
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

# ---- Step 2: Create directory structure ----
echo "[2/6] Creating directory structure..."
mkdir -p "${HMS_DIR}/cache"
mkdir -p "${HMS_DIR}/logs"
mkdir -p "${HMS_DIR}/hooks"
mkdir -p "${HMS_DIR}/prompts"
echo "  ✓ Directories created"

# ---- Step 3: Initialize cache files ----
echo "[3/6] Initializing cache files..."

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
else
    echo "  ✓ compression_history.json already exists"
fi

if [ ! -f "${HMS_DIR}/cache/active_context.md" ]; then
    echo "# Active Context" > "${HMS_DIR}/cache/active_context.md"
    echo "  ✓ Created active_context.md"
else
    echo "  ✓ active_context.md already exists"
fi

if [ ! -f "${HMS_DIR}/cache/pending_processing.jsonl" ]; then
    touch "${HMS_DIR}/cache/pending_processing.jsonl"
    echo "  ✓ Created pending_processing.jsonl"
else
    echo "  ✓ pending_processing.jsonl already exists"
fi

# ---- Step 4: Set permissions ----
echo "[4/6] Setting file permissions..."
chmod +x "${HMS_DIR}/scripts/"*.py 2>/dev/null || true
chmod +x "${SCRIPT_DIR}/setup.sh"
echo "  ✓ Permissions set"

# ---- Step 5: Run tests ----
echo "[5/6] Running tests..."
cd "${SCRIPT_DIR}"
if $PYTHON_CMD hms/scripts/test_e2e.py; then
    echo "  ✓ All tests passed"
else
    echo "  ✗ Some tests failed. Please check the output above."
    echo "  (LLM-dependent tests may fail if OpenClaw model is not configured)"
    echo "  Continuing with setup..."
fi

# ---- Step 6: Print OpenClaw commands ----
echo "[6/6] Setup complete!"
echo ""
echo "============================================"
echo "OpenClaw Integration Commands"
echo "============================================"
echo ""
echo "Run these commands to register HMS v3 with OpenClaw:"
echo ""
echo "# Register hooks"
echo 'openclaw hook register message:received "python3 '"${HMS_DIR}"'/scripts/memory_manager.py received"'
echo 'openclaw hook register before:compaction "python3 '"${HMS_DIR}"'/scripts/memory_manager.py process_pending"'
echo ""
echo "# Register cron jobs"
echo 'openclaw cron add "0 3 * * *" "python3 '"${HMS_DIR}"'/scripts/memory_manager.py consolidate"'
echo 'openclaw cron add "0 4 * * 0" "python3 '"${HMS_DIR}"'/scripts/memory_manager.py forget"'
echo ""
echo "============================================"
echo "v3 Key Features"
echo "============================================"
echo ""
echo "  • LLM-driven perception (replaces dictionary-based)"
echo "  • Three-layer infinite context:"
echo "    Layer 1: Cognitive fingerprint (~2000 tokens, always present)"
echo "    Layer 2: Topic timelines + compressed summaries"
echo "    Layer 3: Recent turns + injected memories"
echo "  • Automatic conversation compression"
echo "  • Dynamic cognitive profile updates"
echo ""
echo "============================================"
echo "Configuration"
echo "============================================"
echo ""
echo "Edit ${HMS_DIR}/config.json to configure:"
echo "  • llm_perception_mode: 'full' | 'lite' | 'llm_only'"
echo "  • llm_budget_tokens_per_day: daily LLM token budget"
echo "  • compression_window_turns: turns per compression batch"
echo "  • context_budget: token allocation ratios"
echo ""
echo "Done! 🧠"
