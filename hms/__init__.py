"""HMS v3.6.3 — Hierarchical Memory Scaffold.

Integration fixes for OpenClaw:
- MemoryAdapter: native tool-first with Gateway HTTP fallback
- LLM routing via Gateway proxy (model="openclaw" = current model)
- Hooks auto-register cron jobs on first import
- Security: no hardcoded secrets, env whitelist, token redaction
- setup_wizard CLI for easy onboarding
"""
__version__ = "3.6.3"

# v3.6.3: Only setup logging when running as standalone CLI,
# NOT when imported as a module (avoids clobbering caller's logging config).
import sys as _sys
if hasattr(_sys, "argv") and len(_sys.argv) > 0:
    _is_cli = True
    try:
        from .scripts import setup_logging as _setup_logging
        _setup_logging()
    except Exception:
        pass
