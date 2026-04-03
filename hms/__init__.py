"""HMS v4.0.0 — Hierarchical Memory Scaffold.

OpenClaw Native Plugin features:
- HMSPlugin: native OpenClaw plugin entry point
- ReconstructiveRecaller: LLM-based memory reconstruction
- DreamEngine: dream consolidation and distant association discovery
- CreativeAssociator: creative cross-topic association
- MemoryOverwriter: conflict resolution and belief supersession

Integration:
- MemoryAdapter: native tool-first with Gateway HTTP fallback
- LLM routing via Gateway proxy (model="openclaw" = current model)
- Hooks auto-register cron jobs on first import
- Security: no hardcoded secrets, env whitelist, token redaction
- setup_wizard CLI for easy onboarding
"""
__version__ = "4.0.0"
__author__ = "CNMD-AMLK"

# Plugin entry point
from .plugin import HMSPlugin  # noqa: F401
