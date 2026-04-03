"""HMS — Hierarchical Memory Scaffold.

Integration fixes for OpenClaw:
- MemoryAdapter: native tool-first with Gateway HTTP fallback
- LLM routing via Gateway proxy (model="openclaw" = current model)
- Hooks auto-register cron jobs on first import
- Security: no hardcoded secrets, env whitelist, token redaction
- setup_wizard CLI for easy onboarding
"""

