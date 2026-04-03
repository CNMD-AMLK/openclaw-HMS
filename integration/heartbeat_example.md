# HMS Integration — HEARTBEAT.md Example

Add this to your `HEARTBEAT.md` to keep HMS running:

## HMS Memory Maintenance

```
### HMS Periodic Tasks (rotate through, 2-4 times per day)

1. **Process pending queue**: `python -m hms process_pending` (via OpenClaw plugin heartbeat)
2. **Health check** (weekly): `python -m hms health`
3. **Run dream cycle** (after consolidation): via `hms_consolidate` tool
4. **Check insights**: review `hms/cache/insights/` for discovered connections
```

### Via OpenClaw Plugin (recommended v4 way)

When HMS is loaded as an OpenClaw plugin, the heartbeat hook runs automatically:

```python
# HMSPlugin.on_heartbeat(ctx) is called by OpenClaw periodically
# It handles:
# - process_pending()
# - forgetting.flush()
# - health monitoring
```

No manual cron needed — OpenClaw manages the heartbeat schedule.

### Via Cron (fallback / standalone mode)

```bash
# Process pending every minute
openclaw cron add --schedule "* * * * *" --command "python -m hms process_pending"

# Daily consolidation at 3 AM
openclaw cron add --schedule "0 3 * * *" --command "python -m hms consolidate"

# Weekly forgetting at Sunday 4 AM
openclaw cron add --schedule "0 4 * * 0" --command "python -m hms forget"
```
