"""
HMS Daemon v5 — asyncio Unix Socket JSON-RPC Server.

Managed by systemd (hms-core.service), communicates with TS plugin via Unix Socket.
Supports concurrent request handling (each client connection = independent coroutine).
"""

import asyncio
import json
import os
import sys
import logging
import argparse
import signal
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[hms] %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("hms.daemon")

_manager = None
_config = None


def load_config() -> dict:
    raw = os.environ.get("HMS_CONFIG", "{}")
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        cfg = {}

    for key in ("dataDir",):
        if key in cfg:
            cfg[key] = os.path.expanduser(cfg[key])

    cfg.setdefault("dataDir", str(Path(__file__).parent.parent / "data"))
    cfg.setdefault("llmModel", "openclaw")
    cfg.setdefault("perceptionMode", "lite")
    cfg.setdefault("contextTier", "auto")
    cfg.setdefault("tokenBudgetDaily", 50000)
    cfg.setdefault("collisionThreshold", 0.7)
    cfg.setdefault("embeddingBackend", "ollama")
    # Map camelCase plugin config to snake_case internal keys (Issue #1 fix)
    if "ollamaBaseUrl" in cfg:
        cfg["embedding_ollama_base_url"] = cfg.pop("ollamaBaseUrl")
    if "embeddingModel" in cfg:
        cfg.setdefault("embedding_ollama_model", cfg.pop("embeddingModel"))
    cfg.setdefault("embedding_ollama_base_url", os.environ.get("HMS_OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    cfg.setdefault("embedding_ollama_model", "qwen3-embedding:0.6b")
    cfg.setdefault("embedding_ollama_dim", 1024)
    cfg.setdefault("retrievalTopK", 30)
    cfg.setdefault("processPendingMaxBatch", 50)
    cfg.setdefault("dedupSimilarityThreshold", 0.95)
    cfg.setdefault("cache_dir", cfg["dataDir"])
    return cfg


def get_manager():
    global _manager, _config
    if _manager is None:
        from hms.core.manager import MemoryManager
        _config = load_config()
        data_dir = _config["dataDir"]
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "insights"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "backups"), exist_ok=True)
        _manager = MemoryManager(_config)
        logger.info("MemoryManager initialized (data_dir=%s)", data_dir)
    return _manager


# ── RPC Methods ─────────────────────────────────────────────────────────────────

def rpc_perceive(params):
    mgr = get_manager()
    message = params.get("message", "")
    if not message:
        return {"error": "message is required"}
    return mgr.on_message_received(message)


def rpc_recall(params):
    mgr = get_manager()
    query = params.get("query", "")
    if not query:
        return {"error": "query is required"}

    from hms.engines.recall import ReconstructiveRecaller
    recaller = ReconstructiveRecaller(_config or {})
    perception = mgr.perception.analyze(query, "", force_heuristic=True)
    return recaller.recall(query, perception, top_k=params.get("top_k", 5))


def rpc_consolidate(params):
    mgr = get_manager()
    report = mgr.consolidate()
    try:
        from hms.engines.dream import DreamEngine
        dream = DreamEngine(_config or {})
        all_memories = mgr.adapter.get_all_memories(limit=100)
        insights = dream.analyze_cluster(all_memories)
        report["dream_insights"] = len(insights)
        for ins in insights:
            dream.save_insight(ins)
    except Exception as e:
        report["dream_error"] = str(e)
    try:
        from hms.engines.creative import CreativeAssociator
        assoc = CreativeAssociator(_config or {})
        links = assoc.find_cross_domain_links(mgr.adapter.get_all_memories(limit=100))
        report["creative_insights"] = len(links)
    except Exception as e:
        report["creative_error"] = str(e)
    return report


def rpc_context_inject(params):
    mgr = get_manager()
    message = params.get("message", "")
    if not message:
        return {"error": "message is required"}
    tier = params.get("tier", "auto")
    if tier != "auto":
        from hms.core.manager import MemoryManager as MM
        cfg = dict(_config or {})
        cfg.update(MM._apply_tier(cfg, tier))
        mgr = MemoryManager(cfg)
    result = mgr.on_message_received(message)
    return result.get("context", {})


def rpc_forget(params):
    mgr = get_manager()
    try:
        from hms.engines.forgetting import ForgettingEngine, MemoryOverwriter
        forgetting = ForgettingEngine(_config or {})
        overwriter = MemoryOverwriter(_config or {})
        memories = mgr.adapter.get_all_memories(limit=100) or []
        evaluation = forgetting.evaluate_all(memories, overwriter)
        if evaluation.get("to_forget"):
            deleted = forgetting.execute_forgetting(
                evaluation["to_forget"], lambda mid: mgr.adapter.forget(mid)
            )
            evaluation["deleted"] = deleted
        return evaluation
    except Exception as e:
        return {"error": str(e)}


def rpc_capture(params):
    mgr = get_manager()
    user_msg = params.get("user_message", "")
    assistant_reply = params.get("assistant_reply", "")
    if user_msg:
        mgr.on_message_sent(user_msg, assistant_reply)
    return {"status": "captured"}


def rpc_health(params):
    try:
        mgr = get_manager()
        result = {
            "status": "healthy",
            "daemon": "running",
            "data_dir": _config.get("dataDir", "unknown"),
        }
        if params.get("detail"):
            result.update(mgr.health_check())
        else:
            result["pending_queue"] = mgr._count_pending()
            result["total_memories"] = mgr.adapter.health_check().get("total_memories", 0)
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}


def rpc_shutdown(params):
    global _manager
    if _manager:
        try:
            _manager.close()
        except Exception:
            pass
    return {"status": "shutting_down"}


METHODS = {
    "perceive": rpc_perceive,
    "recall": rpc_recall,
    "consolidate": rpc_consolidate,
    "context_inject": rpc_context_inject,
    "forget": rpc_forget,
    "capture": rpc_capture,
    "health": rpc_health,
    "shutdown": rpc_shutdown,
}


def handle_request(request: dict) -> dict:
    req_id = request.get("id", 0)
    method = request.get("method", "")
    params = request.get("params", {})

    handler = METHODS.get(method)
    if not handler:
        return {"id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}

    try:
        result = handler(params)
        if isinstance(result, dict) and "error" in result and "id" not in result:
            return {"id": req_id, "error": {"code": -1, "message": str(result["error"])}}
        return {"id": req_id, "result": result}
    except Exception as e:
        logger.exception("Error in %s", method)
        return {"id": req_id, "error": {"code": -32000, "message": str(e)}}


# ── asyncio Unix Socket Server ─────────────────────────────────────────────────

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    logger.debug("Client connected: %s", peer)

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            line = line.decode("utf-8").strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from %s: %.100s", peer, line)
                continue

            response = handle_request(request)
            response_bytes = json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n"
            writer.write(response_bytes)
            await writer.drain()

            if request.get("method") == "shutdown":
                logger.info("Shutdown requested")
                asyncio.get_event_loop().call_later(0.5, lambda: os._exit(0))
                return
    except (ConnectionResetError, BrokenPipeError):
        logger.debug("Client disconnected: %s", peer)
    except Exception as e:
        logger.error("Client handler error: %s", e)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        logger.debug("Client handler cleaned up: %s", peer)


async def run_rpc_server(socket_path: str, data_dir: str):
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    os.makedirs(os.path.dirname(socket_path), exist_ok=True)
    get_manager()

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    os.chmod(socket_path, 0o600)

    logger.info("HMS daemon listening on %s", socket_path)
    sys.stderr.write("HMS_READY\n")
    sys.stderr.flush()

    async with server:
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="HMS Daemon v5")
    parser.add_argument("--mode", default="rpc", choices=["rpc", "cli"])
    parser.add_argument("--socket", default="", help="Unix socket path")
    parser.add_argument("--data-dir", default="", help="Data directory")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["HMS_DATA_DIR"] = args.data_dir

    config = load_config()
    socket_path = args.socket or os.path.join(config["dataDir"], "hms.sock")

    loop = asyncio.new_event_loop()

    def on_signal(sig):
        logger.info("Signal %d received", sig)
        rpc_shutdown({})
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: on_signal(s))
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(run_rpc_server(socket_path, config["dataDir"]))
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Daemon exited")


if __name__ == "__main__":
    main()
