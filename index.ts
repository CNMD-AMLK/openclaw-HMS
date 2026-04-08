// index.ts — OpenClaw HMS v5 Plugin
import { HMSBridge } from "./bridge";

let bridge: HMSBridge;

export default function register(api: any) {
  const config = api.config ?? {};
  bridge = new HMSBridge(config, api.logger);

  // ━━━ Register Tools ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  api.registerTool({
    name: "hms_perceive",
    description: "Perceive user message, retrieve relevant memories and return context.",
    parameters: {
      type: "object",
      properties: {
        message: { type: "string", description: "Message to perceive" },
      },
      required: ["message"],
    },
    async execute(_id: string, params: { message: string }) {
      const result = await bridge.call("perceive", { message: params.message });
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    },
  });

  api.registerTool({
    name: "hms_recall",
    description: "Reconstructive memory recall using embedding vectors.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Recall query" },
        top_k: { type: "number", default: 5 },
        use_reconstructive: { type: "boolean", default: true },
      },
      required: ["query"],
    },
    async execute(_id: string, params: any) {
      const result = await bridge.call("recall", params);
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    },
  });

  api.registerTool({
    name: "hms_consolidate",
    description: "Trigger memory consolidation: compression, dream, creative, forgetting.",
    parameters: { type: "object", properties: {} },
    async execute(_id: string) {
      const result = await bridge.call("consolidate");
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    },
  });

  api.registerTool({
    name: "hms_context_inject",
    description: "Assemble memory-augmented context for prompt injection.",
    parameters: {
      type: "object",
      properties: {
        message: { type: "string", description: "Current user message" },
        tier: { type: "string", default: "auto" },
      },
      required: ["message"],
    },
    async execute(_id: string, params: any) {
      const result = await bridge.call("context_inject", params);
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    },
  });

  api.registerTool({
    name: "hms_forget",
    description: "Trigger forgetting evaluation. Destructive, optional by default.",
    parameters: { type: "object", properties: {} },
    async execute(_id: string) {
      const result = await bridge.call("forget");
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    },
  });

  api.registerTool({
    name: "hms_health",
    description: "HMS health check — returns daemon status, storage stats, pending queue.",
    parameters: {
      type: "object",
      properties: {
        detail: { type: "boolean", default: false },
      },
    },
    async execute(_id: string, params: any) {
      if (!bridge.isReady()) {
        return { content: [{ type: "text", text: JSON.stringify({ status: "down", ready: false }) }] };
      }
      try {
        const result = await bridge.call("health", params || {});
        return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
      } catch (e: any) {
        return { content: [{ type: "text", text: JSON.stringify({ status: "error", error: e.message }) }] };
      }
    },
  });

  // ━━━ Lifecycle Hooks ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  // Before prompt: auto-inject memory context
  if (config.autoInject !== false) {
    api.on("before_prompt_build", async (event: any) => {
      try {
        if (!bridge.isReady()) return;
        const messages = event.messages || [];
        const lastUserMsg = [...messages].reverse().find((m: any) => m.role === "user");
        if (!lastUserMsg?.content) return;

        const content = typeof lastUserMsg.content === "string"
          ? lastUserMsg.content
          : JSON.stringify(lastUserMsg.content);

        const result = await bridge.call("context_inject", {
          message: content,
          tier: config.contextTier || "auto",
        });

        if (result?.context && typeof result.context === "string" && result.context.trim()) {
          return { prependContext: `[HMS 记忆]\n${result.context}\n[/HMS]\n` };
        }
      } catch (e: any) {
        api.logger?.warn?.(`[hms] memory injection failed: ${e.message}`);
      }
    }, { priority: 5 });
  }

  // After agent turn: capture conversation to pending queue
  api.on("after_agent_turn", async (event: any) => {
    try {
      if (!bridge.isReady()) return;
      const messages = event.messages || [];
      const userMsg = [...messages].reverse().find((m: any) => m.role === "user");
      const asstMsg = [...messages].reverse().find((m: any) => m.role === "assistant");
      if (userMsg?.content && asstMsg?.content) {
        await bridge.call("capture", {
          user_message: typeof userMsg.content === "string" ? userMsg.content : JSON.stringify(userMsg.content),
          assistant_reply: typeof asstMsg.content === "string" ? asstMsg.content : JSON.stringify(asstMsg.content),
        }).catch(() => {});
      }
    } catch {}
  });

  // ━━━ Background Services ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  // Connection health monitor
  if (config.enableHealthCheck !== false) {
    let healthInterval: ReturnType<typeof setInterval> | null = null;
    api.registerService({
      id: "hms-health-monitor",
      start: () => {
        bridge.start().catch((e) => api.logger.warn(`[hms] initial connect failed: ${e.message}`));
        healthInterval = setInterval(async () => {
          if (!bridge.isReady()) {
            api.logger.warn("[hms] socket disconnected, attempting reconnect...");
            try { await bridge.start(); } catch {}
          }
        }, 60_000);
      },
      stop: async () => {
        if (healthInterval) clearInterval(healthInterval);
        await bridge.stop();
      },
    });
  }

  // Scheduled consolidation (daily at 3am)
  let consolidateInterval: ReturnType<typeof setInterval> | null = null;
  api.registerService({
    id: "hms-consolidate-cron",
    start: () => {
      const now = new Date();
      const target = new Date(now);
      target.setHours(3, 0, 0, 0);
      if (target <= now) target.setDate(target.getDate() + 1);
      const initialDelay = target.getTime() - now.getTime();

      setTimeout(() => {
        const run = async () => {
          try {
            api.logger.info("[hms] scheduled consolidation");
            await bridge.call("consolidate");
          } catch (e: any) {
            api.logger.warn(`[hms] consolidation failed: ${e.message}`);
          }
        };
        run();
        consolidateInterval = setInterval(run, 24 * 60 * 60 * 1000);
      }, initialDelay);
    },
    stop: () => { if (consolidateInterval) clearInterval(consolidateInterval); },
  });

  api.logger.info("[hms] plugin registered v5.0.0");
}
