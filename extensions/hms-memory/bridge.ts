// bridge.ts — Unix Socket JSON-RPC client to HMS daemon (managed by systemd)
import { connect, type Socket } from "net";
import { resolve, join } from "path";
import { existsSync } from "fs";

interface RPCRequest {
  id: number;
  method: string;
  params: Record<string, any>;
}

interface RPCResponse {
  id: number;
  result?: any;
  error?: { code: number; message: string };
}

interface PendingCall {
  resolve: (value: any) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
  method: string;
}

export class HMSBridge {
  private socket: Socket | null = null;
  private pending = new Map<number, PendingCall>();
  private nextId = 1;
  private readonly timeout = 30_000;
  private readonly socketPath: string;
  private readonly config: Record<string, any>;
  private ready = false;
  private connectPromise: Promise<void> | null = null;
  private logger: any;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempt = 0;
  private readonly maxReconnect = 10;

  constructor(config: Record<string, any>, logger?: any) {
    this.config = config;
    const dataDir = config.dataDir || join(resolve(__dirname), "data");
    this.socketPath = join(dataDir, "hms.sock");
    this.logger = logger || console;
  }

  /** Connect to the running HMS daemon (managed by systemd) */
  async start(): Promise<void> {
    if (this.ready && this.socket) return;
    if (this.connectPromise) return this.connectPromise;
    this.connectPromise = this._connect();
    return this.connectPromise;
  }

  private async _connect(): Promise<void> {
    await this._waitForSocket();

    return new Promise<void>((resolve, reject) => {
      const sock = connect(this.socketPath, () => {
        this.socket = sock;
        this.ready = true;
        this.reconnectAttempt = 0;
        this.connectPromise = null;
        this.logger.info("[hms] connected to daemon");
        resolve();
      });

      let buffer = "";
      sock.on("data", (data) => {
        buffer += data.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.trim()) this._handleResponse(line);
        }
      });

      sock.on("error", (err) => {
        this.ready = false;
        if (this.socket === null) {
          this.connectPromise = null;
          reject(new Error(`HMS socket connect error: ${err.message}`));
        } else {
          this.logger.warn(`[hms] socket error: ${err.message}`);
          this._scheduleReconnect();
        }
      });

      sock.on("close", () => {
        const wasReady = this.ready;
        this.ready = false;
        this.socket = null;
        if (wasReady) {
          this._scheduleReconnect();
        }
      });

      sock.setEncoding("utf-8");

      setTimeout(() => {
        if (!this.ready) {
          sock.destroy();
          this.connectPromise = null;
          reject(new Error(`HMS socket connect timeout (${this.socketPath})`));
        }
      }, 10_000);
    });
  }

  /** Wait for socket file to appear */
  private async _waitForSocket(maxWait = 30_000): Promise<void> {
    const interval = 200;
    let waited = 0;
    while (!existsSync(this.socketPath)) {
      if (waited >= maxWait) {
        throw new Error(
          `HMS socket not found: ${this.socketPath}\n` +
          `Is hms-core.service running? Try: systemctl status hms-core`
        );
      }
      await new Promise((r) => setTimeout(r, interval));
      waited += interval;
    }
  }

  /** Exponential backoff reconnection */
  private _scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    if (this.reconnectAttempt >= this.maxReconnect) {
      this.logger.error(`[hms] max reconnect attempts (${this.maxReconnect}) reached`);
      return;
    }
    const delay = Math.min(1000 * 2 ** this.reconnectAttempt, 30_000);
    this.reconnectAttempt++;
    this.logger.info(`[hms] reconnecting in ${delay}ms (attempt ${this.reconnectAttempt})`);
    this.reconnectTimer = setTimeout(async () => {
      this.reconnectTimer = null;
      try {
        await this._connect();
      } catch {}
    }, delay);
  }

  /** Send RPC request */
  async call(method: string, params: Record<string, any> = {}): Promise<any> {
    if (!this.socket || !this.ready) {
      await this.start();
    }

    const id = this.nextId++;
    const request: RPCRequest = { id, method, params };

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`HMS RPC timeout: ${method} (${this.timeout}ms)`));
      }, this.timeout);

      this.pending.set(id, { resolve, reject, timer, method });
      this.socket!.write(JSON.stringify(request) + "\n");
    });
  }

  /** Disconnect (do NOT kill the daemon process — systemd manages it) */
  async stop(): Promise<void> {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectAttempt = this.maxReconnect;

    for (const [id, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(new Error("HMS bridge stopping"));
    }
    this.pending.clear();

    this.socket?.destroy();
    this.socket = null;
    this.ready = false;
  }

  isReady(): boolean {
    return this.ready && this.socket !== null;
  }

  private _handleResponse(line: string): void {
    try {
      const resp: RPCResponse = JSON.parse(line);
      const p = this.pending.get(resp.id);
      if (!p) return;
      clearTimeout(p.timer);
      this.pending.delete(resp.id);
      if (resp.error) {
        p.reject(new Error(`HMS ${p.method}: ${resp.error.message}`));
      } else {
        p.resolve(resp.result);
      }
    } catch {
      this.logger.error(`[hms] invalid response: ${line.slice(0, 200)}`);
    }
  }
}
