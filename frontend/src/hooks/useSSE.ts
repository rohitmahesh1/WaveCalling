// frontend/src/hooks/useSSE.ts
import * as React from "react";

export type SSEStatus = "idle" | "connecting" | "open" | "error" | "closed";

export type DirtyFlags = Partial<{
  snapshot: boolean;
  overlay: boolean;
  progress: boolean;
  artifacts: boolean;
  list: boolean;
  [k: string]: boolean | undefined;
}>;

export interface BackendEvent {
  phase?: string;
  status?: string;                 // QUEUED | RUNNING | DONE | ERROR | CANCELLED
  dirty?: DirtyFlags;
  overlay_version?: number;
  message?: string;
  progress?: number;
  [k: string]: any;
}

export interface SSEMessage<T = any> {
  raw: MessageEvent;
  data: T | null;
  text: string;
  lastEventId?: string;
}

export interface UseSSEOptions<T = any> {
  url?: string | null;
  withCredentials?: boolean;       // default true (IMPORTANT for cross-origin cookies)
  autoReconnect?: boolean;
  reconnectDelayMs?: number;
  maxReconnectDelayMs?: number;

  maxBuffer?: number;
  paused?: boolean;
  onMessage?: (msg: SSEMessage<T>) => void;
  dedupeWindowMs?: number;

  onDirty?: (flags: DirtyFlags, ctx: { signal: AbortSignal }) => Promise<void> | void;
  onStatus?: (status: string) => void;
  onOpen?: (ev: Event) => void;    // NEW: notify caller when stream opens
  onError?: (ev: Event) => void;   // NEW: bubble error to caller

  stopOnTerminal?: boolean;
  coalesceWindowMs?: number;
}

const NAMED_EVENTS = ["log", "progress", "overlay", "phase", "status", "ping"] as const;

export function useSSE<T = BackendEvent>(opts: UseSSEOptions<T>) {
  const {
    url,
    withCredentials = true,        // << default TRUE
    autoReconnect = true,
    reconnectDelayMs = 800,
    maxReconnectDelayMs = 10_000,

    maxBuffer = 500,
    paused: pausedProp = false,
    onMessage,
    dedupeWindowMs = 250,

    onDirty,
    onStatus,
    onOpen,
    onError,

    stopOnTerminal = true,
    coalesceWindowMs = 50,
  } = opts;

  const [status, setStatus] = React.useState<SSEStatus>("idle");
  const [last, setLast] = React.useState<SSEMessage<T> | null>(null);
  const [messages, setMessages] = React.useState<SSEMessage<T>[]>([]);
  const [paused, _setPaused] = React.useState<boolean>(pausedProp);

  const esRef = React.useRef<EventSource | null>(null);
  const timerRef = React.useRef<number | null>(null);
  const backoffRef = React.useRef<number>(reconnectDelayMs);
  const mountedRef = React.useRef<boolean>(false);
  const [connKey, setConnKey] = React.useState<number>(0);

  const pausedRef = React.useRef<boolean>(pausedProp);
  const onMessageRef = React.useRef<typeof onMessage>(onMessage);
  const maxBufferRef = React.useRef<number>(maxBuffer);
  const onDirtyRef = React.useRef<typeof onDirty>(onDirty);
  const onStatusRef = React.useRef<typeof onStatus>(onStatus);
  const onOpenRef = React.useRef<typeof onOpen>(onOpen);
  const onErrorRef = React.useRef<typeof onError>(onError);
  const stopOnTerminalRef = React.useRef<boolean>(stopOnTerminal);
  const coalesceMsRef = React.useRef<number>(coalesceWindowMs);

  const lastTextRef = React.useRef<string>("");
  const lastTextAtRef = React.useRef<number>(0);

  const lastStatusRef = React.useRef<string | undefined>(undefined);
  const lastOverlayVersionRef = React.useRef<number | undefined>(undefined);

  const dirtyCoalesceTimerRef = React.useRef<number | null>(null);
  const pendingDirtyRef = React.useRef<DirtyFlags | null>(null);
  const inflightAbortRef = React.useRef<AbortController | null>(null);

  React.useEffect(() => { pausedRef.current = pausedProp; _setPaused(pausedProp); }, [pausedProp]);
  React.useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);
  React.useEffect(() => { maxBufferRef.current = maxBuffer; }, [maxBuffer]);
  React.useEffect(() => { onDirtyRef.current = onDirty; }, [onDirty]);
  React.useEffect(() => { onStatusRef.current = onStatus; }, [onStatus]);
  React.useEffect(() => { onOpenRef.current = onOpen; }, [onOpen]);
  React.useEffect(() => { onErrorRef.current = onError; }, [onError]);
  React.useEffect(() => { stopOnTerminalRef.current = stopOnTerminal; }, [stopOnTerminal]);
  React.useEffect(() => { coalesceMsRef.current = coalesceWindowMs; }, [coalesceWindowMs]);

  const setPaused = React.useCallback((p: boolean) => {
    pausedRef.current = p;
    _setPaused(p);
  }, []);

  const clearTimer = React.useCallback(() => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const close = React.useCallback((reason: "manual" | "error" | "cleanup" = "manual") => {
    clearTimer();
    if (dirtyCoalesceTimerRef.current) {
      window.clearTimeout(dirtyCoalesceTimerRef.current);
      dirtyCoalesceTimerRef.current = null;
    }
    inflightAbortRef.current?.abort();
    inflightAbortRef.current = null;

    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    if (mountedRef.current && reason !== "error") {
      setStatus("closed");
    }
  }, [clearTimer]);

  const requestReconnect = React.useCallback(() => {
    const delay = backoffRef.current;
    backoffRef.current = Math.min(Math.max(Math.floor(backoffRef.current * 1.7), reconnectDelayMs), maxReconnectDelayMs);
    clearTimer();
    timerRef.current = window.setTimeout(() => setConnKey((k) => k + 1), delay) as unknown as number;
  }, [clearTimer, reconnectDelayMs, maxReconnectDelayMs]);

  React.useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      close("cleanup");
    };
  }, [close]);

  const scheduleDirty = React.useCallback((flags: DirtyFlags) => {
    if (!onDirtyRef.current) return;
    pendingDirtyRef.current = { ...(pendingDirtyRef.current || {}), ...flags };
    if (dirtyCoalesceTimerRef.current) return;

    dirtyCoalesceTimerRef.current = window.setTimeout(async () => {
      dirtyCoalesceTimerRef.current = null;
      const pending = pendingDirtyRef.current;
      pendingDirtyRef.current = null;
      if (!pending) return;

      inflightAbortRef.current?.abort();
      const ac = new AbortController();
      inflightAbortRef.current = ac;
      try { await onDirtyRef.current?.(pending, { signal: ac.signal }); }
      catch { /* swallow */ }
      finally { if (inflightAbortRef.current === ac) inflightAbortRef.current = null; }
    }, coalesceMsRef.current) as unknown as number;
  }, []);

  // single handler used for both default and named events
  const handleEvent = React.useCallback((evt: MessageEvent) => {
    const text = (evt.data ?? "") as string;

    // dedupe identical bursts
    const now = Date.now();
    if (text && text === lastTextRef.current && now - lastTextAtRef.current <= (opts.dedupeWindowMs ?? 250)) {
      return;
    }
    lastTextRef.current = text;
    lastTextAtRef.current = now;

    let parsed: T | null = null;
    if (typeof text === "string" && text.length) {
      try { parsed = JSON.parse(text) as T; } catch { /* not JSON */ }
    }

    if (parsed && typeof parsed === "object") {
      const be = parsed as unknown as BackendEvent;

      if (be.status && be.status !== lastStatusRef.current) {
        lastStatusRef.current = be.status;
        onStatusRef.current?.(be.status);
        if (stopOnTerminalRef.current && /^(DONE|ERROR|CANCELLED)$/i.test(be.status)) {
          setTimeout(() => close("manual"), 0);
        }
      }

      let flags: DirtyFlags | null = null;
      if (be.dirty && typeof be.dirty === "object") flags = { ...(be.dirty as DirtyFlags) };

      if (typeof be.overlay_version === "number") {
        const prev = lastOverlayVersionRef.current;
        if (prev === undefined) {
          lastOverlayVersionRef.current = be.overlay_version;
        } else if (be.overlay_version > prev) {
          lastOverlayVersionRef.current = be.overlay_version;
          flags = { ...(flags || {}), overlay: true };
        }
      }

      if (flags) scheduleDirty(flags);
    }

    const msg: SSEMessage<T> = { raw: evt, text, data: parsed, lastEventId: (evt as any).lastEventId };
    if (!mountedRef.current) return;

    setLast(msg);
    if (!pausedRef.current) {
      const cb = onMessageRef.current;
      try { cb?.(msg); } catch {}
      setMessages((prev) => {
        const next = [...prev, msg];
        const cap = maxBufferRef.current ?? 500;
        return next.length > cap ? next.slice(next.length - cap) : next;
      });
    }
  }, [close, scheduleDirty, opts.dedupeWindowMs]);

  React.useEffect(() => {
    if (!url) {
      close();
      if (mountedRef.current) setStatus("idle");
      return;
    }

    close();
    if (mountedRef.current) setStatus("connecting");

    let finalUrl: string;
    try { finalUrl = new URL(url, window.location.origin).toString(); }
    catch { if (mountedRef.current) setStatus("error"); return; }

    const es = new EventSource(finalUrl, { withCredentials });
    esRef.current = es;

    es.addEventListener("open", (ev) => {
      backoffRef.current = reconnectDelayMs;
      if (mountedRef.current) setStatus("open");
      onOpenRef.current?.(ev);
    });

    es.addEventListener("error", (ev) => {
      if (mountedRef.current) setStatus("error");
      onErrorRef.current?.(ev);
      close("error");
      if (autoReconnect) requestReconnect();
    });

    // default channel
    es.onmessage = handleEvent;
    // named channels (IMPORTANT)
    for (const name of NAMED_EVENTS) {
      es.addEventListener(name, handleEvent as EventListener);
    }

    return () => {
      es.close();
      esRef.current = null;
      clearTimer();
    };
  }, [
    url,
    withCredentials,
    connKey,
    autoReconnect,
    reconnectDelayMs,
    maxReconnectDelayMs,
    requestReconnect,
    close,
    handleEvent,
  ]);

  return {
    status,
    last,
    messages,
    paused,
    setPaused,
    close,
    reconnect: () => {
      backoffRef.current = reconnectDelayMs;
      setConnKey((k) => k + 1);
    },
  };
}
