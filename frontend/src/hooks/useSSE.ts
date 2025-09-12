// frontend/src/hooks/useSSE.ts
import * as React from "react";

export type SSEStatus = "idle" | "connecting" | "open" | "error" | "closed";

/** Dirty flags our backend may emit over SSE. */
export type DirtyFlags = Partial<{
  snapshot: boolean;
  overlay: boolean;
  progress: boolean;
  artifacts: boolean;
  /** used by the global runs stream */
  list: boolean;
  /** custom/extra flags are allowed at runtime */
  [k: string]: boolean | undefined;
}>;

export interface BackendEvent {
  /** Legacy event field; keep for back-compat. */
  phase?: string;
  /** Canonical run status: QUEUED | RUNNING | DONE | ERROR | CANCELLED */
  status?: string;
  /** Backend "what changed" hints; UI should refetch only these. */
  dirty?: DirtyFlags;
  /** Monotone overlay version; bump means overlay changed. */
  overlay_version?: number;
  /** Free-form message fields from your JobEvent */
  message?: string;
  progress?: number;
  extra?: any;
  [k: string]: any;
}

export interface SSEMessage<T = any> {
  raw: MessageEvent;
  /** Parsed JSON payload if available, otherwise null. */
  data: T | null;
  /** The raw text received from the SSE `data:` line. */
  text: string;
  /** Last-Event-ID, if provided by the server. */
  lastEventId?: string;
}

export interface UseSSEOptions<T = any> {
  /** If API is on a different origin, pass absolute URL. If null/undefined, the hook stays idle. */
  url?: string | null;
  /** Usually false for our API. */
  withCredentials?: boolean;
  /** Reconnect automatically on error/close. */
  autoReconnect?: boolean;
  /** Initial reconnect delay (ms). */
  reconnectDelayMs?: number;
  /** Maximum reconnect delay (ms). */
  maxReconnectDelayMs?: number;

  /** Keep at most the last N messages in memory (prevents unbounded growth). Default 500. */
  maxBuffer?: number;
  /** Start in paused mode (still connected; we just don't push to the buffer/callback). */
  paused?: boolean;
  /** Optional per-message callback (called after onDirty/onStatus fire). */
  onMessage?: (msg: SSEMessage<T>) => void;
  /** Suppress repeated identical `text` messages within this window (ms). Default 250ms. */
  dedupeWindowMs?: number;

  // === Event-driven extras (optional) ===

  /**
   * When SSE indicates something changed, we call this once with the union of all pending flags.
   * We always pass an AbortSignal; implementors should abort in-flight network requests when signaled.
   * If multiple SSE messages arrive quickly, flags are coalesced (see coalesceWindowMs).
   */
  onDirty?: (flags: DirtyFlags, ctx: { signal: AbortSignal }) => Promise<void> | void;

  /** Called when the run status changes (QUEUED/RUNNING/DONE/ERROR/CANCELLED). */
  onStatus?: (status: string) => void;

  /**
   * Auto-close the EventSource when a terminal status is observed (DONE/ERROR/CANCELLED).
   * Default true.
   */
  stopOnTerminal?: boolean;

  /** Coalesce multiple quick successive dirty messages (ms). Default 50ms. */
  coalesceWindowMs?: number;
}

export function useSSE<T = BackendEvent>(opts: UseSSEOptions<T>) {
  const {
    url,
    withCredentials = false,
    autoReconnect = true,
    reconnectDelayMs = 800,
    maxReconnectDelayMs = 10_000,

    maxBuffer = 500,
    paused: pausedProp = false,
    onMessage,
    dedupeWindowMs = 250,

    onDirty,
    onStatus,
    stopOnTerminal = true,
    coalesceWindowMs = 50,
  } = opts;

  // Public state
  const [status, setStatus] = React.useState<SSEStatus>("idle");
  const [last, setLast] = React.useState<SSEMessage<T> | null>(null);
  const [messages, setMessages] = React.useState<SSEMessage<T>[]>([]);
  const [paused, _setPaused] = React.useState<boolean>(pausedProp);

  // Internal refs
  const esRef = React.useRef<EventSource | null>(null);
  const timerRef = React.useRef<number | null>(null);
  const backoffRef = React.useRef<number>(reconnectDelayMs);
  const mountedRef = React.useRef<boolean>(false);
  const [connKey, setConnKey] = React.useState<number>(0);

  // Volatile options in refs
  const pausedRef = React.useRef<boolean>(pausedProp);
  const onMessageRef = React.useRef<typeof onMessage>(onMessage);
  const maxBufferRef = React.useRef<number>(maxBuffer);
  const onDirtyRef = React.useRef<typeof onDirty>(onDirty);
  const onStatusRef = React.useRef<typeof onStatus>(onStatus);
  const stopOnTerminalRef = React.useRef<boolean>(stopOnTerminal);
  const coalesceMsRef = React.useRef<number>(coalesceWindowMs);

  // Deduplication refs
  const lastTextRef = React.useRef<string>("");
  const lastTextAtRef = React.useRef<number>(0);

  // Event-driven state
  const lastStatusRef = React.useRef<string | undefined>(undefined);
  const lastOverlayVersionRef = React.useRef<number | undefined>(undefined);

  // Coalescing & cancellation for onDirty work
  const dirtyCoalesceTimerRef = React.useRef<number | null>(null);
  const pendingDirtyRef = React.useRef<DirtyFlags | null>(null);
  const inflightAbortRef = React.useRef<AbortController | null>(null);

  React.useEffect(() => { pausedRef.current = pausedProp; _setPaused(pausedProp); }, [pausedProp]);
  React.useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);
  React.useEffect(() => { maxBufferRef.current = maxBuffer; }, [maxBuffer]);
  React.useEffect(() => { onDirtyRef.current = onDirty; }, [onDirty]);
  React.useEffect(() => { onStatusRef.current = onStatus; }, [onStatus]);
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

  const close = React.useCallback(
    (reason: "manual" | "error" | "cleanup" = "manual") => {
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
    },
    [clearTimer]
  );

  const requestReconnect = React.useCallback(() => {
    const delay = backoffRef.current;
    backoffRef.current = Math.min(
      Math.max(Math.floor(backoffRef.current * 1.7), reconnectDelayMs),
      maxReconnectDelayMs
    );
    clearTimer();
    timerRef.current = window.setTimeout(() => {
      setConnKey((k) => k + 1);
    }, delay) as unknown as number;
  }, [clearTimer, reconnectDelayMs, maxReconnectDelayMs]);

  React.useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      close("cleanup");
    };
  }, [close]);

  // Coalesce and execute onDirty with abort/dedupe
  const scheduleDirty = React.useCallback((flags: DirtyFlags) => {
    if (!onDirtyRef.current) return;

    // Union new flags into pending
    pendingDirtyRef.current = {
      ...(pendingDirtyRef.current || {}),
      ...flags,
    };

    if (dirtyCoalesceTimerRef.current) return;

    dirtyCoalesceTimerRef.current = window.setTimeout(async () => {
      dirtyCoalesceTimerRef.current = null;
      const pending = pendingDirtyRef.current;
      pendingDirtyRef.current = null;
      if (!pending) return;

      // Abort any in-flight refresh work; start a new one
      inflightAbortRef.current?.abort();
      const ac = new AbortController();
      inflightAbortRef.current = ac;
      try {
        await onDirtyRef.current?.(pending, { signal: ac.signal });
      } catch {
        // swallow errors; UI handlers should log if needed
      } finally {
        if (inflightAbortRef.current === ac) {
          inflightAbortRef.current = null;
        }
      }
    }, coalesceMsRef.current) as unknown as number;
  }, []);

  React.useEffect(() => {
    if (!url) {
      close();
      if (mountedRef.current) setStatus("idle");
      return;
    }

    close();
    if (mountedRef.current) setStatus("connecting");

    let finalUrl: string;
    try {
      finalUrl = new URL(url, window.location.origin).toString();
    } catch {
      if (mountedRef.current) setStatus("error");
      return;
    }

    const es = new EventSource(finalUrl, { withCredentials });
    esRef.current = es;

    es.onopen = () => {
      backoffRef.current = reconnectDelayMs;
      if (mountedRef.current) setStatus("open");
    };

    es.onerror = () => {
      if (mountedRef.current) setStatus("error");
      close("error");
      if (autoReconnect) requestReconnect();
    };

    es.onmessage = (evt: MessageEvent) => {
      const text = (evt.data ?? "") as string;

      // deduplication check
      const now = Date.now();
      if (text && text === lastTextRef.current && now - lastTextAtRef.current <= dedupeWindowMs) {
        return;
      }
      lastTextRef.current = text;
      lastTextAtRef.current = now;

      let parsed: T | null = null;
      if (typeof text === "string" && text.length) {
        try {
          parsed = JSON.parse(text) as T;
        } catch {
          // leave parsed null if not JSON
        }
      }

      // Event-driven behavior for our backend shape
      if (parsed && typeof parsed === "object") {
        const be = parsed as unknown as BackendEvent;

        // Status changes
        if (be.status && be.status !== lastStatusRef.current) {
          lastStatusRef.current = be.status;
          onStatusRef.current?.(be.status);
          if (stopOnTerminalRef.current && /^(DONE|ERROR|CANCELLED)$/i.test(be.status)) {
            // allow the consumer to process last message before closing
            setTimeout(() => close("manual"), 0);
          }
        }

        // Dirty flags
        let flags: DirtyFlags | null = null;
        if (be.dirty && typeof be.dirty === "object") {
          flags = { ...(be.dirty as DirtyFlags) };
        }

        // Overlay version bump implies overlay dirty
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

      const msg: SSEMessage<T> = {
        raw: evt,
        text,
        data: parsed,
        lastEventId: (evt as any).lastEventId,
      };
      if (!mountedRef.current) return;

      setLast(msg);

      if (!pausedRef.current) {
        const cb = onMessageRef.current;
        if (cb) {
          try { cb(msg); } catch {}
        }
        setMessages((prev) => {
          const next = [...prev, msg];
          const cap = maxBufferRef.current ?? 500;
          return next.length > cap ? next.slice(next.length - cap) : next;
        });
      }
    };

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
    dedupeWindowMs,
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
