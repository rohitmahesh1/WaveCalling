// frontend/src/hooks/useSSE.ts
import * as React from "react";

export type SSEStatus = "idle" | "connecting" | "open" | "error" | "closed";

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
  /** Optional per-message callback. */
  onMessage?: (msg: SSEMessage<T>) => void;
  /** Suppress repeated identical `text` messages received within this window (ms). Default 250ms. */
  dedupeWindowMs?: number;
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

export function useSSE<T = any>(opts: UseSSEOptions<T>) {
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

  // Deduplication refs
  const lastTextRef = React.useRef<string>("");
  const lastTextAtRef = React.useRef<number>(0);

  React.useEffect(() => { pausedRef.current = pausedProp; _setPaused(pausedProp); }, [pausedProp]);
  React.useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);
  React.useEffect(() => { maxBufferRef.current = maxBuffer; }, [maxBuffer]);

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
        } catch {}
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
