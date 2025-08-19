// frontend/src/hooks/useSSE.ts
import * as React from "react";

export type SSEStatus = "idle" | "connecting" | "open" | "error" | "closed";

export interface UseSSEOptions {
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
}

export interface SSEMessage<T = any> {
  raw: MessageEvent;
  /** Parsed JSON payload if available, otherwise null. */
  data: T | null;
  /** The raw text received from the SSE `data:` line. */
  text: string;
}

export function useSSE<T = any>(opts: UseSSEOptions) {
  const {
    url,
    withCredentials = false,
    autoReconnect = true,
    reconnectDelayMs = 800,
    maxReconnectDelayMs = 10_000,
  } = opts;

  const [status, setStatus] = React.useState<SSEStatus>("idle");
  const [last, setLast] = React.useState<SSEMessage<T> | null>(null);
  const [messages, setMessages] = React.useState<SSEMessage<T>[]>([]);

  // internal refs/state
  const esRef = React.useRef<EventSource | null>(null);
  const timerRef = React.useRef<number | null>(null);
  const backoffRef = React.useRef<number>(reconnectDelayMs);
  const mountedRef = React.useRef<boolean>(false);
  const [connKey, setConnKey] = React.useState<number>(0); // bump to force reconnection

  const clearTimer = React.useCallback(() => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const close = React.useCallback(() => {
    clearTimer();
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    if (mountedRef.current) setStatus("closed");
  }, [clearTimer]);

  const requestReconnect = React.useCallback(() => {
    // exponential backoff within bounds
    const delay = backoffRef.current;
    backoffRef.current = Math.min(
      Math.max(Math.floor(backoffRef.current * 1.7), reconnectDelayMs),
      maxReconnectDelayMs
    );
    clearTimer();
    timerRef.current = window.setTimeout(() => {
      // bump key to re-run the effect and recreate the EventSource
      setConnKey((k) => k + 1);
    }, delay) as unknown as number;
  }, [clearTimer, reconnectDelayMs, maxReconnectDelayMs]);

  React.useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      close();
    };
  }, [close]);

  React.useEffect(() => {
    // If no URL, stay idle and ensure connection is closed.
    if (!url) {
      close();
      if (mountedRef.current) setStatus("idle");
      return;
    }

    // Always start with a clean slate for (re)connect
    close();
    if (mountedRef.current) setStatus("connecting");

    // Normalize URL (absolute or relative)
    const finalUrl = new URL(url, window.location.origin).toString();

    const es = new EventSource(finalUrl, { withCredentials });
    esRef.current = es;

    es.onopen = () => {
      backoffRef.current = reconnectDelayMs; // reset backoff
      if (mountedRef.current) setStatus("open");
    };

    es.onerror = () => {
      if (mountedRef.current) setStatus("error");
      close();
      if (autoReconnect) requestReconnect();
    };

    es.onmessage = (evt: MessageEvent) => {
      const text = (evt.data ?? "") as string;
      let parsed: T | null = null;
      if (typeof text === "string" && text.length) {
        try {
          parsed = JSON.parse(text) as T;
        } catch {
          // Non-JSON keep-alives or plain text are fine; leave parsed = null
        }
      }
      const msg: SSEMessage<T> = { raw: evt, text, data: parsed };
      if (!mountedRef.current) return;
      setLast(msg);
      setMessages((prev) => [...prev, msg]);
    };

    return () => {
      es.close();
      esRef.current = null;
      clearTimer();
      // don't force status here; `close()` is called by outer cleanup on unmount/url change
    };
    // Reconnect when url / credentials change OR when we bump connKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, withCredentials, connKey, autoReconnect, reconnectDelayMs, maxReconnectDelayMs, requestReconnect, close]);

  // Public API
  return {
    status,
    last,        // { data: T | null, text, raw }
    messages,    // all messages so far
    close,       // manually close
    reconnect: () => {
      backoffRef.current = reconnectDelayMs; // reset backoff
      setConnKey((k) => k + 1);
    },
  };
}
