/**
 * ViolenceSense - WebSocket Hook
 * ==============================
 * Real-time updates for inference scores and alerts.
 * Uses a global singleton to survive React 18 Strict Mode and HMR.
 */

import {
  useEffect,
  useRef,
  useCallback,
  useState,
  useSyncExternalStore,
} from "react";
import {
  InferenceScoreMessage,
  AlertMessage,
  StreamStatusMessage,
} from "@/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8080/ws";
const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY = 2000;

// ============================================================================
// Global WebSocket Manager (Singleton - lives outside React)
// ============================================================================

type MessageHandler = (type: string, data: any) => void;

interface WebSocketManagerState {
  isConnected: boolean;
  scores: Map<string, InferenceScoreMessage>;
  lastScore: InferenceScoreMessage | null;
  lastAlert: AlertMessage | null;
}

class WebSocketManagerClass {
  private ws: WebSocket | null = null;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private handlers = new Set<MessageHandler>();
  private stateListeners = new Set<() => void>();
  private connectPromise: Promise<void> | null = null;
  private intentionalClose = false;
  private serviceUnavailable = false; // Tracks if service is unreachable (stops auto-reconnect)

  private _state: WebSocketManagerState = {
    isConnected: false,
    scores: new Map(),
    lastScore: null,
    lastAlert: null,
  };

  get state(): WebSocketManagerState {
    return this._state;
  }

  private setState(updates: Partial<WebSocketManagerState>) {
    this._state = { ...this._state, ...updates };
    this.stateListeners.forEach((listener) => listener());
  }

  subscribe = (listener: () => void) => {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  };

  getSnapshot = () => this._state;

  addHandler(handler: MessageHandler) {
    this.handlers.add(handler);
  }

  removeHandler(handler: MessageHandler) {
    this.handlers.delete(handler);
  }

  connect(manual: boolean = false): Promise<void> {
    // If service was marked unavailable and this is not a manual reconnect, skip
    if (this.serviceUnavailable && !manual) {
      return Promise.reject(
        new Error("Service unavailable - manual reconnect required"),
      );
    }

    // Reset service unavailable flag on manual reconnect
    if (manual) {
      this.serviceUnavailable = false;
      this.reconnectAttempts = 0;
    }

    if (this.connectPromise) {
      return this.connectPromise;
    }

    if (this.ws?.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }

    if (this.ws?.readyState === WebSocket.CONNECTING) {
      return new Promise((resolve) => {
        const checkOpen = setInterval(() => {
          if (this.ws?.readyState === WebSocket.OPEN) {
            clearInterval(checkOpen);
            resolve();
          }
        }, 50);
      });
    }

    this.intentionalClose = false;

    this.connectPromise = new Promise((resolve, reject) => {
      try {
        console.log("[WS] Connecting to", WS_URL);
        const ws = new WebSocket(WS_URL);
        this.ws = ws;

        const timeout = setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            ws.close();
            reject(new Error("Connection timeout"));
          }
        }, 10000);

        ws.onopen = () => {
          clearTimeout(timeout);
          console.log("[WS] Connected to ViolenceSense");
          this.reconnectAttempts = 0;
          this.serviceUnavailable = false; // Reset on successful connection
          this.connectPromise = null;
          this.setState({ isConnected: true });
          resolve();
        };

        ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        ws.onclose = (event) => {
          clearTimeout(timeout);
          this.ws = null;
          this.connectPromise = null;
          this.setState({ isConnected: false });

          if (event.code !== 1000) {
            console.log(
              "[WS] Disconnected:",
              event.code,
              event.reason || "(no reason)",
            );
          }

          if (!this.intentionalClose && event.code !== 1000) {
            this.scheduleReconnect();
          }
        };

        ws.onerror = () => { };
      } catch (error) {
        this.connectPromise = null;
        console.error("[WS] Connection error:", error);
        reject(error);
      }
    });

    return this.connectPromise;
  }

  private scheduleReconnect() {
    if (this.reconnectTimeout) return;

    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      this.serviceUnavailable = true; // Stop auto-reconnect, require manual reconnect
      console.warn(
        "[WS] RTSP service unreachable. Click 'Reconnect' in Streams tab when service is available.",
      );
      return;
    }

    const delay = BASE_RECONNECT_DELAY * Math.pow(1.5, this.reconnectAttempts);
    this.reconnectAttempts++;

    // Only log first and last attempt to reduce spam
    if (
      this.reconnectAttempts === 1 ||
      this.reconnectAttempts === MAX_RECONNECT_ATTEMPTS
    ) {
      console.log(
        "[WS] Reconnecting in " +
        Math.round(delay / 1000) +
        "s (attempt " +
        this.reconnectAttempts +
        "/" +
        MAX_RECONNECT_ATTEMPTS +
        ")...",
      );
    }

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect().catch(() => { });
    }, delay);
  }

  disconnect() {
    this.intentionalClose = true;

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close(1000, "Client disconnect");
      this.ws = null;
    }

    this.setState({ isConnected: false });
  }

  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  private handleMessage(rawData: string) {
    try {
      const raw = JSON.parse(rawData);
      const type = raw.type || "unknown";
      const data = raw.data || raw;

      switch (type) {
        case "inference_score": {
          const scoreData = data as InferenceScoreMessage;
          if (scoreData?.stream_id) {
            const newScores = new Map(this._state.scores);
            newScores.set(String(scoreData.stream_id), scoreData);
            this.setState({ lastScore: scoreData, scores: newScores });
          }
          break;
        }

        case "stream_status":
          break;

        case "event_start":
        case "event_end":
        case "violence_alert":
        case "alert": {
          const alertData = data as AlertMessage;
          if (alertData) {
            alertData.type = alertData.type || (type as any);
            this.setState({ lastAlert: alertData });
          }
          break;
        }

        case "ping":
          this.send({ type: "pong" });
          return;

        case "pong":
          return;
      }

      this.handlers.forEach((handler) => {
        try {
          handler(type, data);
        } catch (e) {
          console.error("[WS] Handler error:", e);
        }
      });
    } catch (error) { }
  }
}

declare global {
  var __wsManager: WebSocketManagerClass | undefined;
}

const getManager = (): WebSocketManagerClass => {
  if (typeof window === "undefined") {
    return new WebSocketManagerClass();
  }

  if (!globalThis.__wsManager) {
    globalThis.__wsManager = new WebSocketManagerClass();
  }
  return globalThis.__wsManager;
};

// ============================================================================
// React Hooks
// ============================================================================

export interface UseWebSocketOptions {
  onScore?: (data: InferenceScoreMessage) => void;
  onAlert?: (data: AlertMessage) => void;
  onStreamStatus?: (data: StreamStatusMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  enabled?: boolean;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    onScore,
    onAlert,
    onStreamStatus,
    onConnect,
    onDisconnect,
    enabled = true,
  } = options;

  const callbacksRef = useRef({
    onScore,
    onAlert,
    onStreamStatus,
    onConnect,
    onDisconnect,
  });
  callbacksRef.current = {
    onScore,
    onAlert,
    onStreamStatus,
    onConnect,
    onDisconnect,
  };

  const manager = getManager();

  const state = useSyncExternalStore(
    manager.subscribe,
    manager.getSnapshot,
    manager.getSnapshot,
  );

  const wasConnectedRef = useRef(state.isConnected);

  useEffect(() => {
    if (state.isConnected && !wasConnectedRef.current) {
      callbacksRef.current.onConnect?.();
    } else if (!state.isConnected && wasConnectedRef.current) {
      callbacksRef.current.onDisconnect?.();
    }
    wasConnectedRef.current = state.isConnected;
  }, [state.isConnected]);

  useEffect(() => {
    if (!enabled) return;

    const handler = (type: string, data: any) => {
      switch (type) {
        case "inference_score":
          callbacksRef.current.onScore?.(data);
          break;
        case "stream_status":
          callbacksRef.current.onStreamStatus?.(data);
          break;
        case "event_start":
        case "event_end":
        case "violence_alert":
        case "alert":
          callbacksRef.current.onAlert?.(data);
          break;
      }
    };

    manager.addHandler(handler);
    const connectTimeout = setTimeout(() => {
      manager.connect().catch(() => { });
    }, 100); // 100ms delay survives Strict Mode unmount/remount cycle

    return () => {
      clearTimeout(connectTimeout);
      manager.removeHandler(handler);
    };
  }, [enabled, manager]);

  const send = useCallback(
    (data: any) => {
      manager.send(data);
    },
    [manager],
  );

  const subscribeToStream = useCallback(
    (streamId: string) => {
      manager.send({ type: "subscribe", stream_id: streamId });
    },
    [manager],
  );

  const unsubscribeFromStream = useCallback(
    (streamId: string) => {
      manager.send({ type: "unsubscribe", stream_id: streamId });
    },
    [manager],
  );

  const getStreamScore = useCallback(
    (streamId: string): InferenceScoreMessage | undefined => {
      return state.scores.get(streamId);
    },
    [state.scores],
  );

  const connect = useCallback(() => {
    return manager.connect(true); // manual = true, resets service unavailable flag
  }, [manager]);

  const disconnect = useCallback(() => {
    manager.disconnect();
  }, [manager]);

  return {
    isConnected: state.isConnected,
    lastScore: state.lastScore,
    lastAlert: state.lastAlert,
    scores: state.scores,
    connect,
    disconnect,
    send,
    subscribeToStream,
    unsubscribeFromStream,
    getStreamScore,
  };
}

// ============================================================================
// Specialized Hooks
// ============================================================================

export function useStreamScore(streamId: string) {
  const [score, setScore] = useState<InferenceScoreMessage | null>(null);

  const handleScore = useCallback(
    (data: InferenceScoreMessage) => {
      if (String(data.stream_id) === String(streamId)) {
        setScore(data);
      }
    },
    [streamId],
  );

  const { isConnected, getStreamScore } = useWebSocket({
    onScore: handleScore,
  });

  useEffect(() => {
    const existing = getStreamScore(streamId);
    if (existing) {
      setScore(existing);
    }
  }, [streamId, getStreamScore]);

  return { score, isConnected };
}

export function useAlerts() {
  const [alerts, setAlerts] = useState<AlertMessage[]>([]);
  const [pendingCount, setPendingCount] = useState(0);

  const handleAlert = useCallback((data: AlertMessage) => {
    setAlerts((prev) => [data, ...prev].slice(0, 100));
    setPendingCount((prev) => prev + 1);
  }, []);

  const { isConnected } = useWebSocket({
    onAlert: handleAlert,
  });

  const clearPending = useCallback(() => {
    setPendingCount(0);
  }, []);

  const dismissAlert = useCallback((eventId: string) => {
    setAlerts((prev) => prev.filter((a) => a.event_id !== eventId));
  }, []);

  return {
    alerts,
    pendingCount,
    isConnected,
    clearPending,
    dismissAlert,
  };
}
