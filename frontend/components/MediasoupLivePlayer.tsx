"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Device } from "mediasoup-client";
import {
  MediasoupSignalingClient,
  SignalingEnvelope,
} from "@/services/mediasoupSignaling";

type PlayerStatus =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "failed"
  | "stopped";

interface TransportOptions {
  id: string;
  iceParameters: unknown;
  iceCandidates: unknown[];
  dtlsParameters: unknown;
  sctpParameters?: unknown;
}

interface ConsumeParams {
  id: string;
  producerId: string;
  kind: "audio" | "video";
  rtpParameters: unknown;
  appData?: Record<string, unknown>;
}

interface MediasoupLivePlayerProps {
  initialProducerId?: string;
  signalingUrl?: string;
  roomId?: string;
  stunServers?: string[];
}

const BASE_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 12000;
const MAX_RECONNECT_ATTEMPTS = 8;

const STATUS_LABELS: Record<PlayerStatus, string> = {
  idle: "Idle",
  connecting: "Connecting",
  connected: "Connected",
  reconnecting: "Reconnecting",
  failed: "Failed",
  stopped: "Stopped",
};

const normalizeWsUrl = (rawUrl: string): string => {
  const trimmed = rawUrl.trim();

  if (!trimmed) {
    throw new Error("Missing signaling URL");
  }

  if (typeof window !== "undefined" && window.location.protocol === "https:") {
    if (trimmed.startsWith("ws://")) {
      return `wss://${trimmed.slice(5)}`;
    }
  }

  return trimmed;
};

const asRecord = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object") {
    return {};
  }
  return value as Record<string, unknown>;
};

const parseTransportOptions = (value: unknown): TransportOptions => {
  const root = asRecord(value);
  const params = asRecord(root.params);
  const transport = asRecord(root.transport);

  const candidate =
    (Object.keys(params).length ? params : undefined) ||
    (Object.keys(transport).length ? transport : undefined) ||
    root;

  const options = candidate as unknown as TransportOptions;

  if (!options.id || !options.iceParameters || !options.dtlsParameters) {
    throw new Error("Invalid transport options from signaling server");
  }

  return options;
};

const parseConsumeParams = (value: unknown): ConsumeParams => {
  const root = asRecord(value);
  const params = asRecord(root.params);
  const consumer = asRecord(root.consumer);

  const candidate =
    (Object.keys(params).length ? params : undefined) ||
    (Object.keys(consumer).length ? consumer : undefined) ||
    root;

  const parsed = candidate as unknown as ConsumeParams;

  if (
    !parsed.id ||
    !parsed.producerId ||
    !parsed.kind ||
    !parsed.rtpParameters
  ) {
    throw new Error("Invalid consume response from signaling server");
  }

  return parsed;
};

export default function MediasoupLivePlayer({
  initialProducerId = process.env.NEXT_PUBLIC_MEDIASOUP_PRODUCER_ID || "",
  signalingUrl = process.env.NEXT_PUBLIC_MEDIASOUP_SIGNALING_URL ||
    "wss://your-mediasoup-signaling.example/ws",
  roomId = process.env.NEXT_PUBLIC_MEDIASOUP_ROOM_ID || "default-room",
  stunServers,
}: MediasoupLivePlayerProps) {
  const [status, setStatus] = useState<PlayerStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [producerId, setProducerId] = useState(initialProducerId);
  const [retryAttempt, setRetryAttempt] = useState(0);
  const [lastSignalEvent, setLastSignalEvent] = useState("waiting");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const signalingRef = useRef<MediasoupSignalingClient | null>(null);
  const deviceRef = useRef<Device | null>(null);
  const transportRef = useRef<any>(null);
  const consumerRef = useRef<any>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const intentionallyStoppedRef = useRef(false);
  const isConnectingRef = useRef(false);
  const connectFlowRef = useRef<(() => Promise<void>) | null>(null);

  const parsedSignalingUrl = useMemo(
    () => normalizeWsUrl(signalingUrl),
    [signalingUrl],
  );

  const iceServers = useMemo(() => {
    if (stunServers && stunServers.length > 0) {
      return stunServers.map((url) => ({ urls: url }));
    }

    const envStuns = process.env.NEXT_PUBLIC_MEDIASOUP_STUN_SERVERS;
    const values = envStuns
      ? envStuns
          .split(",")
          .map((server) => server.trim())
          .filter(Boolean)
      : ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"];

    return values.map((url) => ({ urls: url }));
  }, [stunServers]);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const teardownSession = useCallback((resetVideo: boolean) => {
    consumerRef.current?.close?.();
    consumerRef.current = null;

    transportRef.current?.close?.();
    transportRef.current = null;

    signalingRef.current?.close(1000, "teardown");
    signalingRef.current = null;

    deviceRef.current = null;

    if (resetVideo && videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    mediaStreamRef.current = null;
  }, []);

  const queueReconnect = useCallback((reason: string) => {
    if (intentionallyStoppedRef.current) {
      return;
    }

    if (reconnectTimerRef.current) {
      return;
    }

    const nextAttempt = reconnectAttemptsRef.current + 1;
    if (nextAttempt > MAX_RECONNECT_ATTEMPTS) {
      setStatus("failed");
      setError(`Reconnect limit reached. Last reason: ${reason}`);
      return;
    }

    reconnectAttemptsRef.current = nextAttempt;
    setRetryAttempt(nextAttempt);
    setStatus("reconnecting");
    setError(reason);

    const delay = Math.min(
      Math.round(BASE_RECONNECT_DELAY_MS * Math.pow(1.8, nextAttempt - 1)),
      MAX_RECONNECT_DELAY_MS,
    );

    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null;
      void connectFlowRef.current?.();
    }, delay);
  }, []);

  const connectFlow = useCallback(async () => {
    if (isConnectingRef.current) {
      return;
    }

    isConnectingRef.current = true;
    setStatus(reconnectAttemptsRef.current > 0 ? "reconnecting" : "connecting");
    setError(null);

    teardownSession(false);

    try {
      const signaling = new MediasoupSignalingClient(parsedSignalingUrl);
      signalingRef.current = signaling;

      signaling.onClose((event) => {
        if (!intentionallyStoppedRef.current && event.code !== 1000) {
          queueReconnect(`Signaling closed (${event.code})`);
        }
      });

      signaling.onMessage((message: SignalingEnvelope) => {
        if (message.type && typeof message.type === "string") {
          setLastSignalEvent(message.type);
        }
      });

      await signaling.connect();

      const routerCapsResponse = await signaling.requestWithFallback<unknown>(
        [
          "getRouterRtpCapabilities",
          "routerRtpCapabilities",
          "getRtpCapabilities",
        ],
        {
          roomId,
        },
      );

      const routerCapsRecord = asRecord(routerCapsResponse);
      const routerRtpCapabilities =
        routerCapsRecord.rtpCapabilities ||
        routerCapsRecord.data ||
        routerCapsResponse;

      const device = new Device();
      await device.load({
        routerRtpCapabilities: routerRtpCapabilities as any,
      });
      deviceRef.current = device;

      const transportResponse = await signaling.requestWithFallback<unknown>(
        ["createConsumerTransport", "createWebRtcTransport", "createTransport"],
        {
          roomId,
          consuming: true,
          producing: false,
          forceTcp: false,
        },
      );

      const transportOptions = parseTransportOptions(transportResponse);

      const recvTransport = device.createRecvTransport({
        ...transportOptions,
        iceServers,
        iceTransportPolicy: "all",
      } as any);

      transportRef.current = recvTransport;

      recvTransport.on(
        "connect",
        (
          { dtlsParameters }: { dtlsParameters: unknown },
          callback: () => void,
          errback: (error: Error) => void,
        ) => {
          signaling
            .requestWithFallback(
              [
                "connectConsumerTransport",
                "connectWebRtcTransport",
                "connectTransport",
              ],
              {
                roomId,
                transportId: recvTransport.id,
                dtlsParameters,
              },
            )
            .then(() => callback())
            .catch((error) => {
              errback(
                error instanceof Error ? error : new Error(String(error)),
              );
            });
        },
      );

      recvTransport.on("connectionstatechange", (connectionState: string) => {
        if (connectionState === "connected") {
          setStatus("connected");
          setError(null);
          reconnectAttemptsRef.current = 0;
          setRetryAttempt(0);
          return;
        }

        if (
          connectionState === "failed" ||
          connectionState === "disconnected"
        ) {
          queueReconnect(`Transport ${connectionState}`);
        }
      });

      const consumeResponse = await signaling.requestWithFallback<unknown>(
        ["consume", "consumeVideo", "createConsumer"],
        {
          roomId,
          transportId: recvTransport.id,
          producerId: producerId || undefined,
          kind: "video",
          rtpCapabilities: device.rtpCapabilities,
        },
      );

      const consumeParams = parseConsumeParams(consumeResponse);

      const consumer = await recvTransport.consume({
        id: consumeParams.id,
        producerId: consumeParams.producerId,
        kind: consumeParams.kind,
        rtpParameters: consumeParams.rtpParameters as any,
        appData: consumeParams.appData || {},
      });

      consumerRef.current = consumer;

      consumer.on("trackended", () => {
        queueReconnect("Remote track ended");
      });

      consumer.on("transportclose", () => {
        queueReconnect("Consumer transport closed");
      });

      const mediaStream = new MediaStream([consumer.track]);
      mediaStreamRef.current = mediaStream;

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.muted = true;
        videoRef.current.autoplay = true;
        videoRef.current.playsInline = true;

        await videoRef.current.play().catch(() => undefined);
      }

      await signaling
        .requestWithFallback(["resumeConsumer", "resume", "consumerResume"], {
          roomId,
          consumerId: consumer.id,
        })
        .catch(() => undefined);

      setStatus("connected");
      setError(null);
      reconnectAttemptsRef.current = 0;
      setRetryAttempt(0);
    } catch (connectionError) {
      const message =
        connectionError instanceof Error
          ? connectionError.message
          : "Failed to connect mediasoup player";

      setStatus("failed");
      setError(message);
      queueReconnect(message);
    } finally {
      isConnectingRef.current = false;
    }
  }, [
    iceServers,
    parsedSignalingUrl,
    producerId,
    queueReconnect,
    roomId,
    teardownSession,
  ]);

  connectFlowRef.current = connectFlow;

  const start = useCallback(async () => {
    intentionallyStoppedRef.current = false;
    reconnectAttemptsRef.current = 0;
    setRetryAttempt(0);
    clearReconnectTimer();
    await connectFlow();
  }, [clearReconnectTimer, connectFlow]);

  const stop = useCallback(() => {
    intentionallyStoppedRef.current = true;
    clearReconnectTimer();
    teardownSession(true);
    setStatus("stopped");
  }, [clearReconnectTimer, teardownSession]);

  useEffect(() => {
    void start();

    return () => {
      intentionallyStoppedRef.current = true;
      clearReconnectTimer();
      teardownSession(true);
    };
  }, [clearReconnectTimer, start, teardownSession]);

  const statusColor =
    status === "connected"
      ? "bg-green-500"
      : status === "failed"
        ? "bg-red-500"
        : status === "reconnecting"
          ? "bg-amber-500"
          : "bg-blue-500";

  return (
    <section className="rounded-2xl border border-cyan-500/20 bg-slate-900/80 p-5 shadow-2xl">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm text-slate-300">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${statusColor}`}
            />
            <span>{STATUS_LABELS[status]}</span>
            {retryAttempt > 0 && (
              <span className="text-slate-400">Retry #{retryAttempt}</span>
            )}
          </div>
          <p className="text-xs text-slate-400">
            Last signaling event: {lastSignalEvent}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => void start()}
            className="rounded-lg bg-cyan-500 px-3 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400"
          >
            Connect
          </button>
          <button
            onClick={stop}
            className="rounded-lg border border-slate-600 px-3 py-2 text-sm font-semibold text-slate-200 transition hover:border-slate-400"
          >
            Stop
          </button>
        </div>
      </div>

      <div className="mb-3 grid gap-2 md:grid-cols-2">
        <label className="text-sm text-slate-300">
          Producer ID (optional)
          <input
            value={producerId}
            onChange={(event) => setProducerId(event.target.value)}
            placeholder="producer-id-from-server"
            className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none ring-cyan-400 transition focus:ring"
          />
        </label>

        <label className="text-sm text-slate-300">
          Signaling URL
          <input
            value={parsedSignalingUrl}
            readOnly
            className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-300"
          />
        </label>
      </div>

      <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-black">
        <video
          ref={videoRef}
          className="aspect-video w-full bg-black object-cover"
          autoPlay
          muted
          playsInline
          controls
        />
        {status !== "connected" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <p className="rounded-lg bg-slate-900/85 px-4 py-2 text-sm text-slate-200">
              {STATUS_LABELS[status]}...
            </p>
          </div>
        )}
      </div>

      <div className="mt-3 text-xs text-slate-400">
        <p>
          Transport uses DTLS + SRTP with WebRTC for ultra-low-latency playback.
        </p>
        {error && <p className="mt-1 text-red-400">{error}</p>}
      </div>
    </section>
  );
}
