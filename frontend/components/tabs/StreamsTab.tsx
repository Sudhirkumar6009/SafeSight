"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { StreamCard } from "@/components/StreamCard";
import { StreamForm } from "@/components/StreamForm";
import {
  Stream,
  StreamCreateRequest,
  StreamStatusMessage,
  InferenceScoreMessage,
} from "@/types";
import { streamService } from "@/services/streamApi";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAppStore } from "@/hooks/useStore";

// View mode types
type ViewMode = "grid" | "list" | "monitor";

// RTSP Service URL for video streams
const RTSP_SERVICE_URL =
  process.env.NEXT_PUBLIC_RTSP_SERVICE_URL || "http://localhost:8080";

// ==================== List View Item Component ====================
interface StreamListItemProps {
  stream: Stream;
  score?: InferenceScoreMessage;
  onStart?: (id: string) => Promise<void>;
  onStop?: (id: string) => Promise<void>;
  onEdit?: (stream: Stream) => void;
  onDelete?: (id: string) => Promise<void>;
}

function StreamListItem({
  stream,
  score,
  onStart,
  onStop,
  onEdit,
  onDelete,
}: StreamListItemProps) {
  const [loading, setLoading] = useState(false);
  const isStreamActive =
    stream.status === "running" || stream.status === "online";

  const violenceScore =
    score?.violence_score ?? stream.last_prediction?.violence_score ?? 0;
  const scoreColor =
    violenceScore > 0.65
      ? "text-red-500"
      : violenceScore > 0.4
        ? "text-yellow-500"
        : "text-green-500";

  const statusColors: Record<string, string> = {
    running: "bg-green-500",
    online: "bg-green-500",
    stopped: "bg-gray-400",
    offline: "bg-gray-400",
    error: "bg-red-500",
    starting: "bg-yellow-500",
    stopping: "bg-yellow-500",
    connecting: "bg-yellow-500",
  };

  const handleStart = async () => {
    if (!onStart) return;
    setLoading(true);
    try {
      await onStart(stream.id);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (!onStop) return;
    setLoading(true);
    try {
      await onStop(stream.id);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!onDelete || !confirm(`Delete stream "${stream.name}"?`)) return;
    setLoading(true);
    try {
      await onDelete(stream.id);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`flex items-center gap-4 p-4 bg-slate-900 border border-slate-800 rounded-lg hover:border-slate-700 transition-colors ${loading ? "opacity-70" : ""}`}
    >
      {/* Status Indicator */}
      <div
        className={`h-3 w-3 rounded-full flex-shrink-0 ${statusColors[stream.status] || "bg-gray-400"}`}
      />

      {/* Stream Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-white truncate">{stream.name}</h3>
          <span className="text-xs text-slate-500 capitalize">
            ({stream.status})
          </span>
        </div>
        <div className="flex items-center gap-4 text-sm text-slate-400 mt-1">
          {stream.location && (
            <span className="truncate">{stream.location}</span>
          )}
          <code
            className="text-xs text-slate-500 truncate max-w-[200px]"
            title={stream.rtsp_url}
          >
            {stream.rtsp_url}
          </code>
        </div>
      </div>

      {/* Violence Score */}
      {isStreamActive && (
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-sm text-slate-400">Score:</span>
          <span className={`text-lg font-bold ${scoreColor}`}>
            {(violenceScore * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {stream.status === "stopped" || stream.status === "error" ? (
          <button
            onClick={handleStart}
            disabled={loading}
            className="px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-sm rounded-md transition-colors disabled:opacity-50"
          >
            {loading ? "..." : "Start"}
          </button>
        ) : isStreamActive ? (
          <button
            onClick={handleStop}
            disabled={loading}
            className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm rounded-md transition-colors disabled:opacity-50"
          >
            {loading ? "..." : "Stop"}
          </button>
        ) : (
          <button
            disabled
            className="px-3 py-1.5 bg-yellow-600 text-white text-sm rounded-md opacity-70"
          >
            {stream.status === "starting"
              ? "Starting..."
              : stream.status === "stopping"
                ? "Stopping..."
                : stream.status}
          </button>
        )}
        {onEdit && (
          <button
            onClick={() => onEdit(stream)}
            disabled={loading}
            className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-md transition-colors disabled:opacity-50"
          >
            Edit
          </button>
        )}
        {onDelete && (
          <button
            onClick={handleDelete}
            disabled={loading || isStreamActive}
            className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-red-400 text-sm rounded-md transition-colors disabled:opacity-50"
          >
            Delete
          </button>
        )}
      </div>
    </motion.div>
  );
}

// ==================== Fullscreen Monitor View Component ====================
interface FullscreenMonitorProps {
  streams: Stream[];
  scores: Map<string, InferenceScoreMessage>;
  onClose: () => void;
  onStart?: (id: string) => Promise<void>;
  onStop?: (id: string) => Promise<void>;
}

function FullscreenMonitor({
  streams,
  scores,
  onClose,
  onStart,
  onStop,
}: FullscreenMonitorProps) {
  const [currentTime, setCurrentTime] = useState(new Date());
  const containerRef = useRef<HTMLDivElement>(null);

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Request fullscreen on mount
  useEffect(() => {
    const enterFullscreen = async () => {
      try {
        if (containerRef.current && document.fullscreenElement === null) {
          await containerRef.current.requestFullscreen();
        }
      } catch (err) {
        console.log("Fullscreen not supported or denied");
      }
    };
    enterFullscreen();

    // Handle escape key and fullscreen exit
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement) {
        onClose();
      }
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, [onClose]);

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (document.fullscreenElement) {
          document.exitFullscreen();
        }
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const count = streams.length;

  // Get optimal grid layout based on stream count
  const getGridLayout = () => {
    if (count === 0) return { cols: 1, rows: 1 };
    if (count === 1) return { cols: 1, rows: 1 };
    if (count === 2) return { cols: 2, rows: 1 };
    if (count <= 4) return { cols: 2, rows: 2 };
    if (count <= 6) return { cols: 3, rows: 2 };
    if (count <= 9) return { cols: 3, rows: 3 };
    if (count <= 12) return { cols: 4, rows: 3 };
    if (count <= 16) return { cols: 4, rows: 4 };
    return { cols: 5, rows: Math.ceil(count / 5) };
  };

  const { cols, rows } = getGridLayout();

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 z-[9999] bg-black flex flex-col"
    >
      {/* Top Header Bar - Security Monitor Style */}
      <div className="flex-shrink-0 bg-gradient-to-b from-slate-900 to-black px-4 py-2 border-b border-slate-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Recording Indicator */}
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-red-600 animate-pulse" />
              <span className="text-red-500 text-sm font-bold tracking-wider">
                REC
              </span>
            </div>
            {/* Brand */}
            <div className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded">
              <svg
                className="w-5 h-5 text-cyan-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              <span className="text-white font-bold text-sm">SafeSight</span>
              <span className="text-slate-500 text-xs">CCTV Monitor</span>
            </div>
          </div>

          {/* Center - Camera Count */}
          <div className="text-slate-400 text-sm">
            <span className="text-white font-semibold">
              {
                streams.filter(
                  (s) => s.status === "running" || s.status === "online",
                ).length
              }
            </span>
            <span className="mx-1">/</span>
            <span>{count}</span>
            <span className="ml-2 text-slate-500">Cameras Online</span>
          </div>

          {/* Right - Time and Close */}
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-white font-mono text-lg tracking-wider">
                {currentTime.toLocaleTimeString()}
              </div>
              <div className="text-slate-500 text-xs">
                {currentTime.toLocaleDateString(undefined, {
                  weekday: "short",
                  year: "numeric",
                  month: "short",
                  day: "numeric",
                })}
              </div>
            </div>
            <button
              onClick={() => {
                if (document.fullscreenElement) {
                  document.exitFullscreen();
                }
                onClose();
              }}
              className="p-2 bg-slate-800 hover:bg-red-600 rounded-lg transition-colors group"
              title="Exit Monitor (ESC)"
            >
              <svg
                className="w-5 h-5 text-slate-400 group-hover:text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Main Grid Area */}
      <div className="flex-1 p-1 overflow-hidden">
        {count === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <svg
                className="w-20 h-20 mx-auto mb-4 text-slate-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              <p className="text-slate-500 text-lg">No Cameras Configured</p>
              <p className="text-slate-600 text-sm mt-2">
                Add streams to view them here
              </p>
            </div>
          </div>
        ) : (
          <div
            className="h-full grid gap-1"
            style={{
              gridTemplateColumns: `repeat(${cols}, 1fr)`,
              gridTemplateRows: `repeat(${rows}, 1fr)`,
            }}
          >
            {streams.map((stream) => (
              <MonitorCell
                key={stream.id}
                stream={stream}
                score={scores.get(String(stream.id))}
                onStart={onStart}
                onStop={onStop}
                compact={count > 4}
              />
            ))}
          </div>
        )}
      </div>

      {/* Bottom Status Bar */}
      <div className="flex-shrink-0 bg-gradient-to-t from-slate-900 to-black px-4 py-2 border-t border-slate-800">
        <div className="flex items-center justify-between text-xs text-slate-500">
          <div className="flex items-center gap-4">
            <span>
              Press{" "}
              <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-400">
                ESC
              </kbd>{" "}
              to exit
            </span>
            <span>|</span>
            <span>
              Layout: {cols}x{rows}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-green-500" />
              <span>Online</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-red-500" />
              <span>Alert</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-slate-500" />
              <span>Offline</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ==================== Monitor Cell Component ====================
interface MonitorCellProps {
  stream: Stream;
  score?: InferenceScoreMessage;
  compact?: boolean;
  onStart?: (id: string) => Promise<void>;
  onStop?: (id: string) => Promise<void>;
}

function MonitorCell({
  stream,
  score,
  compact,
  onStart,
  onStop,
}: MonitorCellProps) {
  const [imageError, setImageError] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);
  const [showControls, setShowControls] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  const isStreamActive =
    stream.status === "running" || stream.status === "online";
  const violenceScore =
    score?.violence_score ?? stream.last_prediction?.violence_score ?? 0;
  const isAlert = violenceScore > 0.65;

  const mjpegUrl = `${RTSP_SERVICE_URL}/api/v1/streams/${stream.id}/mjpeg?overlay=true&_t=${Date.now()}`;

  // Reset states when stream becomes active
  useEffect(() => {
    if (isStreamActive) {
      setImageError(false);
      setImageLoading(true);
    }
  }, [isStreamActive, stream.id]);

  // Poll for MJPEG stream loading
  useEffect(() => {
    if (!isStreamActive || !imageLoading) return;

    const checkLoaded = () => {
      if (imgRef.current && imgRef.current.naturalWidth > 0) {
        setImageLoading(false);
        setImageError(false);
      }
    };

    const pollInterval = setInterval(checkLoaded, 200);
    checkLoaded();

    const timeout = setTimeout(() => {
      if (imgRef.current && imgRef.current.naturalWidth <= 0) {
        setImageLoading(false);
        setImageError(true);
      }
    }, 10000);

    return () => {
      clearInterval(pollInterval);
      clearTimeout(timeout);
    };
  }, [isStreamActive, imageLoading]);

  const handleStart = async () => {
    if (!onStart) return;
    setActionLoading(true);
    try {
      await onStart(stream.id);
    } finally {
      setActionLoading(false);
    }
  };

  const handleStop = async () => {
    if (!onStop) return;
    setActionLoading(true);
    try {
      await onStop(stream.id);
    } finally {
      setActionLoading(false);
    }
  };

  // Get status message for offline streams
  const getOfflineMessage = () => {
    switch (stream.status) {
      case "stopped":
        return "Stream Stopped";
      case "offline":
        return "Camera Offline";
      case "error":
        return "Connection Error";
      case "starting":
        return "Connecting...";
      case "stopping":
        return "Disconnecting...";
      case "connecting":
        return "Connecting...";
      case "reconnecting":
        return "Reconnecting...";
      default:
        return stream.status || "No Signal";
    }
  };

  return (
    <div
      className={`relative bg-slate-950 overflow-hidden border transition-all ${
        isAlert && isStreamActive
          ? "border-red-500 shadow-[0_0_20px_rgba(239,68,68,0.3)]"
          : "border-slate-800 hover:border-slate-700"
      }`}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
    >
      {/* Top Label Bar */}
      <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-2 py-1 bg-gradient-to-b from-black/90 via-black/70 to-transparent">
        <div className="flex items-center gap-2 min-w-0">
          {/* Status dot */}
          <div
            className={`h-2 w-2 rounded-full flex-shrink-0 ${
              isStreamActive
                ? isAlert
                  ? "bg-red-500 animate-pulse"
                  : "bg-green-500"
                : "bg-slate-500"
            }`}
          />
          {/* Channel name */}
          <span
            className={`font-medium truncate ${compact ? "text-[10px]" : "text-xs"} text-white`}
          >
            {stream.name}
          </span>
        </div>

        {/* Violence score badge */}
        {isStreamActive && (
          <span
            className={`flex-shrink-0 px-1.5 py-0.5 rounded text-[10px] font-bold ${
              isAlert
                ? "bg-red-500 text-white animate-pulse"
                : violenceScore > 0.4
                  ? "bg-yellow-500/80 text-black"
                  : "bg-green-500/80 text-black"
            }`}
          >
            {(violenceScore * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {/* Video Feed or Offline Placeholder */}
      {isStreamActive ? (
        <div className="absolute inset-0">
          {/* Loading spinner */}
          {imageLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900 z-10">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-2 border-cyan-500 border-t-transparent mx-auto mb-2" />
                <span className="text-slate-500 text-xs">Connecting...</span>
              </div>
            </div>
          )}

          {/* Video stream */}
          {!imageError ? (
            <img
              ref={imgRef}
              src={mjpegUrl}
              alt={stream.name}
              className={`w-full h-full object-contain transition-opacity duration-300 ${imageLoading ? "opacity-0" : "opacity-100"}`}
              onLoad={() => {
                setImageLoading(false);
                setImageError(false);
              }}
              onError={() => {
                setImageError(true);
                setImageLoading(false);
              }}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
              <div className="text-center">
                <svg
                  className="w-12 h-12 mx-auto mb-2 text-slate-700"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
                <p className="text-slate-500 text-sm">No Signal</p>
                <button
                  onClick={() => {
                    setImageError(false);
                    setImageLoading(true);
                  }}
                  className="mt-2 text-xs text-cyan-500 hover:text-cyan-400"
                >
                  Retry
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Offline State - Shows Logo and Status */
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-900 to-slate-950">
          <div className="text-center px-4">
            {/* SafeSight Logo */}
            <div className="mb-4">
              <svg
                className="w-16 h-16 mx-auto text-slate-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              <p className="text-slate-600 text-xs mt-1 font-medium tracking-wider">
                SafeSight
              </p>
            </div>

            {/* Status Message */}
            <p className="text-slate-400 text-sm font-medium">
              {getOfflineMessage()}
            </p>

            {/* Location if available */}
            {stream.location && (
              <p className="text-slate-600 text-xs mt-1 truncate max-w-[150px] mx-auto">
                {stream.location}
              </p>
            )}

            {/* Start button for stopped streams */}
            {(stream.status === "stopped" || stream.status === "error") &&
              onStart && (
                <button
                  onClick={handleStart}
                  disabled={actionLoading}
                  className="mt-3 px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-xs rounded transition-colors disabled:opacity-50"
                >
                  {actionLoading ? "Starting..." : "Start Stream"}
                </button>
              )}
          </div>
        </div>
      )}

      {/* Violence Alert Effect */}
      {isAlert && isStreamActive && (
        <motion.div
          animate={{ opacity: [0.3, 0.6, 0.3] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          className="absolute inset-0 border-4 border-red-500 pointer-events-none z-10"
        />
      )}

      {/* Hover Controls for active streams */}
      <AnimatePresence>
        {showControls && isStreamActive && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute bottom-0 left-0 right-0 z-20 p-2 bg-gradient-to-t from-black/95 to-transparent"
          >
            <div className="flex items-center justify-center gap-2">
              <button
                onClick={handleStop}
                disabled={actionLoading}
                className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors disabled:opacity-50"
              >
                {actionLoading ? "..." : "Stop"}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Bottom info bar */}
      {!compact && stream.location && isStreamActive && (
        <div className="absolute bottom-0 left-0 right-0 z-10 px-2 py-1 bg-gradient-to-t from-black/80 to-transparent">
          <span className="text-[10px] text-slate-400 truncate block">
            {stream.location}
          </span>
        </div>
      )}
    </div>
  );
}

// ==================== Main Component ====================
export default function StreamsTab() {
  // Use global store for streams to persist across tab switches
  const {
    streams,
    streamsLoaded,
    streamsError,
    setStreams,
    setStreamsError,
    updateStreamStatus,
  } = useAppStore();

  const [loading, setLoading] = useState(!streamsLoaded); // Only show loading if never loaded
  const [formOpen, setFormOpen] = useState(false);
  const [editingStream, setEditingStream] = useState<Stream | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("grid");

  // Handle real-time stream status updates
  const handleStreamStatus = useCallback(
    (statusData: StreamStatusMessage) => {
      updateStreamStatus(String(statusData.stream_id), statusData.status);
    },
    [updateStreamStatus],
  );

  const { scores, isConnected, connect } = useWebSocket({
    onStreamStatus: handleStreamStatus,
  });

  const fetchStreams = useCallback(async () => {
    try {
      const response = await streamService.getStreams();
      if (response.success && response.data) {
        setStreams(response.data);
      } else {
        setStreamsError(response.error || "Failed to fetch streams");
      }
    } catch (err: any) {
      setStreamsError(err.message || "Failed to connect to RTSP service");
    } finally {
      setLoading(false);
    }
  }, [setStreams, setStreamsError]);

  useEffect(() => {
    // Always fetch on mount to get latest data, but don't show loading if we have cached data
    fetchStreams();
    const interval = setInterval(fetchStreams, 10000);
    return () => clearInterval(interval);
  }, [fetchStreams]);

  const handleStart = useCallback(
    async (id: string) => {
      try {
        // Optimistically update status to "starting"
        updateStreamStatus(id, "starting");

        await streamService.startStream(id);

        // Poll more frequently for a few seconds to catch the "running" status
        let pollCount = 0;
        const quickPoll = setInterval(async () => {
          pollCount++;
          await fetchStreams();
          if (pollCount >= 5) {
            clearInterval(quickPoll);
          }
        }, 1000);
      } catch (err: any) {
        console.error("Failed to start stream:", err);
        await fetchStreams();
      }
    },
    [fetchStreams, updateStreamStatus],
  );

  const handleStop = useCallback(
    async (id: string) => {
      try {
        // Optimistically update status to "stopping"
        updateStreamStatus(id, "stopping");

        await streamService.stopStream(id);
        await fetchStreams();
      } catch (err: any) {
        console.error("Failed to stop stream:", err);
        await fetchStreams();
      }
    },
    [fetchStreams, updateStreamStatus],
  );

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await streamService.deleteStream(id);
        await fetchStreams();
      } catch (err: any) {
        console.error("Failed to delete stream:", err);
      }
    },
    [fetchStreams],
  );

  const handleEdit = useCallback((stream: Stream) => {
    setEditingStream(stream);
    setFormOpen(true);
  }, []);

  const handleSubmit = useCallback(
    async (data: StreamCreateRequest) => {
      if (editingStream) {
        await streamService.updateStream(editingStream.id, data);
      } else {
        await streamService.createStream(data);
      }
      setEditingStream(null);
      await fetchStreams();
    },
    [editingStream, fetchStreams],
  );

  const handleCloseForm = useCallback(() => {
    setFormOpen(false);
    setEditingStream(null);
  }, []);

  // Count streams by status
  const runningCount = streams.filter(
    (s) => s.status === "running" || s.status === "online",
  ).length;
  const stoppedCount = streams.filter(
    (s) => s.status === "stopped" || s.status === "offline",
  ).length;
  const errorCount = streams.filter((s) => s.status === "error").length;

  // Show all streams (not just running ones) so users can start stopped streams
  const visibleStreams = streams;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white">Streams</h1>
          <p className="text-slate-400 mt-1">
            Manage RTSP camera feeds for violence detection
          </p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => !isConnected && connect()}
            className={`flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-lg transition-colors ${!isConnected ? "hover:bg-slate-700 cursor-pointer" : ""}`}
            title={
              isConnected
                ? "Connected to real-time updates"
                : "Click to reconnect"
            }
          >
            <div
              className={`h-2 w-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500 animate-pulse"}`}
            />
            <span className="text-sm text-slate-400">
              {isConnected ? "Live" : "Disconnected - Click to reconnect"}
            </span>
          </button>
          <button
            onClick={() => setFormOpen(true)}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg font-medium transition-colors"
          >
            Add Stream
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <p className="text-3xl font-bold text-white">{streams.length}</p>
          <p className="text-sm text-slate-400">Total Streams</p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <p className="text-3xl font-bold text-green-400">{runningCount}</p>
          <p className="text-sm text-slate-400">Running</p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <p className="text-3xl font-bold text-slate-400">{stoppedCount}</p>
          <p className="text-sm text-slate-400">Stopped</p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <p className="text-3xl font-bold text-red-400">{errorCount}</p>
          <p className="text-sm text-slate-400">Errors</p>
        </div>
      </div>

      {/* View Mode Filter */}
      <div className="flex items-center gap-2 mb-6">
        <span className="text-sm text-slate-400 mr-2">View:</span>
        <div className="flex items-center bg-slate-800 rounded-lg p-1 gap-1">
          {/* List View Button */}
          <button
            onClick={() => setViewMode("list")}
            className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
              viewMode === "list"
                ? "bg-cyan-600 text-white shadow-lg"
                : "text-slate-400 hover:text-white hover:bg-slate-700"
            }`}
            title="List View"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 10h16M4 14h16M4 18h16"
              />
            </svg>
            <span className="hidden sm:inline">List</span>
          </button>

          {/* Grid View Button */}
          <button
            onClick={() => setViewMode("grid")}
            className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
              viewMode === "grid"
                ? "bg-cyan-600 text-white shadow-lg"
                : "text-slate-400 hover:text-white hover:bg-slate-700"
            }`}
            title="Grid View"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"
              />
            </svg>
            <span className="hidden sm:inline">Grid</span>
          </button>

          {/* Monitor View Button */}
          <button
            onClick={() => setViewMode("monitor")}
            className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
              viewMode === "monitor"
                ? "bg-cyan-600 text-white shadow-lg"
                : "text-slate-400 hover:text-white hover:bg-slate-700"
            }`}
            title="Monitor View - CCTV Display"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
              />
            </svg>
            <span className="hidden sm:inline">Monitor</span>
          </button>
        </div>
      </div>

      {/* Error */}
      {streamsError && (
        <div className="mb-6 p-4 bg-red-500/20 border border-red-500/30 rounded-lg">
          <div className="flex items-center gap-3">
            <svg
              className="w-5 h-5 text-red-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div>
              <p className="text-red-400 font-medium">Connection Error</p>
              <p className="text-red-300/70 text-sm">{streamsError}</p>
            </div>
            <button
              onClick={fetchStreams}
              className="ml-auto px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-cyan-500" />
        </div>
      )}

      {/* Empty */}
      {!loading && streams.length === 0 && !streamsError && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-20"
        >
          <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-slate-800 flex items-center justify-center">
            <svg
              className="w-10 h-10 text-slate-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">
            No Streams Configured
          </h3>
          <p className="text-slate-400 mb-6 max-w-md mx-auto">
            Add your first RTSP camera stream to start detecting violence in
            real-time.
          </p>
          <button
            onClick={() => setFormOpen(true)}
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg font-medium transition-colors"
          >
            Add Your First Stream
          </button>
        </motion.div>
      )}

      {/* Grid */}
      {!loading && visibleStreams.length > 0 && (
        <AnimatePresence mode="wait">
          {/* List View */}
          {viewMode === "list" && (
            <motion.div
              key="list-view"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="space-y-3"
            >
              {visibleStreams.map((stream) => (
                <StreamListItem
                  key={stream.id}
                  stream={stream}
                  score={scores.get(String(stream.id))}
                  onStart={handleStart}
                  onStop={handleStop}
                  onEdit={handleEdit}
                  onDelete={handleDelete}
                />
              ))}
            </motion.div>
          )}

          {/* Grid View */}
          {viewMode === "grid" && (
            <motion.div
              key="grid-view"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6"
            >
              <AnimatePresence mode="popLayout">
                {visibleStreams.map((stream) => (
                  <StreamCard
                    key={stream.id}
                    stream={stream}
                    score={scores.get(String(stream.id))}
                    onStart={handleStart}
                    onStop={handleStop}
                    onEdit={handleEdit}
                    onDelete={handleDelete}
                  />
                ))}
              </AnimatePresence>
            </motion.div>
          )}

          {/* Monitor View - CCTV Style Display */}
          {viewMode === "monitor" && (
            <FullscreenMonitor
              streams={visibleStreams}
              scores={scores}
              onClose={() => setViewMode("grid")}
              onStart={handleStart}
              onStop={handleStop}
            />
          )}
        </AnimatePresence>
      )}

      <StreamForm
        stream={editingStream}
        isOpen={formOpen}
        onClose={handleCloseForm}
        onSubmit={handleSubmit}
      />
    </div>
  );
}
