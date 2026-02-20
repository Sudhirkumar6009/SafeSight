/**
 * ViolenceSense - Stream Card Component
 * ======================================
 * Displays stream status with real-time violence score indicator and video preview.
 */

"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Stream, InferenceScoreMessage } from "@/types";

// RTSP Service URL for video streams
const RTSP_SERVICE_URL =
  process.env.NEXT_PUBLIC_RTSP_SERVICE_URL || "http://localhost:8080";

interface StreamCardProps {
  stream: Stream;
  score?: InferenceScoreMessage;
  onStart?: (id: string) => Promise<void>;
  onStop?: (id: string) => Promise<void>;
  onEdit?: (stream: Stream) => void;
  onDelete?: (id: string) => Promise<void>;
  onClick?: (stream: Stream) => void;
}

export function StreamCard({
  stream,
  score,
  onStart,
  onStop,
  onEdit,
  onDelete,
  onClick,
}: StreamCardProps) {
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<"card" | "video">("card");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);
  const [streamKey, setStreamKey] = useState(Date.now()); // Cache-busting key
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const connectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Helper to check if stream is actively running
  const isStreamActive =
    stream.status === "running" || stream.status === "online";

  // Handle fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    if (isStreamActive) {
      setImageError(false);
      setImageLoading(true);
      setStreamKey(Date.now());
    }

    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
        retryTimeoutRef.current = null;
      }
      if (connectTimeoutRef.current) {
        clearTimeout(connectTimeoutRef.current);
        connectTimeoutRef.current = null;
      }
    };
  }, [stream.status, isStreamActive]);

  // When switching into video mode, force a fresh MJPEG connection attempt.
  useEffect(() => {
    if (viewMode !== "video" || !isStreamActive) return;
    setImageError(false);
    setImageLoading(true);
    setStreamKey(Date.now());
  }, [viewMode, isStreamActive, stream.id]);

  // Poll for MJPEG stream loading - onLoad doesn't reliably fire for MJPEG streams
  useEffect(() => {
    if (!isStreamActive || viewMode !== "video" || !imageLoading) return;

    const checkLoaded = () => {
      if (imgRef.current && imgRef.current.naturalWidth > 0) {
        setImageLoading(false);
        setImageError(false);
      }
    };

    // Check immediately and then poll every 100ms for faster detection
    const pollInterval = setInterval(checkLoaded, 100);
    checkLoaded();

    // If we still don't have a decoded frame after a reasonable time, surface an error
    // instead of hiding the loader and leaving a black box.
    connectTimeoutRef.current = setTimeout(() => {
      if (imgRef.current && imgRef.current.naturalWidth <= 0) {
        setImageLoading(false);
        setImageError(true);
      }
    }, 8000); // 8 second timeout (reduced from 12s)

    return () => {
      clearInterval(pollInterval);
      if (connectTimeoutRef.current) {
        clearTimeout(connectTimeoutRef.current);
        connectTimeoutRef.current = null;
      }
    };
  }, [isStreamActive, viewMode, imageLoading, imageError, streamKey]);

  // Handle image load success (may not fire for MJPEG, but keep as backup)
  const handleImageLoad = useCallback(() => {
    setImageLoading(false);
    setImageError(false);
    // Clear any pending retry
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  }, []);

  // Handle image error with auto-retry
  const handleImageError = useCallback(() => {
    // For MJPEG, browsers may emit an error during initial connect.
    // Retry once quickly, then fall back to an explicit error state.
    if (!retryTimeoutRef.current) {
      retryTimeoutRef.current = setTimeout(() => {
        retryTimeoutRef.current = null;
        if (imgRef.current && imgRef.current.naturalWidth > 0) {
          setImageLoading(false);
          setImageError(false);
          return;
        }

        // Retry once with a fresh cache-busting key
        if (isStreamActive) {
          setStreamKey(Date.now());
          setImageLoading(true);
          return;
        }

        setImageLoading(false);
        setImageError(true);
      }, 1500);
    }
  }, [isStreamActive]);

  const handleStart = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!onStart) return;
      setLoading(true);
      try {
        await onStart(stream.id);
      } finally {
        setLoading(false);
      }
    },
    [stream.id, onStart],
  );

  const handleStop = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!onStop) return;
      setLoading(true);
      try {
        await onStop(stream.id);
      } finally {
        setLoading(false);
      }
    },
    [stream.id, onStop],
  );

  const handleEdit = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onEdit?.(stream);
    },
    [stream, onEdit],
  );

  const handleDelete = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!onDelete) return;
      if (confirm(`Delete stream "${stream.name}"?`)) {
        setLoading(true);
        try {
          await onDelete(stream.id);
        } finally {
          setLoading(false);
        }
      }
    },
    [stream.id, stream.name, onDelete],
  );

  const toggleViewMode = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setViewMode((prev) => {
      const next = prev === "card" ? "video" : "card";
      if (next === "video") {
        setImageError(false);
        setImageLoading(true);
        setStreamKey(Date.now());
      }
      return next;
    });
  }, []);

  const toggleFullscreen = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!containerRef.current) return;

    try {
      if (!document.fullscreenElement) {
        await containerRef.current.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error("Fullscreen error:", err);
    }
  }, []);

  // Status colors
  const statusColors: Record<string, string> = {
    running: "bg-green-500",
    online: "bg-green-500",
    connected: "bg-green-500", // Backward compatibility
    stopped: "bg-gray-400",
    offline: "bg-gray-400",
    disconnected: "bg-gray-400", // Backward compatibility
    error: "bg-red-500",
    starting: "bg-yellow-500",
    stopping: "bg-yellow-500",
    connecting: "bg-yellow-500",
    reconnecting: "bg-yellow-500", // Reconnecting status
  };

  // Violence score indicator - use WebSocket score or fallback to API's last_prediction
  const violenceScore =
    score?.violence_score ?? stream.last_prediction?.violence_score ?? 0;
  const scoreColor =
    violenceScore > 0.65
      ? "text-red-500"
      : violenceScore > 0.4
        ? "text-yellow-500"
        : "text-green-500";

  const scoreBarColor =
    violenceScore > 0.65
      ? "bg-red-500"
      : violenceScore > 0.4
        ? "bg-yellow-500"
        : "bg-green-500";

  // MJPEG stream URL with cache-busting and score overlay
  const mjpegUrl = `${RTSP_SERVICE_URL}/api/v1/streams/${stream.id}/mjpeg?overlay=true&_t=${streamKey}`;
  const mjpegUrlNoOverlay = `${RTSP_SERVICE_URL}/api/v1/streams/${stream.id}/mjpeg?overlay=false&_t=${streamKey}`;
  const snapshotUrl = `${RTSP_SERVICE_URL}/api/v1/streams/${stream.id}/snapshot?_t=${streamKey}`;

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`
        bg-gray-800 rounded-lg border border-gray-700 overflow-hidden
        hover:border-gray-600 transition-colors
        ${loading ? "opacity-70" : ""}
        ${isFullscreen ? "fixed inset-0 z-50 rounded-none border-0 flex flex-col" : ""}
      `}
    >
      {/* Header */}
      <div
        className={`p-4 border-b border-gray-700 ${isFullscreen ? "flex-shrink-0" : ""}`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className={`h-3 w-3 rounded-full ${statusColors[stream.status] || "bg-gray-400"}`}
            />
            <h3
              className="font-semibold text-white truncate max-w-[180px]"
              title={stream.name}
            >
              {stream.name}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            {/* Violence Score Badge - Always visible when stream is active */}
            {isStreamActive && (
              <div
                className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                  violenceScore > 0.65
                    ? "bg-red-500/20 text-red-400 animate-pulse"
                    : violenceScore > 0.4
                      ? "bg-yellow-500/20 text-yellow-400"
                      : "bg-green-500/20 text-green-400"
                }`}
                title={`Violence Score: ${(violenceScore * 100).toFixed(1)}%`}
              >
                {(violenceScore * 100).toFixed(0)}%
              </div>
            )}
            {/* View Toggle Button */}
            {isStreamActive && (
              <button
                onClick={toggleViewMode}
                className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-300 
                           hover:text-white transition-colors"
                title={viewMode === "card" ? "Show Video" : "Show Card"}
              >
                {viewMode === "card" ? (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                ) : (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 6h16M4 10h16M4 14h16M4 18h16"
                    />
                  </svg>
                )}
              </button>
            )}
            {/* Fullscreen Button (only in video mode) */}
            {viewMode === "video" && isStreamActive && (
              <button
                onClick={toggleFullscreen}
                className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-300 
                           hover:text-white transition-colors"
                title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
              >
                {isFullscreen ? (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                ) : (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
                    />
                  </svg>
                )}
              </button>
            )}
            <span className="text-xs text-gray-400 capitalize">
              {stream.status}
            </span>
          </div>
        </div>
        {stream.location && (
          <p className="text-sm text-gray-400 mt-1 truncate">
            {stream.location}
          </p>
        )}
      </div>

      {/* Video Preview or Card Content */}
      <AnimatePresence mode="wait">
        {viewMode === "video" && isStreamActive ? (
          <motion.div
            key="video"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={`relative bg-black ${isFullscreen ? "flex-1" : "aspect-video"}`}
          >
            {!imageError ? (
              <>
                {/* Loading indicator */}
                {imageLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-10 w-10 border-2 border-blue-500 border-t-transparent mx-auto mb-2" />
                      <p className="text-sm text-gray-400">Loading stream...</p>
                    </div>
                  </div>
                )}
                <img
                  ref={imgRef}
                  src={mjpegUrl}
                  alt={`${stream.name} live feed`}
                  loading="eager"
                  decoding="async"
                  className={`w-full h-full object-contain ${imageLoading ? "opacity-0" : "opacity-100"} transition-opacity duration-100`}
                  onLoad={handleImageLoad}
                  onError={handleImageError}
                />
              </>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                <div className="text-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-12 w-12 mx-auto mb-2"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                  <p className="text-sm">Video unavailable</p>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setImageError(false);
                      setImageLoading(true);
                      setStreamKey(Date.now()); // Force fresh load
                    }}
                    className="mt-2 text-xs text-blue-400 hover:text-blue-300"
                  >
                    Retry
                  </button>
                </div>
              </div>
            )}

            {/* Violence Score Overlay */}
            {score && (
              <div className="absolute top-2 right-2 bg-black/70 rounded px-2 py-1">
                <span className={`text-sm font-bold ${scoreColor}`}>
                  {(violenceScore * 100).toFixed(0)}%
                </span>
              </div>
            )}

            {/* Violence Alert Overlay */}
            {violenceScore > 0.65 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ repeat: Infinity, duration: 1 }}
                className="absolute inset-0 border-4 border-red-500 pointer-events-none"
              />
            )}
          </motion.div>
        ) : (
          <motion.div
            key="card"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Violence Score */}
            {isStreamActive && (
              <div className="p-4 border-b border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Violence Score</span>
                  <span className={`text-lg font-bold ${scoreColor}`}>
                    {(violenceScore * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full ${scoreBarColor}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${violenceScore * 100}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                {(score || stream.last_prediction) && (
                  <p className="text-xs text-gray-500 mt-1">
                    {score?.fps?.toFixed(1)
                      ? `${score.fps.toFixed(1)} FPS • `
                      : ""}
                    {new Date(
                      score?.timestamp ||
                        stream.last_prediction?.timestamp ||
                        Date.now(),
                    ).toLocaleTimeString()}
                  </p>
                )}
              </div>
            )}

            {/* Stream Info */}
            <div className="p-4 border-b border-gray-700 space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <span className="text-gray-500">URL:</span>
                <code
                  className="text-xs text-gray-400 truncate flex-1"
                  title={stream.rtsp_url}
                >
                  {stream.rtsp_url}
                </code>
              </div>
              {stream.inference_enabled !== undefined && (
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-gray-500">Inference:</span>
                  <span
                    className={
                      stream.inference_enabled
                        ? "text-green-400"
                        : "text-gray-400"
                    }
                  >
                    {stream.inference_enabled ? "Enabled" : "Disabled"}
                  </span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Actions */}
      <div
        className={`p-4 flex items-center gap-2 ${isFullscreen ? "flex-shrink-0" : ""}`}
      >
        {stream.status === "stopped" || stream.status === "error" ? (
          <button
            onClick={handleStart}
            disabled={loading}
            className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white 
                       rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {loading ? "Starting..." : "Start"}
          </button>
        ) : isStreamActive ? (
          <button
            onClick={handleStop}
            disabled={loading}
            className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white 
                       rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {loading ? "Stopping..." : "Stop"}
          </button>
        ) : stream.status === "connecting" || stream.status === "starting" ? (
          <button
            disabled
            className="flex-1 px-3 py-2 bg-yellow-600 text-white rounded-lg text-sm font-medium opacity-70"
          >
            Connecting...
          </button>
        ) : (
          <button
            disabled
            className="flex-1 px-3 py-2 bg-gray-600 text-white rounded-lg text-sm font-medium opacity-50"
          >
            {stream.status === "stopping"
              ? "Stopping..."
              : stream.status || "Unknown"}
          </button>
        )}

        {onEdit && (
          <button
            onClick={handleEdit}
            disabled={loading}
            className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white 
                       rounded-lg text-sm transition-colors disabled:opacity-50"
          >
            Edit
          </button>
        )}

        {onDelete && (
          <button
            onClick={handleDelete}
            disabled={loading || isStreamActive}
            className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-red-400 
                       rounded-lg text-sm transition-colors disabled:opacity-50"
          >
            Delete
          </button>
        )}
      </div>
    </motion.div>
  );
}

// Compact version for dashboard
export function StreamCardCompact({
  stream,
  score,
  onClick,
}: Pick<StreamCardProps, "stream" | "score" | "onClick">) {
  const violenceScore = score?.violence_score ?? 0;
  const isAlert = violenceScore > 0.65;
  const isStreamActive =
    stream.status === "running" || stream.status === "online";

  const statusColors: Record<string, string> = {
    running: "bg-green-500",
    online: "bg-green-500",
    stopped: "bg-gray-400",
    offline: "bg-gray-400",
    error: "bg-red-500",
    starting: "bg-yellow-500",
    stopping: "bg-yellow-500",
    connecting: "bg-yellow-500",
    reconnecting: "bg-yellow-500",
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={`
        flex items-center gap-3 p-3 bg-gray-800 rounded-lg border cursor-pointer
        ${isAlert ? "border-red-500 animate-pulse" : "border-gray-700 hover:border-gray-600"}
      `}
      onClick={() => onClick?.(stream)}
    >
      <div
        className={`h-2.5 w-2.5 rounded-full ${statusColors[stream.status] || "bg-gray-400"}`}
      />
      <span className="font-medium text-white flex-1 truncate">
        {stream.name}
      </span>
      {isStreamActive && (
        <span
          className={`text-sm font-semibold ${isAlert ? "text-red-500" : "text-gray-400"}`}
        >
          {(violenceScore * 100).toFixed(0)}%
        </span>
      )}
    </motion.div>
  );
}
