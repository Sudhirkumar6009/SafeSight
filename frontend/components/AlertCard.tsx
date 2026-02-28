/**
 * SafeSight - Alert Card Component
 * =====================================
 * Displays violence event alert with actions.
 */

"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ViolenceEvent } from "@/types";

interface AlertCardProps {
  event: ViolenceEvent;
  onConfirm?: (id: string) => Promise<void>;
  onDismiss?: (id: string) => Promise<void>;
  onViewClip?: (event: ViolenceEvent) => void;
  onClick?: (event: ViolenceEvent) => void;
  compact?: boolean;
}

export function AlertCard({
  event,
  onConfirm,
  onDismiss,
  onViewClip,
  onClick,
  compact = false,
}: AlertCardProps) {
  const [loading, setLoading] = useState(false);

  // Helper to check if status is pending (handles both cases)
  const isPending =
    event.status === "PENDING" ||
    event.status === "pending" ||
    event.status === "NEW";

  const handleConfirm = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!onConfirm) return;
      setLoading(true);
      try {
        await onConfirm(event.id);
      } finally {
        setLoading(false);
      }
    },
    [event.id, onConfirm],
  );

  const handleDismiss = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!onDismiss) return;
      setLoading(true);
      try {
        await onDismiss(event.id);
      } finally {
        setLoading(false);
      }
    },
    [event.id, onDismiss],
  );

  const handleViewClip = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onViewClip?.(event);
    },
    [event, onViewClip],
  );

  // Severity colors
  const severityColors = {
    low: "border-yellow-500 bg-yellow-500/10",
    medium: "border-orange-500 bg-orange-500/10",
    high: "border-red-500 bg-red-500/10",
    critical: "border-red-600 bg-red-600/20",
  };

  const severityBadgeColors = {
    low: "bg-yellow-500/20 text-yellow-400",
    medium: "bg-orange-500/20 text-orange-400",
    high: "bg-red-500/20 text-red-400",
    critical: "bg-red-600/30 text-red-300",
  };

  // Status badge
  const statusColors: Record<string, string> = {
    new: "bg-blue-500/20 text-blue-400",
    pending: "bg-yellow-500/20 text-yellow-400",
    confirmed: "bg-green-500/20 text-green-400",
    dismissed: "bg-gray-500/20 text-gray-400",
    reviewed: "bg-blue-500/20 text-blue-400",
    auto_dismissed: "bg-gray-500/20 text-gray-400",
  };

  const formatTimestamp = (ts: string) => {
    const date = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return "--";
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
  };

  if (compact) {
    return (
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 20 }}
        className={`
          flex items-center gap-3 p-3 rounded-lg border cursor-pointer
          ${severityColors[event.severity] || "border-gray-600"}
          hover:bg-white/5 transition-colors
        `}
        onClick={() => onClick?.(event)}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-white truncate">
              {event.stream_name || `Stream ${event.stream_id.slice(0, 8)}`}
            </span>
            <span
              className={`text-xs px-1.5 py-0.5 rounded ${severityBadgeColors[event.severity]}`}
            >
              {event.severity}
            </span>
          </div>
          <p className="text-sm text-gray-400">
            {formatTimestamp(event.started_at)}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold text-red-400">
            {(event.max_score * 100).toFixed(0)}%
          </span>
          {isPending && (
            <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={`
        rounded-lg border overflow-hidden
        ${severityColors[event.severity] || "border-gray-600 bg-gray-800"}
        ${isPending ? "animate-pulse" : ""}
        ${loading ? "opacity-70" : ""}
      `}
    >
      {/* Header */}
      <div
        className="p-4 cursor-pointer hover:bg-white/5 transition-colors"
        onClick={() => onClick?.(event)}
      >
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-semibold text-white">
                {event.stream_name || `Stream ${event.stream_id.slice(0, 8)}`}
              </h3>
              <span
                className={`text-xs px-2 py-0.5 rounded-full ${severityBadgeColors[event.severity]}`}
              >
                {event.severity.toUpperCase()}
              </span>
              <span
                className={`text-xs px-2 py-0.5 rounded-full ${statusColors[event.status]}`}
              >
                {event.status}
              </span>
            </div>
            {event.notes && (
              <p className="text-sm text-gray-400 mt-1">{event.notes}</p>
            )}
          </div>
          <div className="text-right ml-4">
            <p className="text-2xl font-bold text-red-400">
              {(event.max_score * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-gray-500">max score</p>
          </div>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-4 mt-3 text-sm text-gray-400">
          <span>{formatTimestamp(event.started_at)}</span>
          <span>•</span>
          <span>Duration: {formatDuration(event.duration_seconds)}</span>
          <span>•</span>
          <span>Frames: {event.frame_count || "--"}</span>
        </div>
      </div>

      {/* Actions */}
      {isPending && (onConfirm || onDismiss || onViewClip) && (
        <div className="p-4 border-t border-white/10 flex items-center gap-2">
          {event.clip_path && onViewClip && (
            <button
              onClick={handleViewClip}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white 
                         rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              View Clip
            </button>
          )}
          {onConfirm && (
            <button
              onClick={handleConfirm}
              disabled={loading}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white 
                         rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              Confirm
            </button>
          )}
          {onDismiss && (
            <button
              onClick={handleDismiss}
              disabled={loading}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white 
                         rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              Dismiss
            </button>
          )}
        </div>
      )}

      {/* View clip for non-pending */}
      {!isPending && event.clip_path && onViewClip && (
        <div className="p-4 border-t border-white/10">
          <button
            onClick={handleViewClip}
            disabled={loading}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white 
                       rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            View Clip
          </button>
        </div>
      )}
    </motion.div>
  );
}

// Live alert banner (for new incoming alerts)
export function LiveAlertBanner({
  event,
  onConfirm,
  onDismiss,
  onClose,
}: {
  event: ViolenceEvent;
  onConfirm?: () => void;
  onDismiss?: () => void;
  onClose?: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -100 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -100 }}
      className="fixed top-4 left-1/2 -translate-x-1/2 z-50 w-full max-w-lg
                 bg-red-600 rounded-lg shadow-2xl overflow-hidden"
    >
      <div className="p-4">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0">
            <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-bold text-white">Violence Detected!</h3>
            <p className="text-red-100">
              {event.stream_name || "Unknown stream"} - Score:{" "}
              {(event.max_score * 100).toFixed(0)}%
            </p>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-white/70 hover:text-white"
            >
              <svg
                className="w-5 h-5"
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
          )}
        </div>
        <div className="flex items-center gap-2 mt-3">
          {onConfirm && (
            <button
              onClick={onConfirm}
              className="flex-1 px-3 py-1.5 bg-white text-red-600 rounded font-medium text-sm hover:bg-red-50"
            >
              Confirm
            </button>
          )}
          {onDismiss && (
            <button
              onClick={onDismiss}
              className="flex-1 px-3 py-1.5 bg-red-700 text-white rounded font-medium text-sm hover:bg-red-800"
            >
              Dismiss
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}
