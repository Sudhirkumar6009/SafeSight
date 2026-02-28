/**
 * SafeSight - Event Modal Component
 * =====================================
 * Modal for viewing event details with video clip playback.
 */

"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ViolenceEvent } from "@/types";
import { streamService } from "@/services/streamApi";

interface EventModalProps {
  event: ViolenceEvent | null;
  isOpen: boolean;
  onClose: () => void;
  onConfirm?: (id: string) => Promise<void>;
  onDismiss?: (id: string) => Promise<void>;
}

export function EventModal({
  event,
  isOpen,
  onClose,
  onConfirm,
  onDismiss,
}: EventModalProps) {
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"details" | "clip">("details");

  // Reset tab when modal opens
  useEffect(() => {
    if (isOpen) {
      setActiveTab(event?.clip_path ? "clip" : "details");
    }
  }, [isOpen, event?.clip_path]);

  const handleConfirm = useCallback(async () => {
    if (!event || !onConfirm) return;
    setLoading(true);
    try {
      await onConfirm(event.id);
      onClose();
    } finally {
      setLoading(false);
    }
  }, [event, onConfirm, onClose]);

  const handleDismiss = useCallback(async () => {
    if (!event || !onDismiss) return;
    setLoading(true);
    try {
      await onDismiss(event.id);
      onClose();
    } finally {
      setLoading(false);
    }
  }, [event, onDismiss, onClose]);

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      return () => document.removeEventListener("keydown", handleEscape);
    }
  }, [isOpen, onClose]);

  if (!event) return null;

  const formatDate = (ts: string) => new Date(ts).toLocaleString();

  const formatDuration = (seconds?: number) => {
    if (!seconds) return "--";
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs.toFixed(0)}s` : `${secs.toFixed(1)}s`;
  };

  const severityColors = {
    low: "text-yellow-400 bg-yellow-400/20",
    medium: "text-orange-400 bg-orange-400/20",
    high: "text-red-400 bg-red-400/20",
    critical: "text-red-300 bg-red-500/30",
  };

  const statusColors: Record<string, string> = {
    new: "text-blue-400 bg-blue-400/20",
    pending: "text-yellow-400 bg-yellow-400/20",
    confirmed: "text-green-400 bg-green-400/20",
    dismissed: "text-gray-400 bg-gray-400/20",
    reviewed: "text-blue-400 bg-blue-400/20",
    auto_dismissed: "text-gray-400 bg-gray-400/20",
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
            onClick={onClose}
          />

          {/* Modal Container - Centered */}
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full max-w-3xl max-h-[85vh] bg-gray-900 rounded-xl shadow-2xl flex flex-col overflow-hidden"
            >
              {/* Header */}
              <div className="p-4 border-b border-gray-700 flex items-start justify-between">
                <div>
                  <h2 className="text-xl font-bold text-white">
                    Event Details
                  </h2>
                  <p className="text-sm text-gray-400">
                    {event.stream_name ||
                      `Stream ${event.stream_id.slice(0, 8)}`}
                  </p>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                >
                  <svg
                    className="w-5 h-5 text-gray-400"
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

              {/* Tabs */}
              <div className="flex border-b border-gray-700">
                <button
                  onClick={() => setActiveTab("details")}
                  className={`flex-1 px-4 py-3 text-sm font-medium transition-colors
                  ${activeTab === "details"
                      ? "text-white border-b-2 border-blue-500"
                      : "text-gray-400 hover:text-gray-300"
                    }`}
                >
                  Details
                </button>
                {event.clip_path && (
                  <button
                    onClick={() => setActiveTab("clip")}
                    className={`flex-1 px-4 py-3 text-sm font-medium transition-colors
                    ${activeTab === "clip"
                        ? "text-white border-b-2 border-blue-500"
                        : "text-gray-400 hover:text-gray-300"
                      }`}
                  >
                    Video Clip
                  </button>
                )}
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-4">
                {activeTab === "details" && (
                  <div className="space-y-6">
                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-gray-800 rounded-lg p-4 text-center">
                        <p className="text-3xl font-bold text-red-400">
                          {(event.max_score * 100).toFixed(0)}%
                        </p>
                        <p className="text-xs text-gray-500 mt-1">Max Score</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-4 text-center">
                        <p className="text-3xl font-bold text-white">
                          {(event.avg_score! * 100).toFixed(0)}%
                        </p>
                        <p className="text-xs text-gray-500 mt-1">Avg Score</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-4 text-center">
                        <p className="text-3xl font-bold text-white">
                          {formatDuration(event.duration_seconds)}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">Duration</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-4 text-center">
                        <p className="text-3xl font-bold text-white">
                          {event.frame_count || "--"}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">Frames</p>
                      </div>
                    </div>

                    {/* Info List */}
                    <div className="bg-gray-800 rounded-lg divide-y divide-gray-700">
                      <div className="p-4 flex justify-between items-center">
                        <span className="text-gray-400">Status</span>
                        <span
                          className={`px-3 py-1 rounded-full text-sm ${statusColors[event.status]}`}
                        >
                          {event.status.charAt(0).toUpperCase() +
                            event.status.slice(1)}
                        </span>
                      </div>
                      <div className="p-4 flex justify-between items-center">
                        <span className="text-gray-400">Severity</span>
                        <span
                          className={`px-3 py-1 rounded-full text-sm ${severityColors[event.severity]}`}
                        >
                          {event.severity.toUpperCase()}
                        </span>
                      </div>
                      <div className="p-4 flex justify-between items-center">
                        <span className="text-gray-400">Started At</span>
                        <span className="text-white">
                          {formatDate(event.started_at)}
                        </span>
                      </div>
                      {event.ended_at && (
                        <div className="p-4 flex justify-between items-center">
                          <span className="text-gray-400">Ended At</span>
                          <span className="text-white">
                            {formatDate(event.ended_at)}
                          </span>
                        </div>
                      )}
                      <div className="p-4 flex justify-between items-center">
                        <span className="text-gray-400">Stream ID</span>
                        <code className="text-gray-300 text-sm">
                          {event.stream_id}
                        </code>
                      </div>
                      <div className="p-4 flex justify-between items-center">
                        <span className="text-gray-400">Event ID</span>
                        <code className="text-gray-300 text-sm">
                          {event.id}
                        </code>
                      </div>
                    </div>

                    {/* Notes */}
                    {event.notes && (
                      <div className="bg-gray-800 rounded-lg p-4">
                        <h4 className="text-sm text-gray-400 mb-2">Notes</h4>
                        <p className="text-white">{event.notes}</p>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === "clip" && event.clip_path && (
                  <div className="space-y-4">
                    {/* Video Player */}
                    <div className="bg-black rounded-lg overflow-hidden aspect-video">
                      <video
                        src={streamService.getClipUrl(event.clip_path)}
                        controls
                        autoPlay
                        className="w-full h-full"
                        playsInline
                      >
                        Your browser does not support the video tag.
                      </video>
                    </div>

                    {/* Clip info */}
                    <div className="bg-gray-800 rounded-lg p-4">
                      <h4 className="text-sm text-gray-400 mb-2">
                        Clip Information
                      </h4>
                      <p className="text-white text-sm break-all">
                        {event.clip_path}
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Actions */}
              {event.status === "pending" && (onConfirm || onDismiss) && (
                <div className="p-4 border-t border-gray-700 flex items-center justify-end gap-3">
                  {onDismiss && (
                    <button
                      onClick={handleDismiss}
                      disabled={loading}
                      className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white 
                               rounded-lg font-medium transition-colors disabled:opacity-50"
                    >
                      Dismiss (False Positive)
                    </button>
                  )}
                  {onConfirm && (
                    <button
                      onClick={handleConfirm}
                      disabled={loading}
                      className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white 
                               rounded-lg font-medium transition-colors disabled:opacity-50"
                    >
                      Confirm (Real Violence)
                    </button>
                  )}
                </div>
              )}
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
