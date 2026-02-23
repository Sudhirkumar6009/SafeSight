"use client";

import React, { useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Brain,
  Shield,
  ShieldAlert,
  Clock,
  Trash2,
  Eye,
  TrendingUp,
  Loader2,
  Radio,
  Play,
  Pause,
  X,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Video,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  Users,
} from "lucide-react";
import { apiService } from "@/services/api";
import { streamService } from "@/services/streamApi";
import { Prediction, Video as VideoType, ViolenceEvent } from "@/types";
import { useWebSocket } from "@/hooks/useWebSocket";
import { cn, formatPercentage, formatDate } from "@/lib/utils";
import { EventDetailModal } from "@/components/EventDetailModal";
import type { TabId } from "@/components/DashboardSidebar";

interface HistoryTabProps {
  onTabChange: (tab: TabId) => void;
}

type ViewMode = "all" | "uploads" | "streams";

const severityColors: Record<
  string,
  { bg: string; text: string; border: string }
> = {
  critical: {
    bg: "bg-red-500/20",
    text: "text-red-400",
    border: "border-l-red-500",
  },
  high: {
    bg: "bg-orange-500/20",
    text: "text-orange-400",
    border: "border-l-orange-500",
  },
  medium: {
    bg: "bg-yellow-500/20",
    text: "text-yellow-400",
    border: "border-l-yellow-500",
  },
  low: {
    bg: "bg-blue-500/20",
    text: "text-blue-400",
    border: "border-l-blue-500",
  },
};

const statusIcons: Record<string, React.ReactNode> = {
  PENDING: <AlertTriangle className="w-4 h-4 text-yellow-400" />,
  CONFIRMED: <CheckCircle className="w-4 h-4 text-red-400" />,
  DISMISSED: <XCircle className="w-4 h-4 text-slate-400" />,
  AUTO_DISMISSED: <XCircle className="w-4 h-4 text-slate-500" />,
  ACTION_EXECUTED: <ThumbsUp className="w-4 h-4 text-green-400" />,
  NO_ACTION_REQUIRED: <ThumbsDown className="w-4 h-4 text-slate-400" />,
};

export default function HistoryTab({ onTabChange }: HistoryTabProps) {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [streamEvents, setStreamEvents] = useState<ViolenceEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingEvents, setIsLoadingEvents] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [filter, setFilter] = useState<"all" | "violence" | "non-violence">(
    "all",
  );
  const [viewMode, setViewMode] = useState<ViewMode>("all");
  const [playingClip, setPlayingClip] = useState<string | null>(null);
  const [videoModal, setVideoModal] = useState<{
    url: string;
    title: string;
  } | null>(null);
  const [newEventFlash, setNewEventFlash] = useState(false);
  const [updatingEventId, setUpdatingEventId] = useState<
    string | number | null
  >(null);
  const [selectedEvent, setSelectedEvent] = useState<ViolenceEvent | null>(
    null,
  );
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);

  // Auto-refresh stream events when violence_alert or event_end arrives
  const handleAlert = useCallback((alert: any) => {
    if (
      alert.type === "violence_alert" ||
      alert.type === "event_end" ||
      alert.type === "event_start"
    ) {
      // Flash indicator and refresh events
      setNewEventFlash(true);
      setTimeout(() => setNewEventFlash(false), 3000);
      fetchStreamEvents();
    }
  }, []);

  useWebSocket({ onAlert: handleAlert });

  useEffect(() => {
    fetchPredictions();
  }, [page, filter]);

  useEffect(() => {
    fetchStreamEvents();
  }, []);

  const fetchPredictions = async () => {
    setIsLoading(true);
    try {
      const classification = filter === "all" ? undefined : filter;
      const response = await apiService.getPredictions(
        page,
        10,
        classification,
      );
      if (response.success && response.data) {
        setPredictions(response.data);
        setTotalPages(response.pagination?.pages || 1);
      }
    } catch (error) {
      console.error("Failed to fetch predictions:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchStreamEvents = async () => {
    setIsLoadingEvents(true);
    try {
      const response = await streamService.getEvents({ limit: 50 });
      if (response.success && response.data) {
        setStreamEvents(response.data);
      }
    } catch (error) {
      console.error("Failed to fetch stream events:", error);
    } finally {
      setIsLoadingEvents(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this prediction?")) return;
    try {
      await apiService.deletePrediction(id);
      fetchPredictions();
    } catch (error) {
      console.error("Failed to delete prediction:", error);
    }
  };

  const handleActionExecuted = async (eventId: string | number) => {
    setUpdatingEventId(eventId);
    try {
      const response = await streamService.markActionExecuted(eventId);
      if (response.success) {
        await fetchStreamEvents();
      }
    } catch (error) {
      console.error("Failed to update event:", error);
    } finally {
      setUpdatingEventId(null);
    }
  };

  const handleNoActionRequired = async (eventId: string | number) => {
    setUpdatingEventId(eventId);
    try {
      const response = await streamService.markNoActionRequired(eventId);
      if (response.success) {
        await fetchStreamEvents();
      }
    } catch (error) {
      console.error("Failed to update event:", error);
    } finally {
      setUpdatingEventId(null);
    }
  };

  const getStatusLabel = (status: string) => {
    const labels: Record<string, string> = {
      PENDING: "Pending",
      ACTION_EXECUTED: "Action Executed!",
      NO_ACTION_REQUIRED: "No Action Required!",
      CONFIRMED: "Confirmed",
      DISMISSED: "Dismissed",
      AUTO_DISMISSED: "Auto Dismissed",
    };
    return labels[status] || status.replace("_", " ");
  };

  const getClipUrl = (clipPath: string) => streamService.getClipUrl(clipPath);
  const getThumbnailUrl = (thumbPath: string) =>
    streamService.getThumbnailUrl(thumbPath);

  const formatDuration = (seconds?: number) => {
    if (!seconds) return "N/A";
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  const formatEventTime = (isoString?: string) => {
    if (!isoString) return "Unknown";
    return new Date(isoString).toLocaleString();
  };

  const showUploads = viewMode === "all" || viewMode === "uploads";
  const showStreams = viewMode === "all" || viewMode === "streams";

  return (
    <>
      <div>
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">History</h1>
            <p className="text-slate-400">
              Video analysis results & stream detections
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* View Mode Toggle */}
            <div className="flex items-center gap-1 p-1 bg-slate-800 rounded-xl">
              {(
                [
                  { key: "all", label: "All", icon: Eye },
                  { key: "uploads", label: "Uploads", icon: Video },
                  { key: "streams", label: "Streams", icon: Radio },
                ] as const
              ).map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => setViewMode(key)}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                    viewMode === key
                      ? "bg-cyan-500/20 text-cyan-400"
                      : "text-slate-400 hover:text-white hover:bg-slate-700",
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              ))}
            </div>

            {/* Classification Filter (for uploads) */}
            {showUploads && (
              <div className="flex items-center gap-1 p-1 bg-slate-800 rounded-xl">
                {(["all", "violence", "non-violence"] as const).map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={cn(
                      "px-3 py-2 rounded-lg text-sm font-medium transition-colors capitalize",
                      filter === f
                        ? f === "violence"
                          ? "bg-red-500/20 text-red-400"
                          : f === "non-violence"
                            ? "bg-green-500/20 text-green-400"
                            : "bg-cyan-500/20 text-cyan-400"
                        : "text-slate-400 hover:text-white hover:bg-slate-700",
                    )}
                  >
                    {f === "non-violence"
                      ? "Safe"
                      : f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ========== Stream Detection Events ========== */}
        {showStreams && (
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center">
                <Radio className="w-4 h-4 text-red-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">
                  Live Stream Detections
                </h2>
                <p className="text-sm text-slate-400">
                  Violence detected on RTSP/camera streams
                </p>
              </div>
              {streamEvents.length > 0 && (
                <span className="ml-auto px-2.5 py-1 bg-red-500/10 text-red-400 text-sm font-medium rounded-lg">
                  {streamEvents.length} event
                  {streamEvents.length !== 1 ? "s" : ""}
                </span>
              )}
              {newEventFlash && (
                <span className="px-2.5 py-1 bg-green-500/20 text-green-400 text-xs font-medium rounded-lg animate-pulse">
                  New event!
                </span>
              )}
              <button
                onClick={() => fetchStreamEvents()}
                className="ml-2 p-1.5 text-slate-500 hover:text-cyan-400 hover:bg-slate-800 rounded-lg transition-colors"
                title="Refresh events"
              >
                <RefreshCw
                  className={cn("w-4 h-4", isLoadingEvents && "animate-spin")}
                />
              </button>
            </div>

            {isLoadingEvents ? (
              <div className="flex items-center justify-center py-10">
                <Loader2 className="w-6 h-6 text-cyan-500 animate-spin" />
              </div>
            ) : streamEvents.length === 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-slate-900/50 border border-slate-800 rounded-xl p-8 text-center mb-6"
              >
                <Radio className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                <h3 className="text-lg font-medium text-slate-300 mb-1">
                  No Stream Events
                </h3>
                <p className="text-sm text-slate-500">
                  Violence detections from live streams will appear here
                </p>
              </motion.div>
            ) : (
              <div className="grid gap-3 mb-6">
                {streamEvents.map((event, index) => {
                  const severity =
                    severityColors[event.severity] || severityColors.low;
                  const isPlaying = playingClip === String(event.id);

                  return (
                    <motion.div
                      key={event.id}
                      initial={{ opacity: 0, y: 15 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.04 }}
                      className={cn(
                        "bg-slate-900 border border-slate-800 rounded-xl overflow-hidden border-l-4 cursor-pointer hover:border-slate-600 transition-colors",
                        severity.border,
                      )}
                      onClick={() => {
                        setSelectedEvent(event);
                        setIsDetailModalOpen(true);
                      }}
                    >
                      <div className="flex gap-4 p-4">
                        {/* Thumbnail / Video Player */}
                        <div className="flex-shrink-0 w-48 h-28 bg-slate-800 rounded-lg overflow-hidden relative group">
                          {event.thumbnail_path ? (
                            <>
                              <img
                                src={getThumbnailUrl(event.thumbnail_path)}
                                alt="Event thumbnail"
                                className="w-full h-full object-cover"
                              />
                              {event.clip_path && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setVideoModal({
                                      url: getClipUrl(event.clip_path!),
                                      title:
                                        event.stream_name ||
                                        `Event #${event.id}`,
                                    });
                                  }}
                                  className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity"
                                >
                                  <div className="w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center">
                                    <Play className="w-5 h-5 text-white ml-0.5" />
                                  </div>
                                </button>
                              )}
                            </>
                          ) : event.clip_path ? (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setVideoModal({
                                  url: getClipUrl(event.clip_path!),
                                  title:
                                    event.stream_name || `Event #${event.id}`,
                                });
                              }}
                              className="w-full h-full flex flex-col items-center justify-center text-slate-500 hover:text-white transition-colors"
                            >
                              <Play className="w-8 h-8 mb-1" />
                              <span className="text-xs">Play Clip</span>
                            </button>
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-slate-600">
                              <Video className="w-8 h-8" />
                            </div>
                          )}

                          {/* Clip duration badge */}
                          {event.clip_duration != null &&
                            event.clip_duration > 0 && (
                              <div className="absolute bottom-1 right-1 px-1.5 py-0.5 bg-black/70 text-white text-xs rounded">
                                {formatDuration(event.clip_duration)}
                              </div>
                            )}
                        </div>

                        {/* Event Details */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <ShieldAlert
                                className={cn("w-5 h-5", severity.text)}
                              />
                              <span
                                className={cn("font-semibold", severity.text)}
                              >
                                Violence Detected
                              </span>
                              <span
                                className={cn(
                                  "px-2 py-0.5 rounded text-xs font-semibold uppercase",
                                  severity.bg,
                                  severity.text,
                                )}
                              >
                                {event.severity}
                              </span>
                            </div>
                            {event.status === "PENDING" ? (
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-slate-400 mr-1">
                                  Action?
                                </span>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleActionExecuted(event.id);
                                  }}
                                  disabled={updatingEventId === event.id}
                                  className="px-2 py-1 text-xs font-medium bg-green-500/20 text-green-400 hover:bg-green-500/30 rounded-lg transition-colors disabled:opacity-50"
                                >
                                  {updatingEventId === event.id ? "..." : "Yes"}
                                </button>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleNoActionRequired(event.id);
                                  }}
                                  disabled={updatingEventId === event.id}
                                  className="px-2 py-1 text-xs font-medium bg-slate-700 text-slate-300 hover:bg-slate-600 rounded-lg transition-colors disabled:opacity-50"
                                >
                                  {updatingEventId === event.id ? "..." : "No"}
                                </button>
                              </div>
                            ) : (
                              <div className="flex items-center gap-1.5 text-sm">
                                {statusIcons[event.status]}
                                <span className="text-slate-400">
                                  {getStatusLabel(event.status)}
                                </span>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Stream info */}
                        <div className="flex items-center gap-3 mb-2 text-sm">
                          <span className="flex items-center gap-1.5 px-2 py-0.5 bg-slate-800 rounded-lg">
                            <Radio className="w-3.5 h-3.5 text-cyan-400" />
                            <span className="text-slate-300 font-medium">
                              {event.stream_name ||
                                `Stream #${event.stream_id}`}
                            </span>
                          </span>
                          {event.duration_seconds != null && (
                            <span className="text-slate-400">
                              Duration: {formatDuration(event.duration_seconds)}
                            </span>
                          )}
                        </div>

                        {/* Confidence bar */}
                        <div className="flex items-center gap-3 mb-2">
                          <div className="flex-1">
                            <div className="flex justify-between text-xs mb-1">
                              <span className="text-slate-400">
                                Max:{" "}
                                {formatPercentage(
                                  event.max_confidence ?? event.max_score ?? 0,
                                )}
                              </span>
                              <span className="text-slate-400">
                                Avg:{" "}
                                {formatPercentage(
                                  event.avg_confidence ?? event.avg_score ?? 0,
                                )}
                              </span>
                            </div>
                            <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className={cn(
                                  "h-full rounded-full",
                                  (event.max_confidence ??
                                    event.max_score ??
                                    0) >= 0.8
                                    ? "bg-red-500"
                                    : (event.max_confidence ??
                                          event.max_score ??
                                          0) >= 0.65
                                      ? "bg-orange-500"
                                      : "bg-yellow-500",
                                )}
                                style={{
                                  width: `${(event.max_confidence ?? event.max_score ?? 0) * 100}%`,
                                }}
                              />
                            </div>
                          </div>
                        </div>

                        {/* Timestamp & Person count */}
                        <div className="flex items-center justify-between text-xs text-slate-500">
                          <div className="flex items-center gap-1">
                            <Clock className="w-3.5 h-3.5" />
                            {formatEventTime(
                              event.start_time ??
                                event.started_at ??
                                event.created_at,
                            )}
                          </div>
                          <div className="flex items-center gap-3">
                            {(event.person_count ?? 0) > 0 && (
                              <span className="flex items-center gap-1 px-2 py-0.5 bg-cyan-500/10 text-cyan-400 rounded-lg text-xs">
                                <Users className="w-3 h-3" />
                                {event.person_count} person
                                {(event.person_count ?? 0) > 1 ? "s" : ""}
                              </span>
                            )}
                            <span className="text-cyan-400/60 text-xs">
                              Click for details →
                            </span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ========== Upload Predictions ========== */}
        {showUploads && (
          <div>
            {showStreams && (
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
                  <Video className="w-4 h-4 text-cyan-400" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-white">
                    Upload Predictions
                  </h2>
                  <p className="text-sm text-slate-400">
                    Results from manually uploaded videos
                  </p>
                </div>
              </div>
            )}

            {/* Content */}
            {isLoading ? (
              <div className="flex items-center justify-center py-20">
                <Loader2 className="w-8 h-8 text-cyan-500 animate-spin" />
              </div>
            ) : predictions.length === 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center"
              >
                <Brain className="w-16 h-16 text-slate-500 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">
                  No Predictions Yet
                </h3>
                <p className="text-slate-400 mb-6">
                  Upload and analyze videos to see predictions here
                </p>
                <button
                  onClick={() => onTabChange("upload")}
                  className="inline-flex items-center gap-2 px-6 py-3 bg-cyan-500 hover:bg-cyan-600 text-white rounded-xl font-medium transition-colors"
                >
                  <TrendingUp className="w-5 h-5" /> Analyze Video
                </button>
              </motion.div>
            ) : (
              <div className="grid gap-4">
                {predictions.map((prediction, index) => {
                  const isViolent = prediction.classification === "violence";
                  const video = prediction.videoId as VideoType;
                  return (
                    <motion.div
                      key={prediction._id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className={cn(
                        "bg-slate-900 border border-slate-800 rounded-xl p-4 border-l-4",
                        isViolent ? "border-l-red-500" : "border-l-green-500",
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div
                            className={cn(
                              "w-14 h-14 rounded-xl flex items-center justify-center",
                              isViolent ? "bg-red-500/20" : "bg-green-500/20",
                            )}
                          >
                            {isViolent ? (
                              <ShieldAlert className="w-7 h-7 text-red-400" />
                            ) : (
                              <Shield className="w-7 h-7 text-green-400" />
                            )}
                          </div>
                          <div>
                            <div className="flex items-center gap-3 mb-1">
                              <span
                                className={cn(
                                  "text-lg font-semibold capitalize",
                                  isViolent ? "text-red-400" : "text-green-400",
                                )}
                              >
                                {(
                                  prediction.classification ?? "unknown"
                                ).replace("-", " ")}
                              </span>
                              <span
                                className={cn(
                                  "px-2 py-0.5 rounded text-sm font-medium",
                                  isViolent
                                    ? "bg-red-500/10 text-red-400"
                                    : "bg-green-500/10 text-green-400",
                                )}
                              >
                                {formatPercentage(prediction.confidence)}
                              </span>
                            </div>
                            <div className="flex items-center gap-4 text-sm text-slate-400">
                              <span className="truncate max-w-xs">
                                {video && typeof video === "object"
                                  ? (video.originalName ??
                                    video.filename ??
                                    "Unknown video")
                                  : "Unknown video"}
                              </span>
                              <span className="flex items-center gap-1">
                                <Clock className="w-4 h-4" />
                                {formatDate(prediction.createdAt)}
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="hidden md:block w-40">
                            <div className="flex justify-between text-xs mb-1">
                              <span className="text-red-400">
                                V:{" "}
                                {formatPercentage(
                                  prediction.probabilities?.violence ?? 0,
                                )}
                              </span>
                              <span className="text-green-400">
                                NV:{" "}
                                {formatPercentage(
                                  prediction.probabilities?.nonViolence ?? 0,
                                )}
                              </span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full overflow-hidden flex">
                              <div
                                className="h-full bg-red-500"
                                style={{
                                  width: `${(prediction.probabilities?.violence ?? 0) * 100}%`,
                                }}
                              />
                              <div
                                className="h-full bg-green-500"
                                style={{
                                  width: `${(prediction.probabilities?.nonViolence ?? 0) * 100}%`,
                                }}
                              />
                            </div>
                          </div>
                          <button
                            onClick={() => handleDelete(prediction._id)}
                            className="p-2 hover:bg-red-500/10 rounded-lg transition-colors"
                          >
                            <Trash2 className="w-5 h-5 text-slate-400 hover:text-red-400" />
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            )}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-8">
                <button
                  onClick={() => setPage(Math.max(1, page - 1))}
                  disabled={page === 1}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <span className="px-4 py-2 text-slate-400">
                  Page {page} of {totalPages}
                </span>
                <button
                  onClick={() => setPage(Math.min(totalPages, page + 1))}
                  disabled={page === totalPages}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Full-screen Video Player Modal */}
      {videoModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
          onClick={() => setVideoModal(null)}
        >
          <div
            className="relative w-full max-w-4xl mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
              <span className="text-white font-semibold truncate">
                {videoModal.title}
              </span>
              <button
                onClick={() => setVideoModal(null)}
                className="ml-4 p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            {/* Video */}
            <video
              src={videoModal.url}
              className="w-full rounded-xl bg-black max-h-[75vh]"
              controls
              autoPlay
              onEnded={() => setVideoModal(null)}
            />
          </div>
        </div>
      )}

      {/* Event Detail Modal */}
      <EventDetailModal
        event={selectedEvent}
        isOpen={isDetailModalOpen}
        onClose={() => {
          setIsDetailModalOpen(false);
          setSelectedEvent(null);
        }}
        onActionExecuted={(id) => {
          handleActionExecuted(id);
          setIsDetailModalOpen(false);
          setSelectedEvent(null);
        }}
        onNoActionRequired={(id) => {
          handleNoActionRequired(id);
          setIsDetailModalOpen(false);
          setSelectedEvent(null);
        }}
      />
    </>
  );
}
