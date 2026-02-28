/**
 * SafeSight - Event Detail Modal
 * ===================================
 * Full-detail modal for violence events showing clip, person captures,
 * confidence stats, and action buttons.
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Play,
  Clock,
  Radio,
  ShieldAlert,
  TrendingUp,
  Users,
  Video,
  ThumbsUp,
  ThumbsDown,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Image as ImageIcon,
  Search,
  Loader2,
} from "lucide-react";
import { ViolenceEvent } from "@/types";
import { streamService } from "@/services/streamApi";
import { cn, formatPercentage } from "@/lib/utils";

interface EventDetailModalProps {
  event: ViolenceEvent | null;
  isOpen: boolean;
  onClose: () => void;
  onActionExecuted?: (id: string) => void;
  onNoActionRequired?: (id: string) => void;
}

const severityConfig: Record<
  string,
  { bg: string; text: string; label: string }
> = {
  critical: { bg: "bg-red-500/20", text: "text-red-400", label: "Critical" },
  high: { bg: "bg-orange-500/20", text: "text-orange-400", label: "High" },
  medium: { bg: "bg-yellow-500/20", text: "text-yellow-400", label: "Medium" },
  low: { bg: "bg-blue-500/20", text: "text-blue-400", label: "Low" },
};

const statusConfig: Record<
  string,
  { icon: React.ReactNode; label: string; color: string }
> = {
  PENDING: {
    icon: <AlertTriangle className="w-4 h-4" />,
    label: "Pending Review",
    color: "text-yellow-400",
  },
  CONFIRMED: {
    icon: <CheckCircle className="w-4 h-4" />,
    label: "Confirmed",
    color: "text-green-400",
  },
  DISMISSED: {
    icon: <XCircle className="w-4 h-4" />,
    label: "Dismissed",
    color: "text-slate-400",
  },
  AUTO_DISMISSED: {
    icon: <XCircle className="w-4 h-4" />,
    label: "Auto Dismissed",
    color: "text-slate-500",
  },
  ACTION_EXECUTED: {
    icon: <ThumbsUp className="w-4 h-4" />,
    label: "Action Executed!",
    color: "text-green-400",
  },
  NO_ACTION_REQUIRED: {
    icon: <ThumbsDown className="w-4 h-4" />,
    label: "No Action Required",
    color: "text-slate-400",
  },
};

export function EventDetailModal({
  event,
  isOpen,
  onClose,
  onActionExecuted,
  onNoActionRequired,
}: EventDetailModalProps) {
  const [selectedPersonImage, setSelectedPersonImage] = useState<string | null>(
    null,
  );
  const [isPlayingClip, setIsPlayingClip] = useState(false);
  const [isExtractingFaces, setIsExtractingFaces] = useState(false);
  const [extractedFaces, setExtractedFaces] = useState<string[]>([]);
  const [extractionError, setExtractionError] = useState<string | null>(null);

  // Handle Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (selectedPersonImage) {
          setSelectedPersonImage(null);
        } else {
          onClose();
        }
      }
    };
    if (isOpen) {
      document.addEventListener("keydown", handler);
      return () => document.removeEventListener("keydown", handler);
    }
  }, [isOpen, onClose, selectedPersonImage]);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (!isOpen) {
      setSelectedPersonImage(null);
      setIsPlayingClip(false);
      setExtractedFaces([]);
      setExtractionError(null);
    }
  }, [isOpen]);

  // Handle face extraction
  const handleExtractFaces = async () => {
    if (!event?.id && !event?.event_id) return;

    setIsExtractingFaces(true);
    setExtractionError(null);

    try {
      const eventId = event.event_id || event.id;
      const result = await streamService.extractFaces(eventId);

      if (result.success && result.data) {
        setExtractedFaces(result.data.faces || []);
        if (result.data.count === 0) {
          setExtractionError("No faces detected in this clip");
        }
      } else {
        setExtractionError(result.error || "Failed to extract faces");
      }
    } catch (err: any) {
      setExtractionError(err.message || "Face extraction failed");
    } finally {
      setIsExtractingFaces(false);
    }
  };

  if (!event) return null;

  const severity = severityConfig[event.severity] || severityConfig.low;
  const status = statusConfig[event.status] || statusConfig.PENDING;

  const clipUrl = event.clip_path
    ? streamService.getClipUrl(event.clip_path)
    : null;
  const thumbnailUrl = event.thumbnail_path
    ? streamService.getThumbnailUrl(event.thumbnail_path)
    : null;

  // Support both legacy person_images, new face_paths, and manually extracted faces
  const personImages = event.person_images || [];
  const facePaths = event.face_paths || [];
  // Combine face_paths with manually extracted faces (avoid duplicates)
  const allFaces = extractedFaces.length > 0 ? extractedFaces : facePaths;
  const hasParticipants = personImages.length > 0 || allFaces.length > 0;
  const participantsCount =
    extractedFaces.length ||
    event.participants_count ||
    event.person_count ||
    facePaths.length ||
    personImages.length;

  // Check if we can extract faces (has clip but no faces yet)
  const canExtractFaces =
    !!event.clip_path && allFaces.length === 0 && personImages.length === 0;

  const formatTime = (ts?: string) => {
    if (!ts) return "--";
    return new Date(ts).toLocaleString();
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return "--";
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="sticky top-0 bg-slate-900/95 backdrop-blur-sm border-b border-slate-800 px-6 py-4 flex items-center justify-between z-10 rounded-t-2xl">
              <div className="flex items-center gap-3">
                <ShieldAlert className={cn("w-6 h-6", severity.text)} />
                <div>
                  <h2 className="text-lg font-semibold text-white">
                    Violence Event Details
                  </h2>
                  <p className="text-sm text-slate-400">
                    {event.stream_name || `Stream #${event.stream_id}`}
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="px-6 py-5 space-y-6">
              {/* Video Player / Thumbnail */}
              <div className="rounded-xl overflow-hidden bg-slate-800 aspect-video relative group">
                {isPlayingClip && clipUrl ? (
                  <video
                    src={clipUrl}
                    className="w-full h-full object-contain bg-black"
                    controls
                    autoPlay
                    onEnded={() => setIsPlayingClip(false)}
                  />
                ) : thumbnailUrl ? (
                  <>
                    <img
                      src={thumbnailUrl}
                      alt="Event thumbnail"
                      className="w-full h-full object-contain"
                    />
                    {clipUrl && (
                      <button
                        onClick={() => setIsPlayingClip(true)}
                        className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <div className="w-16 h-16 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center">
                          <Play className="w-8 h-8 text-white ml-1" />
                        </div>
                      </button>
                    )}
                  </>
                ) : clipUrl ? (
                  <button
                    onClick={() => setIsPlayingClip(true)}
                    className="w-full h-full flex flex-col items-center justify-center text-slate-400 hover:text-white transition-colors"
                  >
                    <Play className="w-12 h-12 mb-2" />
                    <span className="text-sm">Play Clip</span>
                  </button>
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-600">
                    <Video className="w-12 h-12" />
                  </div>
                )}

                {/* Duration badge */}
                {event.clip_duration != null && event.clip_duration > 0 && (
                  <div className="absolute bottom-2 right-2 px-2 py-1 bg-black/70 text-white text-xs rounded-lg">
                    {formatDuration(event.clip_duration)}
                  </div>
                )}
              </div>

              {/* Status & Severity Row */}
              <div className="flex items-center gap-3 flex-wrap">
                <span
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-sm font-semibold uppercase",
                    severity.bg,
                    severity.text,
                  )}
                >
                  {severity.label}
                </span>
                <span
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-slate-800",
                    status.color,
                  )}
                >
                  {status.icon}
                  {status.label}
                </span>
                <span className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 rounded-lg text-sm text-slate-300">
                  <Radio className="w-3.5 h-3.5 text-cyan-400" />
                  {event.stream_name || `Stream #${event.stream_id}`}
                </span>
              </div>

              {/* Action Buttons (if PENDING) */}
              {event.status === "PENDING" &&
                (onActionExecuted || onNoActionRequired) && (
                  <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                    <p className="text-sm text-slate-300 mb-3">
                      Was action taken for this violence event?
                    </p>
                    <div className="flex gap-3">
                      {onActionExecuted && (
                        <button
                          onClick={() => onActionExecuted(event.id)}
                          className="flex items-center gap-2 px-4 py-2 bg-green-500/20 text-green-400 hover:bg-green-500/30 rounded-lg font-medium text-sm transition-colors"
                        >
                          <ThumbsUp className="w-4 h-4" />
                          Yes, Action Taken
                        </button>
                      )}
                      {onNoActionRequired && (
                        <button
                          onClick={() => onNoActionRequired(event.id)}
                          className="flex items-center gap-2 px-4 py-2 bg-slate-700 text-slate-300 hover:bg-slate-600 rounded-lg font-medium text-sm transition-colors"
                        >
                          <ThumbsDown className="w-4 h-4" />
                          No Action Needed
                        </button>
                      )}
                    </div>
                  </div>
                )}

              {/* Confidence Stats */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <StatCard
                  label="Max Confidence"
                  value={formatPercentage(
                    event.max_confidence ?? event.max_score ?? 0,
                  )}
                  icon={<TrendingUp className="w-4 h-4 text-red-400" />}
                />
                <StatCard
                  label="Avg Confidence"
                  value={formatPercentage(
                    event.avg_confidence ?? event.avg_score ?? 0,
                  )}
                  icon={<TrendingUp className="w-4 h-4 text-orange-400" />}
                />
                <StatCard
                  label="Duration"
                  value={formatDuration(event.duration_seconds)}
                  icon={<Clock className="w-4 h-4 text-cyan-400" />}
                />
                <StatCard
                  label="Frames"
                  value={`${event.frame_count ?? 0}`}
                  icon={<Video className="w-4 h-4 text-purple-400" />}
                />
              </div>

              {/* Timestamps */}
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 space-y-2">
                <h3 className="text-sm font-medium text-slate-300 mb-2">
                  Timestamps
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-slate-500">Start: </span>
                    <span className="text-slate-300">
                      {formatTime(event.start_time ?? event.started_at)}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">End: </span>
                    <span className="text-slate-300">
                      {formatTime(event.end_time ?? event.ended_at)}
                    </span>
                  </div>
                  {event.reviewed_at && (
                    <div>
                      <span className="text-slate-500">Reviewed: </span>
                      <span className="text-slate-300">
                        {formatTime(event.reviewed_at)}
                      </span>
                    </div>
                  )}
                  <div>
                    <span className="text-slate-500">Created: </span>
                    <span className="text-slate-300">
                      {formatTime(event.created_at)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Person Captures / Participants */}
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Users className="w-5 h-5 text-cyan-400" />
                    <h3 className="text-sm font-medium text-slate-300">
                      Participants{" "}
                      {hasParticipants ? `(${participantsCount})` : ""}
                    </h3>
                  </div>
                  {/* Fetch Participants Button */}
                  {event.clip_path && (
                    <button
                      onClick={handleExtractFaces}
                      disabled={isExtractingFaces}
                      className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
                    >
                      {isExtractingFaces ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Detecting...
                        </>
                      ) : (
                        <>
                          <Search className="w-4 h-4" />
                          {hasParticipants ? "Re-scan Faces" : "Detect Faces"}
                        </>
                      )}
                    </button>
                  )}
                </div>

                {/* Extraction Error */}
                {extractionError && (
                  <div className="mb-3 px-3 py-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-yellow-400 text-sm">
                    {extractionError}
                  </div>
                )}

                {hasParticipants ? (
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                    {/* Show face_paths (from clip analysis or manual extraction) */}
                    {allFaces.map((img, idx) => (
                      <motion.div
                        key={`face-${idx}`}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setSelectedPersonImage(img)}
                        className="relative aspect-[3/4] bg-slate-900 rounded-lg overflow-hidden cursor-pointer group border border-slate-700 hover:border-cyan-500/50 transition-colors"
                      >
                        <img
                          src={streamService.getFaceUrl(img)}
                          alt={`Participant ${idx + 1}`}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display =
                              "none";
                          }}
                        />
                        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                          <span className="text-xs text-white font-medium">
                            Participant {idx + 1}
                          </span>
                        </div>
                        <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                          <ImageIcon className="w-6 h-6 text-white" />
                        </div>
                      </motion.div>
                    ))}
                    {/* Show legacy person_images */}
                    {personImages.map((img, idx) => (
                      <motion.div
                        key={`person-${idx}`}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setSelectedPersonImage(img)}
                        className="relative aspect-[3/4] bg-slate-900 rounded-lg overflow-hidden cursor-pointer group border border-slate-700 hover:border-cyan-500/50 transition-colors"
                      >
                        <img
                          src={streamService.getPersonImageUrl(img)}
                          alt={`Person ${idx + 1}`}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display =
                              "none";
                          }}
                        />
                        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                          <span className="text-xs text-white font-medium">
                            Person {idx + 1}
                          </span>
                        </div>
                        <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                          <ImageIcon className="w-6 h-6 text-white" />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-6 text-slate-500">
                    <Users className="w-10 h-10 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">
                      {event.clip_path
                        ? 'Click "Detect Faces" to scan for participants'
                        : "No clip available for face detection"}
                    </p>
                  </div>
                )}
              </div>

              {/* Notes */}
              {event.notes && (
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                  <h3 className="text-sm font-medium text-slate-300 mb-2">
                    Notes
                  </h3>
                  <p className="text-sm text-slate-400">{event.notes}</p>
                </div>
              )}
            </div>
          </motion.div>

          {/* Person Image Lightbox */}
          <AnimatePresence>
            {selectedPersonImage && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-[60] bg-black/90 flex items-center justify-center p-8"
                onClick={() => setSelectedPersonImage(null)}
              >
                <motion.img
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.8, opacity: 0 }}
                  src={
                    selectedPersonImage.includes("face_participants")
                      ? streamService.getFaceUrl(selectedPersonImage)
                      : streamService.getPersonImageUrl(selectedPersonImage)
                  }
                  alt="Participant capture"
                  className="max-w-full max-h-full object-contain rounded-xl shadow-2xl"
                  onClick={(e) => e.stopPropagation()}
                />
                <button
                  onClick={() => setSelectedPersonImage(null)}
                  className="absolute top-6 right-6 p-2 text-white/70 hover:text-white bg-black/50 rounded-full transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// Stat card sub-component
function StatCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-3 text-center">
      <div className="flex items-center justify-center gap-1.5 mb-1">
        {icon}
        <span className="text-xs text-slate-500">{label}</span>
      </div>
      <span className="text-lg font-semibold text-white">{value}</span>
    </div>
  );
}

export default EventDetailModal;
