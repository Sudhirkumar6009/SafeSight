/**
 * SafeSight - Streams Management Page
 * ========================================
 * Manage RTSP camera streams for violence detection.
 */

"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Navbar } from "@/components";
import { StreamCard } from "@/components/StreamCard";
import { StreamForm } from "@/components/StreamForm";
import { Stream, StreamCreateRequest, InferenceScoreMessage } from "@/types";
import { streamService } from "@/services/streamApi";
import { useWebSocket } from "@/hooks/useWebSocket";

export default function StreamsPage() {
  const [streams, setStreams] = useState<Stream[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [editingStream, setEditingStream] = useState<Stream | null>(null);

  // WebSocket for real-time scores
  const { scores, isConnected } = useWebSocket();

  // Fetch streams
  const fetchStreams = useCallback(async () => {
    try {
      const response = await streamService.getStreams();
      if (response.success && response.data) {
        setStreams(response.data);
        setError(null);
      } else {
        setError(response.error || "Failed to fetch streams");
      }
    } catch (err: any) {
      setError(err.message || "Failed to connect to RTSP service");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStreams();
    // Poll for stream status updates
    const interval = setInterval(fetchStreams, 10000);
    return () => clearInterval(interval);
  }, [fetchStreams]);

  // Stream actions
  const handleStart = useCallback(
    async (id: string) => {
      try {
        await streamService.startStream(id);
        await fetchStreams();
      } catch (err: any) {
        console.error("Failed to start stream:", err);
      }
    },
    [fetchStreams],
  );

  const handleStop = useCallback(
    async (id: string) => {
      try {
        await streamService.stopStream(id);
        await fetchStreams();
      } catch (err: any) {
        console.error("Failed to stop stream:", err);
      }
    },
    [fetchStreams],
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

  // Stats
  const runningCount = streams.filter((s) => s.status === "running").length;
  const stoppedCount = streams.filter((s) => s.status === "stopped").length;
  const errorCount = streams.filter((s) => s.status === "error").length;

  return (
    <div className="min-h-screen bg-gray-950">
      <Navbar />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Streams</h1>
            <p className="text-gray-400 mt-1">
              Manage RTSP camera feeds for violence detection
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* WebSocket Status */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-lg">
              <div
                className={`h-2 w-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"}`}
              />
              <span className="text-sm text-gray-400">
                {isConnected ? "Live" : "Disconnected"}
              </span>
            </div>
            <button
              onClick={() => setFormOpen(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              Add Stream
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-3xl font-bold text-white">{streams.length}</p>
            <p className="text-sm text-gray-400">Total Streams</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-3xl font-bold text-green-400">{runningCount}</p>
            <p className="text-sm text-gray-400">Running</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-3xl font-bold text-gray-400">{stoppedCount}</p>
            <p className="text-sm text-gray-400">Stopped</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-3xl font-bold text-red-400">{errorCount}</p>
            <p className="text-sm text-gray-400">Errors</p>
          </div>
        </div>

        {/* Error State */}
        {error && (
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
                <p className="text-red-300/70 text-sm">{error}</p>
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

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500" />
          </div>
        )}

        {/* Empty State */}
        {!loading && streams.length === 0 && !error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-20"
          >
            <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gray-800 flex items-center justify-center">
              <svg
                className="w-10 h-10 text-gray-500"
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
            <p className="text-gray-400 mb-6 max-w-md mx-auto">
              Add your first RTSP camera stream to start detecting violence in
              real-time.
            </p>
            <button
              onClick={() => setFormOpen(true)}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              Add Your First Stream
            </button>
          </motion.div>
        )}

        {/* Streams Grid */}
        {!loading && streams.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence mode="popLayout">
              {streams.map((stream) => (
                <StreamCard
                  key={stream.id}
                  stream={stream}
                  score={scores.get(stream.id)}
                  onStart={handleStart}
                  onStop={handleStop}
                  onEdit={handleEdit}
                  onDelete={handleDelete}
                />
              ))}
            </AnimatePresence>
          </div>
        )}
      </main>

      {/* Stream Form Modal */}
      <StreamForm
        stream={editingStream}
        isOpen={formOpen}
        onClose={handleCloseForm}
        onSubmit={handleSubmit}
      />
    </div>
  );
}
