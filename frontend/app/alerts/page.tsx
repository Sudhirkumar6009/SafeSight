/**
 * SafeSight - Alerts Page
 * ===========================
 * View and manage violence detection events.
 */

"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Navbar } from "@/components";
import { AlertCard, LiveAlertBanner } from "@/components/AlertCard";
import { EventModal } from "@/components/EventModal";
import { ViolenceEvent, EventFilters, AlertMessage } from "@/types";
import { eventService } from "@/services/streamApi";
import { useWebSocket } from "@/hooks/useWebSocket";

type StatusFilter = "all" | "pending" | "confirmed" | "dismissed";
type SeverityFilter = "all" | "low" | "medium" | "high" | "critical";

export default function AlertsPage() {
  const [events, setEvents] = useState<ViolenceEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<ViolenceEvent | null>(
    null,
  );
  const [modalOpen, setModalOpen] = useState(false);

  // Filters
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [severityFilter, setSeverityFilter] = useState<SeverityFilter>("all");
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [total, setTotal] = useState(0);

  // Live alert banner
  const [liveAlert, setLiveAlert] = useState<ViolenceEvent | null>(null);

  // WebSocket for live alerts
  const { isConnected } = useWebSocket({
    onAlert: useCallback((data: AlertMessage) => {
      // Show live alert banner for new events
      if (data.type === "event_start") {
        const newEvent: ViolenceEvent = {
          id: data.event_id,
          stream_id: data.stream_id,
          stream_name: data.stream_name,
          started_at: data.timestamp,
          max_score: data.max_score,
          avg_score: 0,
          status: "pending",
          severity: data.severity || "high",
        };
        setLiveAlert(newEvent);

        // Auto-dismiss after 10 seconds
        setTimeout(() => setLiveAlert(null), 10000);

        // Add to events list
        setEvents((prev) => [newEvent, ...prev]);
      }
    }, []),
  });

  // Fetch events
  const fetchEvents = useCallback(
    async (reset = false) => {
      try {
        const filters: EventFilters = {
          limit: 20,
          offset: reset ? 0 : offset,
        };

        if (statusFilter !== "all") filters.status = statusFilter;
        if (severityFilter !== "all") filters.severity = severityFilter;

        const response = await eventService.getEvents(filters);

        if (response.success) {
          if (reset) {
            setEvents(response.data);
          } else {
            setEvents((prev) => [...prev, ...response.data]);
          }
          setTotal(response.pagination?.total || response.data.length);
          setHasMore(response.pagination?.hasMore || false);
          setError(null);
        } else {
          setError("Failed to fetch events");
        }
      } catch (err: any) {
        setError(err.message || "Failed to connect to backend");
      } finally {
        setLoading(false);
      }
    },
    [statusFilter, severityFilter, offset],
  );

  // Initial fetch and filter changes
  useEffect(() => {
    setLoading(true);
    setOffset(0);
    fetchEvents(true);
  }, [statusFilter, severityFilter]);

  // Pagination
  const loadMore = useCallback(() => {
    if (!loading && hasMore) {
      setOffset((prev) => prev + 20);
    }
  }, [loading, hasMore]);

  useEffect(() => {
    if (offset > 0) {
      fetchEvents(false);
    }
  }, [offset]);

  // Event actions
  const handleConfirm = useCallback(async (id: string) => {
    try {
      await eventService.updateEventStatus(id, { status: "confirmed" });
      setEvents((prev) =>
        prev.map((e) => (e.id === id ? { ...e, status: "confirmed" } : e)),
      );
    } catch (err) {
      console.error("Failed to confirm event:", err);
    }
  }, []);

  const handleDismiss = useCallback(async (id: string) => {
    try {
      await eventService.updateEventStatus(id, { status: "dismissed" });
      setEvents((prev) =>
        prev.map((e) => (e.id === id ? { ...e, status: "dismissed" } : e)),
      );
    } catch (err) {
      console.error("Failed to dismiss event:", err);
    }
  }, []);

  const handleViewClip = useCallback((event: ViolenceEvent) => {
    setSelectedEvent(event);
    setModalOpen(true);
  }, []);

  const handleEventClick = useCallback((event: ViolenceEvent) => {
    setSelectedEvent(event);
    setModalOpen(true);
  }, []);

  // Stats
  const pendingCount = events.filter((e) => e.status === "pending").length;
  const confirmedCount = events.filter((e) => e.status === "confirmed").length;
  const dismissedCount = events.filter((e) => e.status === "dismissed").length;

  return (
    <div className="min-h-screen bg-gray-950">
      <Navbar />

      {/* Live Alert Banner */}
      <AnimatePresence>
        {liveAlert && (
          <LiveAlertBanner
            event={liveAlert}
            onConfirm={() => {
              handleConfirm(liveAlert.id);
              setLiveAlert(null);
            }}
            onDismiss={() => {
              handleDismiss(liveAlert.id);
              setLiveAlert(null);
            }}
            onClose={() => setLiveAlert(null)}
          />
        )}
      </AnimatePresence>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Alerts</h1>
            <p className="text-gray-400 mt-1">
              Violence detection events requiring review
            </p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-lg">
            <div
              className={`h-2 w-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"}`}
            />
            <span className="text-sm text-gray-400">
              {isConnected ? "Live Updates" : "Disconnected"}
            </span>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-3xl font-bold text-white">{total}</p>
            <p className="text-sm text-gray-400">Total Events</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center gap-2">
              <p className="text-3xl font-bold text-yellow-400">
                {pendingCount}
              </p>
              {pendingCount > 0 && (
                <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
              )}
            </div>
            <p className="text-sm text-gray-400">Pending Review</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-3xl font-bold text-green-400">
              {confirmedCount}
            </p>
            <p className="text-sm text-gray-400">Confirmed</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-3xl font-bold text-gray-400">{dismissedCount}</p>
            <p className="text-sm text-gray-400">Dismissed</p>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-4 mb-6">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Status:</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
              className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All</option>
              <option value="pending">Pending</option>
              <option value="confirmed">Confirmed</option>
              <option value="dismissed">Dismissed</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Severity:</span>
            <select
              value={severityFilter}
              onChange={(e) =>
                setSeverityFilter(e.target.value as SeverityFilter)
              }
              className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
          <button
            onClick={() => fetchEvents(true)}
            className="ml-auto px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-lg transition-colors"
          >
            Refresh
          </button>
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
                onClick={() => fetchEvents(true)}
                className="ml-auto px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && events.length === 0 && (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500" />
          </div>
        )}

        {/* Empty State */}
        {!loading && events.length === 0 && !error && (
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
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">
              No Events Found
            </h3>
            <p className="text-gray-400 max-w-md mx-auto">
              {statusFilter !== "all" || severityFilter !== "all"
                ? "Try adjusting your filters to see more events."
                : "Violence detection events will appear here when detected."}
            </p>
          </motion.div>
        )}

        {/* Events List */}
        {events.length > 0 && (
          <div className="space-y-4">
            <AnimatePresence mode="popLayout">
              {events.map((event) => (
                <AlertCard
                  key={event.id}
                  event={event}
                  onConfirm={handleConfirm}
                  onDismiss={handleDismiss}
                  onViewClip={handleViewClip}
                  onClick={handleEventClick}
                />
              ))}
            </AnimatePresence>

            {/* Load More */}
            {hasMore && (
              <div className="text-center pt-4">
                <button
                  onClick={loadMore}
                  disabled={loading}
                  className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg 
                             font-medium transition-colors disabled:opacity-50"
                >
                  {loading ? "Loading..." : "Load More"}
                </button>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Event Modal */}
      <EventModal
        event={selectedEvent}
        isOpen={modalOpen}
        onClose={() => {
          setModalOpen(false);
          setSelectedEvent(null);
        }}
        onConfirm={handleConfirm}
        onDismiss={handleDismiss}
      />
    </div>
  );
}
