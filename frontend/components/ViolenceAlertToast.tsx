"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  X,
  Play,
  Clock,
  Radio,
  CheckCircle,
  BellOff,
  Eye,
} from "lucide-react";
import { AlertMessage } from "@/types";

type AlertState = "active" | "acknowledged" | "dismissed";

interface ActiveAlert {
  id: string;
  alert: AlertMessage;
  timestamp: Date;
  state: AlertState;
}

interface ViolenceAlertToastProps {
  alerts: AlertMessage[];
  onDismiss?: (eventId: string) => void;
  onViewHistory?: () => void;
}

export default function ViolenceAlertToast({
  alerts,
  onDismiss,
  onViewHistory,
}: ViolenceAlertToastProps) {
  const [activeAlerts, setActiveAlerts] = useState<ActiveAlert[]>([]);
  const processedEventsRef = useRef<Set<string>>(new Set());
  const dismissedEventsRef = useRef<Set<string>>(new Set());

  // Process new alerts — only show ONE alert per violence event
  useEffect(() => {
    if (alerts.length === 0) return;
    const latest = alerts[0];
    if (!latest?.event_id) return;

    const eventId = latest.event_id;

    // If this event was already dismissed by user, don't show it again
    if (dismissedEventsRef.current.has(eventId)) {
      return;
    }

    // Handle event_start — show the alert
    if (latest.type === "event_start") {
      // Only create one alert per event_id
      if (processedEventsRef.current.has(eventId)) return;
      processedEventsRef.current.add(eventId);

      const id = `${eventId}_alert`;
      setActiveAlerts((prev) => {
        const exists = prev.some((a) => a.alert.event_id === eventId);
        if (exists) return prev;
        const newAlert: ActiveAlert = {
          id,
          alert: latest,
          timestamp: new Date(),
          state: "active",
        };
        return [newAlert, ...prev].slice(0, 5);
      });
    }

    // Handle event_end / violence_alert — update existing alert
    if (latest.type === "violence_alert" || latest.type === "event_end") {
      setActiveAlerts((prev) =>
        prev.map((a) => {
          if (a.alert.event_id === eventId) {
            // Update with clip info from event_end
            return {
              ...a,
              alert: {
                ...a.alert,
                ...latest,
                type: latest.type,
              },
            };
          }
          return a;
        }),
      );

      // If this event was acknowledged, auto-remove after 10s (gives user time to see "clip saved")
      setTimeout(() => {
        setActiveAlerts((prev) =>
          prev.filter((a) => {
            if (
              a.alert.event_id === eventId &&
              a.state === "acknowledged"
            ) {
              return false;
            }
            return true;
          }),
        );
      }, 10000);
    }
  }, [alerts]);

  // Play audio alert only for new active alerts
  const prevAlertCountRef = useRef(0);
  useEffect(() => {
    const activeCount = activeAlerts.filter(
      (a) => a.state === "active",
    ).length;
    if (activeCount > prevAlertCountRef.current) {
      try {
        const ctx = new (
          window.AudioContext || (window as any).webkitAudioContext
        )();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 880;
        osc.type = "sine";
        gain.gain.value = 0.15;
        osc.start();
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
        osc.stop(ctx.currentTime + 0.5);
      } catch {
        // Audio not available
      }
    }
    prevAlertCountRef.current = activeCount;
  }, [activeAlerts]);

  // Option 1: "I know, proceeding" — acknowledge but keep visible until violence stops
  const handleAcknowledge = useCallback((alertId: string) => {
    setActiveAlerts((prev) =>
      prev.map((a) =>
        a.id === alertId ? { ...a, state: "acknowledged" as AlertState } : a,
      ),
    );
  }, []);

  // Option 2: "Stop Alerting" — dismiss immediately, clip goes to history
  const handleStopAlerting = useCallback(
    (alertId: string, eventId: string) => {
      dismissedEventsRef.current.add(eventId);
      setActiveAlerts((prev) => prev.filter((a) => a.id !== alertId));
      onDismiss?.(eventId);
    },
    [onDismiss],
  );

  // Clean up old dismissed events after 10 minutes
  useEffect(() => {
    const interval = setInterval(() => {
      if (dismissedEventsRef.current.size > 50) {
        dismissedEventsRef.current.clear();
      }
    }, 600000);
    return () => clearInterval(interval);
  }, []);

  // Don't render anything if no active alerts
  if (activeAlerts.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-[100] flex flex-col gap-3 max-w-sm w-full pointer-events-none">
      <AnimatePresence mode="popLayout">
        {activeAlerts.map((alertItem) => {
          const isAcknowledged = alertItem.state === "acknowledged";
          const violenceEnded =
            alertItem.alert.type === "violence_alert" ||
            alertItem.alert.type === "event_end";

          return (
            <motion.div
              key={alertItem.id}
              initial={{ opacity: 0, x: 100, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 100, scale: 0.9 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className="pointer-events-auto"
            >
              <div
                className={`backdrop-blur-lg border rounded-xl shadow-2xl overflow-hidden transition-colors duration-500 ${isAcknowledged
                  ? "bg-amber-950/90 border-amber-500/40 shadow-amber-500/10"
                  : "bg-red-950/95 border-red-500/50 shadow-red-500/20"
                  }`}
              >
                {/* Top pulsing bar */}
                <div
                  className={`h-1 ${isAcknowledged
                    ? "bg-gradient-to-r from-amber-500 via-yellow-500 to-amber-500"
                    : "bg-gradient-to-r from-red-500 via-orange-500 to-red-500 animate-pulse"
                    }`}
                />

                <div className="p-4">
                  {/* Header */}
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center ${isAcknowledged
                          ? "bg-amber-500/20"
                          : "bg-red-500/20 animate-pulse"
                          }`}
                      >
                        {isAcknowledged ? (
                          <Eye className="w-4 h-4 text-amber-400" />
                        ) : (
                          <AlertTriangle className="w-4 h-4 text-red-400" />
                        )}
                      </div>
                      <div>
                        <h4
                          className={`text-sm font-semibold ${isAcknowledged
                            ? "text-amber-300"
                            : "text-red-300"
                            }`}
                        >
                          {isAcknowledged
                            ? violenceEnded
                              ? "Violence Ended — Clip Saved"
                              : "Monitoring — Violence Active"
                            : "⚠ Violence Detected"}
                        </h4>
                        <div className="flex items-center gap-1 mt-0.5">
                          <Radio className="w-3 h-3 text-red-400" />
                          <span className="text-xs text-slate-400">
                            {alertItem.alert.stream_name}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Details */}
                  <div className="mt-3 space-y-2">
                    {alertItem.alert.message && (
                      <p className="text-xs text-slate-300">
                        {alertItem.alert.message}
                      </p>
                    )}

                    <div className="flex items-center gap-4 text-xs text-slate-400">
                      {(alertItem.alert.max_confidence ||
                        alertItem.alert.confidence) && (
                          <span className="flex items-center gap-1">
                            <span className="text-red-400 font-medium">
                              {(
                                (alertItem.alert.max_confidence ||
                                  alertItem.alert.confidence ||
                                  0) * 100
                              ).toFixed(0)}
                              %
                            </span>
                            confidence
                          </span>
                        )}
                      {alertItem.alert.clip_duration && (
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {alertItem.alert.clip_duration.toFixed(0)}s clip
                        </span>
                      )}
                    </div>

                    {/* Confidence bar */}
                    {(alertItem.alert.max_confidence ||
                      alertItem.alert.confidence) && (
                        <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${isAcknowledged
                              ? "bg-gradient-to-r from-amber-500 to-yellow-400"
                              : "bg-gradient-to-r from-red-500 to-orange-400"
                              }`}
                            style={{
                              width: `${(alertItem.alert.max_confidence || alertItem.alert.confidence || 0) * 100}%`,
                            }}
                          />
                        </div>
                      )}
                  </div>

                  {/* Action buttons */}
                  <div className="mt-3 flex gap-2">
                    {!isAcknowledged ? (
                      <>
                        {/* Option 1: Acknowledged / Proceeding */}
                        <button
                          onClick={() => handleAcknowledge(alertItem.id)}
                          className="flex-1 text-xs bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 py-2 px-3 rounded-lg transition-colors flex items-center justify-center gap-1.5 font-medium"
                        >
                          <CheckCircle className="w-3.5 h-3.5" />
                          Acknowledged
                        </button>

                        {/* Option 2: Stop Alerting */}
                        <button
                          onClick={() =>
                            handleStopAlerting(
                              alertItem.id,
                              alertItem.alert.event_id,
                            )
                          }
                          className="flex-1 text-xs bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 py-2 px-3 rounded-lg transition-colors flex items-center justify-center gap-1.5 font-medium"
                        >
                          <BellOff className="w-3.5 h-3.5" />
                          Stop Alerting
                        </button>
                      </>
                    ) : violenceEnded ? (
                      <>
                        {/* Violence ended — show View History button */}
                        <button
                          onClick={() => {
                            onViewHistory?.();
                            handleStopAlerting(
                              alertItem.id,
                              alertItem.alert.event_id,
                            );
                          }}
                          className="flex-1 text-xs bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 py-2 px-3 rounded-lg transition-colors flex items-center justify-center gap-1.5 font-medium"
                        >
                          <Play className="w-3.5 h-3.5" />
                          View Clip in History
                        </button>
                        <button
                          onClick={() =>
                            handleStopAlerting(
                              alertItem.id,
                              alertItem.alert.event_id,
                            )
                          }
                          className="text-xs bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 py-2 px-3 rounded-lg transition-colors flex items-center justify-center gap-1.5"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </>
                    ) : (
                      /* Acknowledged + violence still active — show monitoring state */
                      <div className="flex-1 flex items-center justify-center gap-2 text-xs text-amber-400/80 py-2">
                        <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                        <span>Monitoring — awaiting event end...</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
