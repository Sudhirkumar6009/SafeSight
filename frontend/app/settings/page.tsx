/**
 * SafeSight - Settings Page
 * =============================
 * Configure system settings and detection thresholds.
 */

"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Navbar } from "@/components";

interface DetectionSettings {
  threshold: number;
  minConsecutiveFrames: number;
  cooldownSeconds: number;
  clipBeforeSeconds: number;
  clipAfterSeconds: number;
}

interface MLSettings {
  batchSize: number;
  targetFps: number;
  skipFrames: number;
}

export default function SettingsPage() {
  const [saved, setSaved] = useState(false);
  const [detection, setDetection] = useState<DetectionSettings>({
    threshold: 0.65,
    minConsecutiveFrames: 5,
    cooldownSeconds: 10,
    clipBeforeSeconds: 5,
    clipAfterSeconds: 10,
  });

  const [ml, setMl] = useState<MLSettings>({
    batchSize: 16,
    targetFps: 5,
    skipFrames: 6,
  });

  const handleDetectionChange = useCallback(
    (field: keyof DetectionSettings, value: number) => {
      setDetection((prev) => ({ ...prev, [field]: value }));
      setSaved(false);
    },
    [],
  );

  const handleMlChange = useCallback(
    (field: keyof MLSettings, value: number) => {
      setMl((prev) => ({ ...prev, [field]: value }));
      setSaved(false);
    },
    [],
  );

  const handleSave = useCallback(() => {
    // In production, this would save to backend
    console.log("Saving settings:", { detection, ml });
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  }, [detection, ml]);

  const handleReset = useCallback(() => {
    setDetection({
      threshold: 0.65,
      minConsecutiveFrames: 5,
      cooldownSeconds: 10,
      clipBeforeSeconds: 5,
      clipAfterSeconds: 10,
    });
    setMl({
      batchSize: 16,
      targetFps: 5,
      skipFrames: 6,
    });
    setSaved(false);
  }, []);

  return (
    <div className="min-h-screen bg-gray-950">
      <Navbar />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">Settings</h1>
          <p className="text-gray-400 mt-1">
            Configure detection thresholds and inference parameters
          </p>
        </div>

        {/* Saved Toast */}
        {saved && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-green-500/20 border border-green-500/30 rounded-lg flex items-center gap-3"
          >
            <svg
              className="w-5 h-5 text-green-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
            <span className="text-green-400">Settings saved successfully</span>
          </motion.div>
        )}

        <div className="space-y-8">
          {/* Detection Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-800 rounded-xl p-6"
          >
            <h2 className="text-xl font-semibold text-white mb-6">
              Event Detection
            </h2>

            <div className="space-y-6">
              {/* Threshold */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-300">
                    Violence Threshold
                  </label>
                  <span className="text-sm text-blue-400 font-mono">
                    {(detection.threshold * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0.3}
                  max={0.95}
                  step={0.05}
                  value={detection.threshold}
                  onChange={(e) =>
                    handleDetectionChange(
                      "threshold",
                      parseFloat(e.target.value),
                    )
                  }
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Minimum confidence score to trigger an event. Lower = more
                  sensitive.
                </p>
              </div>

              {/* Min Consecutive Frames */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-300">
                    Minimum Consecutive Frames
                  </label>
                  <span className="text-sm text-blue-400 font-mono">
                    {detection.minConsecutiveFrames}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={15}
                  step={1}
                  value={detection.minConsecutiveFrames}
                  onChange={(e) =>
                    handleDetectionChange(
                      "minConsecutiveFrames",
                      parseInt(e.target.value),
                    )
                  }
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Number of consecutive frames above threshold to confirm an
                  event.
                </p>
              </div>

              {/* Cooldown */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-300">
                    Cooldown Period
                  </label>
                  <span className="text-sm text-blue-400 font-mono">
                    {detection.cooldownSeconds}s
                  </span>
                </div>
                <input
                  type="range"
                  min={5}
                  max={60}
                  step={5}
                  value={detection.cooldownSeconds}
                  onChange={(e) =>
                    handleDetectionChange(
                      "cooldownSeconds",
                      parseInt(e.target.value),
                    )
                  }
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Seconds to wait after an event before allowing a new one from
                  the same stream.
                </p>
              </div>

              {/* Clip Before */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium text-gray-300">
                      Clip Before
                    </label>
                    <span className="text-sm text-blue-400 font-mono">
                      {detection.clipBeforeSeconds}s
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={30}
                    step={1}
                    value={detection.clipBeforeSeconds}
                    onChange={(e) =>
                      handleDetectionChange(
                        "clipBeforeSeconds",
                        parseInt(e.target.value),
                      )
                    }
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Seconds of footage to include before event start.
                  </p>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium text-gray-300">
                      Clip After
                    </label>
                    <span className="text-sm text-blue-400 font-mono">
                      {detection.clipAfterSeconds}s
                    </span>
                  </div>
                  <input
                    type="range"
                    min={5}
                    max={60}
                    step={1}
                    value={detection.clipAfterSeconds}
                    onChange={(e) =>
                      handleDetectionChange(
                        "clipAfterSeconds",
                        parseInt(e.target.value),
                      )
                    }
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Seconds of footage to include after event ends.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* ML/Inference Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-gray-800 rounded-xl p-6"
          >
            <h2 className="text-xl font-semibold text-white mb-6">
              Inference Settings
            </h2>

            <div className="space-y-6">
              {/* Target FPS */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-300">
                    Target Inference FPS
                  </label>
                  <span className="text-sm text-blue-400 font-mono">
                    {ml.targetFps} fps
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={30}
                  step={1}
                  value={ml.targetFps}
                  onChange={(e) =>
                    handleMlChange("targetFps", parseInt(e.target.value))
                  }
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Frames per second to send to ML service. Higher = more CPU
                  usage.
                </p>
              </div>

              {/* Skip Frames */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-300">
                    Frame Skip
                  </label>
                  <span className="text-sm text-blue-400 font-mono">
                    {ml.skipFrames}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={30}
                  step={1}
                  value={ml.skipFrames}
                  onChange={(e) =>
                    handleMlChange("skipFrames", parseInt(e.target.value))
                  }
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Process every Nth frame. Higher = faster but less accurate.
                </p>
              </div>

              {/* Batch Size */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-300">
                    Batch Size
                  </label>
                  <span className="text-sm text-blue-400 font-mono">
                    {ml.batchSize}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={64}
                  step={1}
                  value={ml.batchSize}
                  onChange={(e) =>
                    handleMlChange("batchSize", parseInt(e.target.value))
                  }
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Number of frames to process in each batch. Higher = more
                  efficient on GPU.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Info Box */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4"
          >
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-blue-400 mt-0.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div>
                <p className="text-blue-400 font-medium">Note</p>
                <p className="text-blue-300/70 text-sm mt-1">
                  Some settings require restarting streams to take effect. For
                  production deployments, configure these values via environment
                  variables.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-4">
            <button
              onClick={handleReset}
              className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
            >
              Reset to Defaults
            </button>
            <button
              onClick={handleSave}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              Save Settings
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
