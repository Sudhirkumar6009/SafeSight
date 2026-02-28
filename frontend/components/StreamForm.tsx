/**
 * SafeSight - Stream Form Component
 * =====================================
 * Form for adding or editing RTSP streams.
 */

"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Stream, StreamCreateRequest } from "@/types";

interface StreamFormProps {
  stream?: Stream | null;
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: StreamCreateRequest) => Promise<void>;
}

export function StreamForm({
  stream,
  isOpen,
  onClose,
  onSubmit,
}: StreamFormProps) {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState<StreamCreateRequest>({
    name: "",
    rtsp_url: "",
    location: "",
    inference_enabled: true,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Initialize form data when stream changes
  useEffect(() => {
    if (stream) {
      setFormData({
        name: stream.name,
        rtsp_url: stream.rtsp_url,
        location: stream.location || "",
        inference_enabled: stream.inference_enabled ?? true,
      });
    } else {
      setFormData({
        name: "",
        rtsp_url: "",
        location: "",
        inference_enabled: true,
      });
    }
    setErrors({});
  }, [stream, isOpen]);

  const validate = useCallback(() => {
    const newErrors: Record<string, string> = {};

    if (!formData.name.trim()) {
      newErrors.name = "Name is required";
    } else if (formData.name.length > 100) {
      newErrors.name = "Name must be less than 100 characters";
    }

    const rtspUrl = formData.rtsp_url || "";
    if (!rtspUrl.trim()) {
      newErrors.rtsp_url = "RTSP URL is required";
    } else if (!rtspUrl.match(/^rtsp:\/\/.+/)) {
      newErrors.rtsp_url = "Must be a valid RTSP URL (rtsp://...)";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      if (!validate()) return;

      setLoading(true);
      try {
        await onSubmit(formData);
        onClose();
      } catch (error: any) {
        setErrors({ submit: error.message || "Failed to save stream" });
      } finally {
        setLoading(false);
      }
    },
    [formData, validate, onSubmit, onClose],
  );

  const handleChange = useCallback(
    (field: keyof StreamCreateRequest, value: any) => {
      setFormData((prev) => ({ ...prev, [field]: value }));
      setErrors((prev) => ({ ...prev, [field]: "" }));
    },
    [],
  );

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

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 flex items-center justify-center z-50 p-4"
          >
            <div className="w-full max-w-md bg-gray-900 rounded-xl shadow-2xl overflow-hidden">
              {/* Header */}
              <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                <h2 className="text-xl font-bold text-white">
                  {stream ? "Edit Stream" : "Add Stream"}
                </h2>
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

              {/* Form */}
              <form onSubmit={handleSubmit} className="p-4 space-y-4">
                {/* Name */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Stream Name *
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => handleChange("name", e.target.value)}
                    placeholder="e.g., Front Entrance Camera"
                    className={`w-full px-3 py-2 bg-gray-800 border rounded-lg text-white
                    placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500
                    ${errors.name ? "border-red-500" : "border-gray-700"}`}
                  />
                  {errors.name && (
                    <p className="text-red-400 text-sm mt-1">{errors.name}</p>
                  )}
                </div>

                {/* RTSP URL */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    RTSP URL *
                  </label>
                  <input
                    type="text"
                    value={formData.rtsp_url || ""}
                    onChange={(e) =>
                      handleChange(
                        "rtsp_url" as keyof StreamCreateRequest,
                        e.target.value,
                      )
                    }
                    placeholder="rtsp://username:password@ip:port/stream"
                    className={`w-full px-3 py-2 bg-gray-800 border rounded-lg text-white font-mono text-sm
                    placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500
                    ${errors.rtsp_url ? "border-red-500" : "border-gray-700"}`}
                  />
                  {errors.rtsp_url && (
                    <p className="text-red-400 text-sm mt-1">
                      {errors.rtsp_url}
                    </p>
                  )}
                  <p className="text-gray-500 text-xs mt-1">
                    Examples: rtsp://192.168.1.100:554/stream1,
                    rtsp://admin:pass@camera.local/live
                  </p>
                </div>

                {/* Location */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Location (optional)
                  </label>
                  <input
                    type="text"
                    value={formData.location}
                    onChange={(e) => handleChange("location", e.target.value)}
                    placeholder="e.g., Building A, Floor 2"
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white
                    placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {/* Inference Enabled */}
                <div className="flex items-center gap-3">
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formData.inference_enabled ?? true}
                      onChange={(e) =>
                        handleChange(
                          "inference_enabled" as keyof StreamCreateRequest,
                          e.target.checked,
                        )
                      }
                      className="sr-only peer"
                    />
                    <div
                      className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-2 
                                  peer-focus:ring-blue-500 rounded-full peer 
                                  peer-checked:after:translate-x-full peer-checked:after:border-white 
                                  after:content-[''] after:absolute after:top-[2px] after:left-[2px] 
                                  after:bg-white after:rounded-full after:h-5 after:w-5 
                                  after:transition-all peer-checked:bg-blue-600"
                    ></div>
                  </label>
                  <div>
                    <span className="text-sm font-medium text-gray-300">
                      Enable Inference
                    </span>
                    <p className="text-xs text-gray-500">
                      Run violence detection on this stream
                    </p>
                  </div>
                </div>

                {/* Submit Error */}
                {errors.submit && (
                  <div className="p-3 bg-red-500/20 border border-red-500/30 rounded-lg">
                    <p className="text-red-400 text-sm">{errors.submit}</p>
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center justify-end gap-3 pt-4">
                  <button
                    type="button"
                    onClick={onClose}
                    disabled={loading}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white 
                             rounded-lg font-medium transition-colors disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={loading}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white 
                             rounded-lg font-medium transition-colors disabled:opacity-50"
                  >
                    {loading
                      ? "Saving..."
                      : stream
                        ? "Save Changes"
                        : "Add Stream"}
                  </button>
                </div>
              </form>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
