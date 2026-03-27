"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Cpu,
  FolderOpen,
  CheckCircle,
  AlertCircle,
  Loader2,
  Settings,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { apiService } from "@/services/api";
import { useAppStore } from "@/hooks/useStore";

interface ModelLoaderProps {
  onModelLoaded?: () => void;
}

const architectures = [
  {
    value: "videomae",
    label: "VideoMAE",
    description: "Masked Autoencoder for Video",
  },
  {
    value: "timesformer",
    label: "TimeSformer",
    description: "Temporal Transformer",
  },
  {
    value: "slowfast",
    label: "SlowFast",
    description: "Two-pathway Architecture",
  },
  {
    value: "resnet3d",
    label: "3D ResNet",
    description: "3D Convolutional Network",
  },
  { value: "i3d", label: "I3D", description: "Inflated 3D ConvNet" },
  { value: "custom", label: "Custom", description: "Custom Architecture" },
];

export default function ModelLoader({ onModelLoaded }: ModelLoaderProps) {
  const router = useRouter();
  const [modelPath, setModelPath] = useState("");
  const [architecture, setArchitecture] = useState("videomae");
  const [modelName, setModelName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const { showNotification, setModelStatus } = useAppStore();

  const handleLoadModel = async () => {
    if (!modelPath.trim()) {
      setError("Please enter a model path");
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await apiService.loadModel({
        modelPath: modelPath.trim(),
        architecture,
        name: modelName.trim() || undefined,
      });

      if (response.success) {
        setSuccess(true);
        showNotification("success", "Model loaded successfully!");

        // Refresh model status
        const statusResponse = await apiService.getModelStatus();
        if (statusResponse.success && statusResponse.data) {
          setModelStatus(statusResponse.data);
        }

        onModelLoaded?.();
        router.push("/videos");
      } else {
        throw new Error(response.error || "Failed to load model");
      }
    } catch (err: any) {
      setError(
        err.response?.data?.error || err.message || "Failed to load model",
      );
      showNotification("error", err.message || "Failed to load model");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl p-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-primary-500/20 flex items-center justify-center">
          <Cpu className="w-6 h-6 text-primary-400" />
        </div>
        <div>
          <h3 className="text-xl font-semibold text-white">Load Model</h3>
          <p className="text-sm text-dark-400">
            Configure and load your PyTorch model
          </p>
        </div>
      </div>

      <div className="space-y-5">
        {/* Model Path Input */}
        <div>
          <label className="block text-sm font-medium text-dark-300 mb-2">
            Model Path (.pth)
          </label>
          <div className="relative">
            <FolderOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-500" />
            <input
              type="text"
              value={modelPath}
              onChange={(e) => setModelPath(e.target.value)}
              placeholder="/path/to/your/model.pth"
              className="w-full pl-11 pr-4 py-3 bg-dark-800 border border-dark-600 rounded-xl text-white placeholder:text-dark-500 focus:outline-none focus:border-primary-500 transition-colors"
            />
          </div>
          <p className="mt-1.5 text-xs text-dark-500">
            Enter the absolute path to your PyTorch model file
          </p>
        </div>

        {/* Model Name Input */}
        <div>
          <label className="block text-sm font-medium text-dark-300 mb-2">
            Model Name (Optional)
          </label>
          <div className="relative">
            <Settings className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-500" />
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="My Violence Detection Model"
              className="w-full pl-11 pr-4 py-3 bg-dark-800 border border-dark-600 rounded-xl text-white placeholder:text-dark-500 focus:outline-none focus:border-primary-500 transition-colors"
            />
          </div>
        </div>

        {/* Architecture Selection */}
        <div>
          <label className="block text-sm font-medium text-dark-300 mb-2">
            Model Architecture
          </label>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {architectures.map((arch) => (
              <button
                key={arch.value}
                onClick={() => setArchitecture(arch.value)}
                className={cn(
                  "p-3 rounded-xl border text-left transition-all",
                  architecture === arch.value
                    ? "border-primary-500 bg-primary-500/10"
                    : "border-dark-600 bg-dark-800/50 hover:border-dark-500",
                )}
              >
                <div
                  className={cn(
                    "font-medium",
                    architecture === arch.value
                      ? "text-primary-400"
                      : "text-white",
                  )}
                >
                  {arch.label}
                </div>
                <div className="text-xs text-dark-500 mt-0.5">
                  {arch.description}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Status Messages */}
        {error && (
          <div className="flex items-center gap-3 p-4 bg-danger-500/10 border border-danger-500/30 rounded-xl">
            <AlertCircle className="w-5 h-5 text-danger-500 flex-shrink-0" />
            <span className="text-danger-400 text-sm">{error}</span>
          </div>
        )}

        {success && (
          <div className="flex items-center gap-3 p-4 bg-success-500/10 border border-success-500/30 rounded-xl">
            <CheckCircle className="w-5 h-5 text-success-500 flex-shrink-0" />
            <span className="text-success-400 text-sm">
              Model loaded successfully!
            </span>
          </div>
        )}

        {/* Load Button */}
        <button
          onClick={handleLoadModel}
          disabled={isLoading || !modelPath.trim()}
          className={cn(
            "w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-medium transition-all",
            isLoading || !modelPath.trim()
              ? "bg-dark-700 text-dark-500 cursor-not-allowed"
              : "bg-primary-600 hover:bg-primary-700 text-white glow-primary",
          )}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Loading Model...
            </>
          ) : (
            <>
              <Zap className="w-5 h-5" />
              Load Model
            </>
          )}
        </button>
      </div>
    </motion.div>
  );
}
