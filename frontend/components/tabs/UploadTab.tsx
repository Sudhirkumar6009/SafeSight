"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Video,
  Brain,
  ArrowRight,
  CheckCircle,
  Upload,
  ShieldAlert,
  Shield,
  Car,
  AlertTriangle,
  Sparkles,
  ChevronLeft,
} from "lucide-react";
import { VideoUpload } from "@/components";
import { apiService } from "@/services/api";
import { Prediction, AccidentPrediction } from "@/types";
import { useAppStore } from "@/hooks/useStore";
import { cn, formatPercentage } from "@/lib/utils";

type DetectionType = "violence" | "accident" | null;
type AnalysisResult = {
  type: DetectionType;
  prediction: Prediction | AccidentPrediction | null;
};

export default function UploadTab() {
  const [selectedType, setSelectedType] = useState<DetectionType>(null);
  const [uploadedVideoId, setUploadedVideoId] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { showNotification } = useAppStore();

  const handleUploadComplete = (videoId: string, fileName?: string) => {
    setUploadedVideoId(videoId);
    setUploadedFileName(fileName || "video");
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!uploadedVideoId || !selectedType) return;
    setIsAnalyzing(true);
    setError(null);

    try {
      if (selectedType === "violence") {
        const response = await apiService.runInference(uploadedVideoId);
        if (response.success && response.data) {
          setResult({ type: "violence", prediction: response.data });
          showNotification("success", "Violence analysis completed!");
        } else {
          throw new Error(response.error || "Analysis failed");
        }
      } else {
        const response = await apiService.runAccidentInference(uploadedVideoId);
        if (response.success && response.data) {
          setResult({ type: "accident", prediction: response.data });
          showNotification("success", "Accident analysis completed!");
        } else {
          throw new Error(response.error || "Analysis failed");
        }
      }
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.error || err.message || "Analysis failed";
      setError(errorMessage);
      showNotification("error", errorMessage);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedType(null);
    setUploadedVideoId(null);
    setUploadedFileName(null);
    setResult(null);
    setError(null);
  };

  const handleBack = () => {
    if (result) {
      setResult(null);
    } else if (uploadedVideoId) {
      setUploadedVideoId(null);
      setUploadedFileName(null);
    } else {
      setSelectedType(null);
    }
  };

  // Violence result rendering
  const isViolent =
    result?.type === "violence" &&
    (result.prediction as Prediction)?.classification === "violence";

  // Accident result rendering
  const isAccident =
    result?.type === "accident" &&
    (result.prediction as AccidentPrediction)?.classification === "accident";

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <div className="flex items-center justify-center gap-2 mb-3">
          <Sparkles className="w-6 h-6 text-cyan-400" />
          <h1 className="text-3xl font-bold text-white">AI Video Analysis</h1>
          <Sparkles className="w-6 h-6 text-cyan-400" />
        </div>
        <p className="text-slate-400">
          Upload a video and analyze it using our advanced AI models
        </p>
      </motion.div>

      {/* Back Button */}
      {(selectedType || uploadedVideoId || result) && (
        <motion.button
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={handleBack}
          className="flex items-center gap-2 text-slate-400 hover:text-white mb-6 transition-colors"
        >
          <ChevronLeft className="w-5 h-5" />
          Back
        </motion.button>
      )}

      <AnimatePresence mode="wait">
        {/* Step 1: Select Detection Type */}
        {!selectedType && !result && (
          <motion.div
            key="select-type"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="grid md:grid-cols-2 gap-6"
          >
            {/* Violence Detection Card */}
            <motion.button
              whileHover={{ scale: 1.02, y: -5 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setSelectedType("violence")}
              className="group relative overflow-hidden bg-gradient-to-br from-red-500/10 via-slate-900 to-slate-900 border-2 border-red-500/30 hover:border-red-500/60 rounded-2xl p-8 text-left transition-all duration-300"
            >
              {/* Glow effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

              {/* Icon */}
              <div className="relative mb-6">
                <div className="w-20 h-20 rounded-2xl bg-red-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <ShieldAlert className="w-10 h-10 text-red-400" />
                </div>
                <div className="absolute -top-1 -right-1 w-6 h-6 rounded-full bg-red-500 flex items-center justify-center">
                  <Brain className="w-3.5 h-3.5 text-white" />
                </div>
              </div>

              {/* Content */}
              <h2 className="text-2xl font-bold text-white mb-3 group-hover:text-red-400 transition-colors">
                Violence Detection
              </h2>
              <p className="text-slate-400 mb-6 leading-relaxed">
                Analyze videos for violent content, fights, assaults, and
                aggressive behavior using our trained AI model.
              </p>

              {/* Features */}
              <div className="flex flex-wrap gap-2 mb-6">
                {["16 Frame Analysis", "Real-time", "High Accuracy"].map(
                  (feature) => (
                    <span
                      key={feature}
                      className="px-3 py-1 rounded-full bg-red-500/10 text-red-400 text-xs font-medium"
                    >
                      {feature}
                    </span>
                  ),
                )}
              </div>

              {/* CTA */}
              <div className="flex items-center gap-2 text-red-400 font-semibold group-hover:gap-4 transition-all">
                <span>Start Detection</span>
                <ArrowRight className="w-5 h-5" />
              </div>
            </motion.button>

            {/* Accident Detection Card */}
            <motion.button
              whileHover={{ scale: 1.02, y: -5 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setSelectedType("accident")}
              className="group relative overflow-hidden bg-gradient-to-br from-amber-500/10 via-slate-900 to-slate-900 border-2 border-amber-500/30 hover:border-amber-500/60 rounded-2xl p-8 text-left transition-all duration-300"
            >
              {/* Glow effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

              {/* Icon */}
              <div className="relative mb-6">
                <div className="w-20 h-20 rounded-2xl bg-amber-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Car className="w-10 h-10 text-amber-400" />
                </div>
                <div className="absolute -top-1 -right-1 w-6 h-6 rounded-full bg-amber-500 flex items-center justify-center">
                  <Brain className="w-3.5 h-3.5 text-white" />
                </div>
              </div>

              {/* Content */}
              <h2 className="text-2xl font-bold text-white mb-3 group-hover:text-amber-400 transition-colors">
                Accident Detection
              </h2>
              <p className="text-slate-400 mb-6 leading-relaxed">
                Detect vehicle accidents, collisions, and traffic incidents in
                video footage using our specialized AI model.
              </p>

              {/* Features */}
              <div className="flex flex-wrap gap-2 mb-6">
                {["16 Frame Analysis", "Real-time", "ONNX Model"].map(
                  (feature) => (
                    <span
                      key={feature}
                      className="px-3 py-1 rounded-full bg-amber-500/10 text-amber-400 text-xs font-medium"
                    >
                      {feature}
                    </span>
                  ),
                )}
              </div>

              {/* CTA */}
              <div className="flex items-center gap-2 text-amber-400 font-semibold group-hover:gap-4 transition-all">
                <span>Start Detection</span>
                <ArrowRight className="w-5 h-5" />
              </div>
            </motion.button>
          </motion.div>
        )}

        {/* Step 2: Upload Video */}
        {selectedType && !uploadedVideoId && !result && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            {/* Selected Type Badge */}
            <div className="flex justify-center mb-6">
              <div
                className={cn(
                  "inline-flex items-center gap-2 px-4 py-2 rounded-full",
                  selectedType === "violence"
                    ? "bg-red-500/20 text-red-400"
                    : "bg-amber-500/20 text-amber-400",
                )}
              >
                {selectedType === "violence" ? (
                  <ShieldAlert className="w-5 h-5" />
                ) : (
                  <Car className="w-5 h-5" />
                )}
                <span className="font-semibold">
                  {selectedType === "violence"
                    ? "Violence Detection"
                    : "Accident Detection"}
                </span>
              </div>
            </div>

            <VideoUpload onUploadComplete={handleUploadComplete} />
          </motion.div>
        )}

        {/* Step 3: Uploaded - Ready to Analyze */}
        {uploadedVideoId && !result && !isAnalyzing && (
          <motion.div
            key="ready"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className={cn(
              "bg-slate-900 border-2 rounded-2xl p-8 text-center",
              selectedType === "violence"
                ? "border-red-500/30"
                : "border-amber-500/30",
            )}
          >
            <div className="flex items-center justify-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-400" />
              </div>
              <div className="text-left">
                <h3 className="text-lg font-semibold text-white">
                  Video Uploaded Successfully!
                </h3>
                <p className="text-sm text-slate-400">{uploadedFileName}</p>
              </div>
            </div>

            {error && (
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 mb-6">
                {error}
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className={cn(
                "inline-flex items-center gap-3 px-10 py-5 rounded-xl font-bold text-lg shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed mb-4",
                selectedType === "violence"
                  ? "bg-red-600 hover:bg-red-700 text-white"
                  : "bg-amber-600 hover:bg-amber-700 text-white",
              )}
            >
              <Brain className="w-7 h-7" />
              {selectedType === "violence"
                ? "Detect Violence"
                : "Detect Accident"}
              <ArrowRight className="w-6 h-6" />
            </button>

            <p className="text-slate-500 text-sm mb-6">
              Click to analyze this video using AI
            </p>

            <button
              onClick={handleReset}
              className="inline-flex items-center gap-2 px-4 py-2 text-slate-400 hover:text-white transition-colors"
            >
              <Upload className="w-4 h-4" />
              Start Over
            </button>
          </motion.div>
        )}

        {/* Analyzing State */}
        {isAnalyzing && (
          <motion.div
            key="analyzing"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className={cn(
              "bg-slate-900 border-2 rounded-2xl p-12 text-center",
              selectedType === "violence"
                ? "border-red-500/30"
                : "border-amber-500/30",
            )}
          >
            <div className="relative inline-flex items-center justify-center w-24 h-24 mb-6">
              <div
                className={cn(
                  "absolute inset-0 rounded-full border-4",
                  selectedType === "violence"
                    ? "border-red-500/20"
                    : "border-amber-500/20",
                )}
              />
              <div
                className={cn(
                  "absolute inset-0 rounded-full border-4 border-t-transparent animate-spin",
                  selectedType === "violence"
                    ? "border-red-500"
                    : "border-amber-500",
                )}
              />
              <Brain
                className={cn(
                  "w-10 h-10",
                  selectedType === "violence"
                    ? "text-red-400"
                    : "text-amber-400",
                )}
              />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">
              {selectedType === "violence"
                ? "Detecting Violence..."
                : "Detecting Accident..."}
            </h3>
            <p className="text-slate-400 mb-2">{uploadedFileName}</p>
            <p className="text-slate-500 mb-6">
              AI model is analyzing video frames
            </p>
            <div className="flex justify-center gap-8 text-center">
              <div>
                <div
                  className={cn(
                    "text-2xl font-bold",
                    selectedType === "violence"
                      ? "text-red-400"
                      : "text-amber-400",
                  )}
                >
                  16
                </div>
                <div className="text-sm text-slate-500">Frames</div>
              </div>
              <div>
                <div
                  className={cn(
                    "text-2xl font-bold",
                    selectedType === "violence"
                      ? "text-red-400"
                      : "text-amber-400",
                  )}
                >
                  224x224
                </div>
                <div className="text-sm text-slate-500">Resolution</div>
              </div>
              <div>
                <div
                  className={cn(
                    "text-2xl font-bold",
                    selectedType === "violence"
                      ? "text-red-400"
                      : "text-amber-400",
                  )}
                >
                  ONNX
                </div>
                <div className="text-sm text-slate-500">Model</div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Violence Result */}
        {result?.type === "violence" && result.prediction && (
          <motion.div
            key="violence-result"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <div
              className={cn(
                "bg-slate-900 border-2 rounded-2xl p-8 text-center",
                isViolent ? "border-red-500/50" : "border-green-500/50",
              )}
            >
              <div
                className={cn(
                  "w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6",
                  isViolent ? "bg-red-500/20" : "bg-green-500/20",
                )}
              >
                {isViolent ? (
                  <ShieldAlert className="w-12 h-12 text-red-400" />
                ) : (
                  <Shield className="w-12 h-12 text-green-400" />
                )}
              </div>

              <h2
                className={cn(
                  "text-3xl font-bold mb-2",
                  isViolent ? "text-red-400" : "text-green-400",
                )}
              >
                {isViolent ? "VIOLENCE DETECTED" : "NON-VIOLENT"}
              </h2>
              <p className="text-slate-400 mb-4">{uploadedFileName}</p>

              <div className="mb-6">
                <p className="text-sm text-slate-500 mb-2">Confidence</p>
                <p
                  className={cn(
                    "text-4xl font-bold",
                    isViolent ? "text-red-400" : "text-green-400",
                  )}
                >
                  {formatPercentage(
                    (result.prediction as Prediction).confidence,
                  )}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto mb-8">
                <div className="bg-red-500/10 rounded-xl p-4">
                  <p className="text-sm text-slate-400 mb-1">Violence</p>
                  <p className="text-xl font-bold text-red-400">
                    {formatPercentage(
                      (result.prediction as Prediction).probabilities
                        ?.violence || 0,
                    )}
                  </p>
                </div>
                <div className="bg-green-500/10 rounded-xl p-4">
                  <p className="text-sm text-slate-400 mb-1">Non-Violence</p>
                  <p className="text-xl font-bold text-green-400">
                    {formatPercentage(
                      (result.prediction as Prediction).probabilities
                        ?.nonViolence || 0,
                    )}
                  </p>
                </div>
              </div>

              <button
                onClick={handleReset}
                className="inline-flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white rounded-xl font-medium transition-colors"
              >
                <Video className="w-5 h-5" />
                Analyze Another Video
              </button>
            </div>
          </motion.div>
        )}

        {/* Accident Result */}
        {result?.type === "accident" && result.prediction && (
          <motion.div
            key="accident-result"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <div
              className={cn(
                "bg-slate-900 border-2 rounded-2xl p-8 text-center",
                isAccident ? "border-amber-500/50" : "border-green-500/50",
              )}
            >
              <div
                className={cn(
                  "w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6",
                  isAccident ? "bg-amber-500/20" : "bg-green-500/20",
                )}
              >
                {isAccident ? (
                  <AlertTriangle className="w-12 h-12 text-amber-400" />
                ) : (
                  <Car className="w-12 h-12 text-green-400" />
                )}
              </div>

              <h2
                className={cn(
                  "text-3xl font-bold mb-2",
                  isAccident ? "text-amber-400" : "text-green-400",
                )}
              >
                {isAccident ? "ACCIDENT DETECTED" : "NORMAL VIDEO"}
              </h2>
              <p className="text-slate-400 mb-4">{uploadedFileName}</p>

              <div className="mb-6">
                <p className="text-sm text-slate-500 mb-2">Confidence</p>
                <p
                  className={cn(
                    "text-4xl font-bold",
                    isAccident ? "text-amber-400" : "text-green-400",
                  )}
                >
                  {formatPercentage(
                    (result.prediction as AccidentPrediction).confidence,
                  )}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto mb-8">
                <div className="bg-amber-500/10 rounded-xl p-4">
                  <p className="text-sm text-slate-400 mb-1">Accident</p>
                  <p className="text-xl font-bold text-amber-400">
                    {formatPercentage(
                      (result.prediction as AccidentPrediction).probabilities
                        ?.accident || 0,
                    )}
                  </p>
                </div>
                <div className="bg-green-500/10 rounded-xl p-4">
                  <p className="text-sm text-slate-400 mb-1">Normal</p>
                  <p className="text-xl font-bold text-green-400">
                    {formatPercentage(
                      (result.prediction as AccidentPrediction).probabilities
                        ?.normal || 0,
                    )}
                  </p>
                </div>
              </div>

              <button
                onClick={handleReset}
                className="inline-flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white rounded-xl font-medium transition-colors"
              >
                <Video className="w-5 h-5" />
                Analyze Another Video
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
