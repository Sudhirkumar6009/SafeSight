"use client";

import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  subMessage?: string;
}

export default function LoadingOverlay({
  isLoading,
  message = "Loading",
  subMessage = "Please wait...",
}: LoadingOverlayProps) {
  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center"
        >
          {/* Blur Background Overlay */}
          <div className="absolute inset-0 bg-secondary-900/80 backdrop-blur-sm" />

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="relative flex flex-col items-center gap-6 z-10"
          >
            {/* Animated Logo */}
            <div className="relative">
              {/* Spinning Logo */}
              <motion.div
                animate={{
                  rotate: [0, 360],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "linear",
                }}
                className="relative w-20 h-20"
              >
                <Image
                  src="/assets/logo.png"
                  alt="SafeSight Logo"
                  fill
                  className="object-contain"
                  priority
                />
              </motion.div>

              {/* Pulse rings around logo */}
              <motion.div
                animate={{ scale: [1, 1.8], opacity: [0.6, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
                className="absolute inset-0 w-20 h-20 border-4 border-primary-500 rounded-full"
              />
              <motion.div
                animate={{ scale: [1, 2], opacity: [0.4, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut", delay: 0.3 }}
                className="absolute inset-0 w-20 h-20 border-2 border-primary-400 rounded-full"
              />
            </div>

            {/* Loading Text */}
            <div className="text-center">
              <h2 className="text-xl font-bold text-white mb-1">{message}</h2>
              <p className="text-secondary-400 text-sm">{subMessage}</p>
            </div>

            {/* Animated Dots */}
            <div className="flex gap-2">
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  animate={{ y: [0, -8, 0] }}
                  transition={{
                    duration: 0.6,
                    repeat: Infinity,
                    delay: i * 0.15,
                  }}
                  className="w-2 h-2 bg-primary-500 rounded-full"
                />
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
