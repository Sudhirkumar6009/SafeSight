"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import { Shield, Loader2, CheckCircle, XCircle } from "lucide-react";
import { authService } from "@/services/authService";
import { useAuth } from "@/hooks/useAuth";

export default function AuthCallbackPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { refreshUser } = useAuth();
  const [status, setStatus] = useState<"loading" | "success" | "error">(
    "loading",
  );
  const [message, setMessage] = useState("Processing authentication...");

  useEffect(() => {
    const handleCallback = async () => {
      const token = searchParams.get("token");
      const error = searchParams.get("error");

      if (error) {
        setStatus("error");
        setMessage("Authentication failed. Please try again.");
        setTimeout(() => router.push("/login?error=auth_failed"), 2000);
        return;
      }

      if (token) {
        try {
          // Decode token to get user info (basic JWT decode)
          const payloadBase64 = token.split(".")[1];
          const payload = JSON.parse(atob(payloadBase64));

          // Store auth data
          authService.setAuth(token, {
            id: payload.id,
            email: payload.email,
            username: payload.username,
            provider: "google",
          });

          // Update React context with the user data
          await refreshUser();

          setStatus("success");
          setMessage("Login successful! Redirecting...");
          setTimeout(() => router.push("/dashboard"), 1500);
        } catch (err) {
          setStatus("error");
          setMessage("Failed to process token. Please try again.");
          setTimeout(() => router.push("/login?error=token_error"), 2000);
        }
      } else {
        setStatus("error");
        setMessage("No authentication token received.");
        setTimeout(() => router.push("/login"), 2000);
      }
    };

    handleCallback();
  }, [searchParams, router, refreshUser]);

  return (
    <div className="min-h-screen bg-black flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="text-center"
      >
        {/* Logo */}
        <div className="inline-flex items-center gap-3 mb-8">
          <div className="w-12 h-12 bg-cyan-500/20 rounded-xl flex items-center justify-center">
            <Shield className="w-6 h-6 text-cyan-400" />
          </div>
          <span className="text-2xl font-bold text-white">
            Safe<span className="text-cyan-400">Sight</span>
          </span>
        </div>

        {/* Status Icon */}
        <div className="mb-6">
          {status === "loading" && (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              className="w-16 h-16 mx-auto"
            >
              <Loader2 className="w-16 h-16 text-cyan-400" />
            </motion.div>
          )}
          {status === "success" && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="w-16 h-16 mx-auto bg-emerald-500/20 rounded-full flex items-center justify-center"
            >
              <CheckCircle className="w-10 h-10 text-emerald-400" />
            </motion.div>
          )}
          {status === "error" && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="w-16 h-16 mx-auto bg-red-500/20 rounded-full flex items-center justify-center"
            >
              <XCircle className="w-10 h-10 text-red-400" />
            </motion.div>
          )}
        </div>

        {/* Message */}
        <p className="text-lg text-gray-300">{message}</p>
      </motion.div>
    </div>
  );
}
