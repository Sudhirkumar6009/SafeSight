"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Video,
  Brain,
  Activity,
  Database,
  Cpu,
  BarChart3,
  AlertTriangle,
  CheckCircle,
  Clock,
  Play,
  Upload,
  Eye,
  Zap,
  ArrowRight,
  Terminal,
  Radio,
} from "lucide-react";
import { apiService } from "@/services/api";
import { cn } from "@/lib/utils";
import { Navbar } from "@/components";
import { useAuth } from "@/hooks/useAuth";
import { PredictionStats } from "@/types";

interface HealthStatus {
  mongodb: boolean;
  mlService: boolean;
}

export default function Home() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const [stats, setStats] = useState<PredictionStats | null>(null);
  const [isLogoHovered, setIsLogoHovered] = useState(false);
  const [isLogoSpinningOnLoad, setIsLogoSpinningOnLoad] = useState(true);
  const [health, setHealth] = useState<HealthStatus>({
    mongodb: false,
    mlService: false,
  });
  useEffect(() => {
    if (document.readyState === "complete") {
      setIsLogoSpinningOnLoad(false);
      return;
    }

    const handleWindowLoad = () => setIsLogoSpinningOnLoad(false);
    window.addEventListener("load", handleWindowLoad);

    return () => window.removeEventListener("load", handleWindowLoad);
  }, []);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, healthRes] = await Promise.all([
          apiService.getPredictionStats(),
          apiService.getHealth(),
        ]);
        if (statsRes.success && statsRes.data) setStats(statsRes.data);
        if (healthRes.success && healthRes.data) {
          setHealth({
            mongodb: healthRes.data.services?.mongodb?.status === "connected",
            mlService:
              healthRes.data.services?.mlService?.status === "connected",
          });
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const accuracy = stats?.avgConfidence
    ? (stats.avgConfidence * 100).toFixed(1)
    : "99.2";
  const avgTime = stats?.avgInferenceTime
    ? stats.avgInferenceTime.toFixed(0)
    : "12";
  const totalAnalyses = stats?.total || 0;
  const footerHref = (dashboardPath: string, publicPath: string) =>
    isAuthenticated ? dashboardPath : publicPath;

  return (
    <div className="min-h-screen bg-black">
      <Navbar />

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-6 pb-16">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-16 items-center">
          {/* Left Content */}
          <div className="lg:col-span-2">
            {/* Section Label */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-4 mb-8"
            >
              <span className="text-xs tracking-[0.3em] text-gray-500 uppercase font-medium mt-10">
                Next-Gen Security Intelligence
              </span>
              <div className="flex-1 mt-10 h-[2px] bg-gray-800" />
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-5xl lg:text-6xl font-bold text-white mb-6"
            >
              REAL-TIME
              <br />
              <span className="text-cyan-400 text-s">VIOLENCE</span>
              <br />
              DETECTION
            </motion.h1>

            {/* Description */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-gray-400 text-lg mb-10 max-w-md leading-relaxed"
            >
              AI-powered surveillance system for instant threat detection.
              Analyze video streams and uploads with deep learning precision.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="flex flex-wrap gap-4 mb-16"
            >
              <button
                onClick={() =>
                  router.push(
                    isAuthenticated
                      ? "/dashboard"
                      : "/login?redirect=/dashboard?tab=streams",
                  )
                }
                className="group flex items-center gap-3 px-6 py-3 bg-cyan-500 text-black font-semibold rounded-lg hover:bg-cyan-400 transition-all"
              >
                <Upload className="w-5 h-5" />
                Start Analysis
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>
              <button
                onClick={() =>
                  router.push(
                    isAuthenticated
                      ? "/dashboard?tab=streams"
                      : "/login?redirect=/dashboard?tab=streams",
                  )
                }
                className="flex items-center gap-3 px-6 py-3 bg-transparent text-white font-medium rounded-lg border border-gray-700 hover:border-gray-500 transition-all"
              >
                <Radio className="w-5 h-5" />
                Live Streams
              </button>
            </motion.div>

            {/* Stats Row */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="grid grid-cols-4 gap-6"
            >
              <StatItem value={`${accuracy}%`} label="Accuracy" />
              <StatItem value={`${avgTime}ms`} label="Inference" />
              <StatItem
                value={health.mlService ? "Live" : "Off"}
                label="ML Status"
                isStatus
                statusActive={health.mlService}
              />
              <StatItem value={`${totalAnalyses}`} label="Analyses" />
            </motion.div>
          </div>

          {/* Right Content - Hero Image */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="relative lg:col-span-3"
          >
            <div className="relative h-full flex items-center">
              {/* Glow Effect */}
              <div className="absolute inset-0 bg-cyan-500/10 blur-3xl rounded-full" />
              <img
                src="/assets/landing_demo.png"
                alt="SafeSight - Real-time Violence Detection System"
                className="relative w-full h-auto max-h-[700px] object-contain rounded-2xl shadow-2xl"
              />
            </div>
          </motion.div>
        </div>
      </main>

      {/* Features Section */}
      <section className="border-t border-gray-900 bg-black">
        <div className="max-w-7xl mx-auto px-6 py-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="text-xs tracking-[0.3em] text-gray-500 uppercase font-medium">
              Core Capabilities
            </span>
            <h2 className="text-3xl font-bold text-white mt-4">
              Powered by Advanced AI
            </h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard
              icon={Video}
              title="Video Analysis"
              description="Upload MP4, AVI, or MOV videos for instant violence detection with frame-by-frame analysis."
              href="/upload"
            />
            <FeatureCard
              icon={Radio}
              title="RTSP Streaming"
              description="Connect live camera feeds for real-time monitoring and instant threat alerts."
              href="/streams"
            />
            <FeatureCard
              icon={Brain}
              title="Deep Learning"
              description="State-of-the-art neural networks trained on extensive datasets for high accuracy."
              href="/model"
            />
          </div>
        </div>
      </section>

      {/* System Status Section */}
      <section className="border-t border-gray-900 bg-gray-950">
        <div className="max-w-7xl mx-auto px-6 py-16">
          <div className="flex items-center justify-between mb-8">
            <div>
              <span className="text-xs tracking-[0.3em] text-gray-500 uppercase font-medium">
                System Health
              </span>
              <h3 className="text-xl font-semibold text-white mt-2">
                Service Status
              </h3>
            </div>
            <div
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium",
                health.mongodb && health.mlService
                  ? "bg-emerald-500/10 text-emerald-400"
                  : "bg-amber-500/10 text-amber-400",
              )}
            >
              <div
                className={cn(
                  "w-2 h-2 rounded-full",
                  health.mongodb && health.mlService
                    ? "bg-emerald-500"
                    : "bg-amber-500",
                )}
              />
              {health.mongodb && health.mlService
                ? "All Systems Operational"
                : "Partial Outage"}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <SystemCard
              icon={Database}
              title="MongoDB"
              status={health.mongodb ? "Connected" : "Disconnected"}
              isActive={health.mongodb}
            />
            <SystemCard
              icon={Cpu}
              title="ML Service"
              status={health.mlService ? "Running" : "Stopped"}
              isActive={health.mlService}
            />
            <SystemCard
              icon={Activity}
              title="API Server"
              status="Healthy"
              isActive={true}
            />
          </div>
        </div>
      </section>

      {/* Quick Stats Section */}
      <section className="border-t border-gray-900 bg-black">
        <div className="max-w-7xl mx-auto px-6 py-16">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <QuickStat
              icon={BarChart3}
              value={stats?.total || 0}
              label="Total Analyses"
              color="cyan"
            />
            <QuickStat
              icon={AlertTriangle}
              value={stats?.violent || 0}
              label="Threats Detected"
              color="red"
            />
            <QuickStat
              icon={CheckCircle}
              value={stats?.nonViolent || 0}
              label="Safe Videos"
              color="green"
            />
            <QuickStat
              icon={Clock}
              value={
                stats?.avgInferenceTime
                  ? `${stats.avgInferenceTime.toFixed(2)}s`
                  : "0s"
              }
              label="Avg. Processing"
              color="purple"
            />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-cyan-500/20 bg-black">
        <div className="max-w-7xl mx-auto px-6 py-14">
          <div className="rounded-3xl border border-cyan-500/20 bg-gray-950/80 backdrop-blur-sm p-8 md:p-10">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
              {/* Brand + CTA */}
              <div className="lg:col-span-5">
                <div className="flex items-center gap-3 mb-5">
                  <motion.div
                    className="relative"
                    onHoverStart={() => setIsLogoHovered(true)}
                    onHoverEnd={() => setIsLogoHovered(false)}
                    style={{ transformOrigin: "50% 50%" }}
                    animate={
                      isLogoSpinningOnLoad || isLogoHovered
                        ? { rotate: 360 }
                        : { rotate: 0 }
                    }
                    transition={
                      isLogoSpinningOnLoad || isLogoHovered
                        ? {
                            duration: 1.1,
                            ease: "linear",
                            repeat: Infinity,
                          }
                        : {
                            type: "spring",
                            stiffness: 80,
                            damping: 16,
                          }
                    }
                  >
                    <div className="w-10 h-10 flex items-center justify-center overflow-hidden">
                      <Image
                        src="/assets/logo.png"
                        alt="SafeSight Logo"
                        width={100}
                        height={100}
                        className="object-contain"
                      />
                    </div>
                  </motion.div>
                  <div>
                    <p className="text-xs tracking-[0.25em] uppercase text-cyan-500/80">
                      Agentic Security
                    </p>
                    <span className="text-2xl font-bold text-white">
                      SafeSight
                    </span>
                  </div>
                </div>
                <p className="text-gray-400 leading-relaxed max-w-md mb-7">
                  Built for continuous intelligence loops: observe, detect,
                  decide, and alert in real time with robust AI video analysis.
                </p>
                <div className="flex flex-wrap gap-3">
                  <Link
                    href={
                      isAuthenticated
                        ? "/dashboard"
                        : "/login?redirect=/dashboard"
                    }
                    className="inline-flex items-center gap-2 px-5 py-2.5 bg-cyan-500 text-black font-semibold rounded-lg hover:bg-cyan-400 transition-colors"
                  >
                    Open Dashboard
                    <ArrowRight className="w-4 h-4" />
                  </Link>
                  {isAuthenticated ? (
                    <Link
                      href="/dashboard?tab=streams"
                      className="inline-flex items-center gap-2 px-5 py-2.5 border border-gray-700 text-gray-200 rounded-lg hover:border-cyan-500/40 hover:text-cyan-300 transition-colors"
                    >
                      Open Streams
                    </Link>
                  ) : (
                    <Link
                      href="/register"
                      className="inline-flex items-center gap-2 px-5 py-2.5 border border-gray-700 text-gray-200 rounded-lg hover:border-cyan-500/40 hover:text-cyan-300 transition-colors"
                    >
                      Create Account
                    </Link>
                  )}
                </div>
              </div>

              {/* Navigation */}
              <div className="lg:col-span-3">
                <h4 className="text-white font-semibold mb-4">Platform</h4>
                <ul className="space-y-3">
                  <li>
                    <Link
                      href={footerHref("/dashboard?tab=upload", "/upload")}
                      className="text-gray-400 hover:text-cyan-300 transition-colors"
                    >
                      Upload Analysis
                    </Link>
                  </li>
                  <li>
                    <Link
                      href={footerHref("/dashboard?tab=streams", "/streams")}
                      className="text-gray-400 hover:text-cyan-300 transition-colors"
                    >
                      Live Streams
                    </Link>
                  </li>
                  <li>
                    <Link
                      href={footerHref(
                        "/dashboard?tab=history",
                        "/predictions",
                      )}
                      className="text-gray-400 hover:text-cyan-300 transition-colors"
                    >
                      Prediction Logs
                    </Link>
                  </li>
                  <li>
                    <Link
                      href={footerHref("/dashboard?tab=alerts", "/alerts")}
                      className="text-gray-400 hover:text-cyan-300 transition-colors"
                    >
                      Alert Center
                    </Link>
                  </li>
                </ul>
              </div>

              {/* Resources */}
              <div className="lg:col-span-2">
                <h4 className="text-white font-semibold mb-4">Resources</h4>
                <ul className="space-y-3">
                  <li>
                    <Link
                      href={footerHref("/dashboard?tab=videos", "/videos")}
                      className="text-gray-400 hover:text-cyan-300 transition-colors"
                    >
                      Video Library
                    </Link>
                  </li>
                  <li>
                    <Link
                      href={footerHref("/dashboard?tab=settings", "/model")}
                      className="text-gray-400 hover:text-cyan-300 transition-colors"
                    >
                      Model Details
                    </Link>
                  </li>
                  <li>
                    <Link
                      href={footerHref("/dashboard?tab=settings", "/settings")}
                      className="text-gray-400 hover:text-cyan-300 transition-colors"
                    >
                      System Settings
                    </Link>
                  </li>
                </ul>
              </div>
            </div>

            <div className="mt-10 pt-6 border-t border-gray-800 flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
              <p className="text-gray-500 text-sm">
                2026 SafeSight. Real-time AI vigilance for modern
                infrastructure.
              </p>
              <div className="flex items-center gap-6 text-sm">
                {isAuthenticated ? (
                  <>
                    <Link
                      href="/dashboard"
                      className="text-gray-500 hover:text-cyan-300 transition-colors"
                    >
                      Dashboard
                    </Link>
                    <Link
                      href="/dashboard?tab=profile"
                      className="text-gray-500 hover:text-cyan-300 transition-colors"
                    >
                      Profile
                    </Link>
                    <Link
                      href="/dashboard?tab=settings"
                      className="text-gray-500 hover:text-cyan-300 transition-colors"
                    >
                      Settings
                    </Link>
                  </>
                ) : (
                  <>
                    <Link
                      href="/login"
                      className="text-gray-500 hover:text-cyan-300 transition-colors"
                    >
                      Sign In
                    </Link>
                    <Link
                      href="/register"
                      className="text-gray-500 hover:text-cyan-300 transition-colors"
                    >
                      Register
                    </Link>
                    <Link
                      href="/login?redirect=/dashboard"
                      className="text-gray-500 hover:text-cyan-300 transition-colors"
                    >
                      Dashboard
                    </Link>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

/* Stat Item Component */
interface StatItemProps {
  value: string;
  label: string;
  isStatus?: boolean;
  statusActive?: boolean;
}

function StatItem({ value, label, isStatus, statusActive }: StatItemProps) {
  return (
    <div className="text-center">
      <div
        className={cn(
          "text-2xl lg:text-3xl font-bold mb-1",
          isStatus
            ? statusActive
              ? "text-emerald-400"
              : "text-red-400"
            : "text-white",
        )}
      >
        {value}
      </div>
      <div className="text-xs text-gray-500 uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}

/* Dashboard Preview Component */
interface DashboardPreviewProps {
  health: HealthStatus;
  stats: PredictionStats | null;
}

function DashboardPreview({ health, stats }: DashboardPreviewProps) {
  return (
    <div className="relative">
      {/* Glow Effect */}
      <div className="absolute inset-0 bg-cyan-500/5 blur-3xl rounded-full" />

      {/* Terminal Window */}
      <div className="relative bg-gray-950 border border-gray-800 rounded-2xl overflow-hidden">
        {/* Title Bar */}
        <div className="flex items-center justify-between px-4 py-3 bg-gray-900/50 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
          </div>
          <span className="text-xs text-gray-500 font-mono">
            violence-detection.monitor
          </span>
          <div className="w-16" />
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          {/* Detection Frame */}
          <div className="relative bg-gray-900 rounded-lg aspect-video flex items-center justify-center border border-gray-800">
            <div className="absolute inset-4 border-2 border-dashed border-cyan-500/30 rounded-lg" />
            <div className="absolute top-6 left-6 px-2 py-1 bg-cyan-500/20 rounded text-xs text-cyan-400 font-mono">
              FRAME: 00:00:12
            </div>
            <div className="absolute bottom-6 right-6 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs text-emerald-400 font-mono">
                ANALYZING
              </span>
            </div>

            {/* Mock Detection Boxes */}
            <div className="absolute top-1/3 left-1/4 w-16 h-20 border-2 border-cyan-400 rounded" />
            <div className="absolute top-1/3 right-1/3 w-14 h-18 border-2 border-emerald-400 rounded" />

            <Video className="w-12 h-12 text-gray-700" />
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
              <div className="text-xs text-gray-500 mb-1">Status</div>
              <div className="flex items-center gap-2">
                <div
                  className={cn(
                    "w-2 h-2 rounded-full",
                    health.mlService ? "bg-emerald-500" : "bg-red-500",
                  )}
                />
                <span className="text-sm text-white font-mono">
                  {health.mlService ? "ACTIVE" : "OFFLINE"}
                </span>
              </div>
            </div>
            <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
              <div className="text-xs text-gray-500 mb-1">Confidence</div>
              <span className="text-sm text-cyan-400 font-mono">87.3%</span>
            </div>
            <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
              <div className="text-xs text-gray-500 mb-1">Threats</div>
              <span className="text-sm text-red-400 font-mono">
                {stats?.violent || 0}
              </span>
            </div>
          </div>

          {/* Log Lines */}
          <div className="font-mono text-xs space-y-1 text-gray-500">
            <div>
              <span className="text-cyan-400">[INFO]</span> Model loaded
              successfully
            </div>
            <div>
              <span className="text-emerald-400">[OK]</span> Database connection
              established
            </div>
            <div>
              <span className="text-yellow-400">[SCAN]</span> Monitoring active
              streams...
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* Feature Card Component */
interface FeatureCardProps {
  icon: React.ElementType;
  title: string;
  description: string;
  href: string;
}

function FeatureCard({
  icon: Icon,
  title,
  description,
  href,
}: FeatureCardProps) {
  return (
    <Link href={href}>
      <motion.div
        whileHover={{ y: -4 }}
        className="group p-6 bg-gray-950 border border-gray-800 rounded-xl hover:border-gray-700 transition-all cursor-pointer h-full"
      >
        <div className="w-12 h-12 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4 group-hover:bg-cyan-500/20 transition-colors">
          <Icon className="w-6 h-6 text-cyan-400" />
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
        <p className="text-gray-500 text-sm leading-relaxed">{description}</p>
      </motion.div>
    </Link>
  );
}

/* System Card Component */
interface SystemCardProps {
  icon: React.ElementType;
  title: string;
  status: string;
  isActive: boolean;
}

function SystemCard({ icon: Icon, title, status, isActive }: SystemCardProps) {
  return (
    <div className="flex items-center justify-between p-4 bg-black border border-gray-800 rounded-xl">
      <div className="flex items-center gap-3">
        <Icon
          className={cn(
            "w-5 h-5",
            isActive ? "text-emerald-400" : "text-gray-600",
          )}
        />
        <span className="text-white font-medium">{title}</span>
      </div>
      <div
        className={cn(
          "flex items-center gap-2 text-sm",
          isActive ? "text-emerald-400" : "text-gray-500",
        )}
      >
        <div
          className={cn(
            "w-2 h-2 rounded-full",
            isActive ? "bg-emerald-500" : "bg-gray-600",
          )}
        />
        {status}
      </div>
    </div>
  );
}

/* Quick Stat Component */
interface QuickStatProps {
  icon: React.ElementType;
  value: number | string;
  label: string;
  color: "cyan" | "red" | "green" | "purple";
}

function QuickStat({ icon: Icon, value, label, color }: QuickStatProps) {
  const colorClasses = {
    cyan: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
    red: "bg-red-500/10 text-red-400 border-red-500/20",
    green: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
  };

  return (
    <div className={cn("p-6 rounded-xl border", colorClasses[color])}>
      <div className="flex items-center gap-3 mb-3">
        <Icon className="w-5 h-5" />
        <span className="text-sm text-gray-400">{label}</span>
      </div>
      <div className="text-3xl font-bold text-white">{value}</div>
    </div>
  );
}
