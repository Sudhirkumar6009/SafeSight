"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Shield,
  Upload,
  BarChart3,
  Video,
  Activity,
  Camera,
  Bell,
  AlertTriangle,
  User,
  LogOut,
  ChevronDown,
  Home,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAlerts } from "@/hooks/useWebSocket";
import { useAuth } from "@/hooks/useAuth";

const navigation = [
  { name: "Home", href: "/", icon: Home },
  { name: "Dashboard", href: "/dashboard", icon: BarChart3 },
  { name: "Streams", href: "/streams", icon: Camera },
  { name: "Alerts", href: "/alerts", icon: Bell, badge: true },
  { name: "Upload", href: "/upload", icon: Upload },
  { name: "Videos", href: "/videos", icon: Video },
];

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { pendingCount, isConnected, clearPending } = useAlerts();
  const { user, isAuthenticated, logout, isLoading } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);

  // Clear pending count when navigating to alerts page
  useEffect(() => {
    if (pathname === "/alerts") {
      clearPending();
    }
  }, [pathname, clearPending]);

  const handleLogout = async () => {
    await logout();
    setShowUserMenu(false);
    router.push("/");
  };

  return (
    <nav className="bg-black/90 backdrop-blur-md border-b border-gray-900 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-3">
            <motion.div
              className="relative"
            >
              <div className="w-10 h-10 rounded-xl bg-cyan-500/20 flex items-center justify-center">
                <Shield className="w-6 h-6 text-cyan-400" />
              </div>
            </motion.div>
            <div>
              <span className="text-xl font-bold text-white">
                Safe<span className="text-cyan-400">Sight</span>
              </span>
            </div>
          </Link>

          {/* Navigation Links moved into profile dropdown */}

          {/* Status Indicator and Auth */}
          <div className="flex items-center space-x-4">
            <div
              className={cn(
                "hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-full border",
                isConnected
                  ? "bg-emerald-500/10 border-emerald-500/30"
                  : "bg-yellow-500/10 border-yellow-500/30",
              )}
            >
              <div
                className={cn(
                  "w-2 h-2 rounded-full animate-pulse",
                  isConnected ? "bg-emerald-500" : "bg-yellow-500",
                )}
              />
              <span
                className={cn(
                  "text-xs font-medium",
                  isConnected ? "text-emerald-400" : "text-yellow-400",
                )}
              >
                {isConnected ? "Live" : "Connecting..."}
              </span>
            </div>

            {/* Auth Section */}
            {!isLoading && (
              <>
                {isAuthenticated && user ? (
                  <div
                    className="relative"
                    onMouseEnter={() => setShowUserMenu(true)}
                    onMouseLeave={() => setShowUserMenu(false)}
                  >
                    <button
                      onClick={() => setShowUserMenu(!showUserMenu)}
                      className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-800 hover:border-gray-700 transition-colors"
                    >
                      {user.avatar ? (
                        <img
                          src={user.avatar}
                          alt={user.username}
                          className="w-6 h-6 rounded-full"
                        />
                      ) : (
                        <div className="w-6 h-6 rounded-full bg-cyan-500/20 flex items-center justify-center">
                          <User className="w-4 h-4 text-cyan-400" />
                        </div>
                      )}
                      <span className="text-sm text-gray-300 hidden sm:block">
                        {user.username}
                      </span>
                      <ChevronDown className="w-4 h-4 text-gray-500" />
                    </button>

                    {/* User Dropdown Menu */}
                    {showUserMenu && (
                      <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="absolute right-0 mt-2 w-64 py-2 bg-gray-900 border border-gray-800 rounded-xl shadow-xl z-50"
                      >
                        <div className="py-2 border-b border-gray-800">
                          {navigation.map((item) => {
                            const isActive = pathname === item.href;
                            const Icon = item.icon;
                            const showBadge = item.badge && pendingCount > 0;

                            return (
                              <Link
                                key={item.name}
                                href={item.href}
                                className={cn(
                                  "flex items-center gap-3 px-4 py-2 text-sm transition-colors",
                                  isActive
                                    ? "text-cyan-400 bg-cyan-500/10"
                                    : "text-gray-400 hover:text-white hover:bg-gray-800",
                                )}
                              >
                                <div className="relative">
                                  <Icon className="w-4 h-4" />
                                  {showBadge && (
                                    <span className="absolute -top-1.5 -right-1.5 flex h-4 w-4 items-center justify-center">
                                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                                      <span className="relative inline-flex h-3 w-3 rounded-full bg-red-500 text-[8px] text-white font-bold items-center justify-center">
                                        {pendingCount > 9 ? "9+" : pendingCount}
                                      </span>
                                    </span>
                                  )}
                                </div>
                                <span className="text-sm font-medium">
                                  {item.name}
                                </span>
                              </Link>
                            );
                          })}
                        </div>

                        <div className="px-4 py-2 border-b border-gray-800">
                          <p className="text-sm font-medium text-white">
                            {user.username}
                          </p>
                          <p className="text-xs text-gray-500">{user.email}</p>
                        </div>
                        <button
                          onClick={handleLogout}
                          className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
                        >
                          <LogOut className="w-4 h-4" />
                          Sign out
                        </button>
                      </motion.div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <Link href="/login">
                      <button className="px-4 py-1.5 text-sm text-gray-300 hover:text-white transition-colors">
                        Sign in
                      </button>
                    </Link>
                    <Link href="/register">
                      <button className="px-4 py-1.5 text-sm bg-cyan-500 text-black font-medium rounded-lg hover:bg-cyan-400 transition-colors">
                        Sign up
                      </button>
                    </Link>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
