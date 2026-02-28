import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "SafeSight - AI Video Violence Detection",
  description: "AI-powered video violence detection using deep learning",
  keywords: [
    "violence detection",
    "AI",
    "video analysis",
    "deep learning",
    "machine learning",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} antialiased`}>
        <Providers>
          <div className="min-h-screen bg-grid">{children}</div>
        </Providers>
      </body>
    </html>
  );
}
