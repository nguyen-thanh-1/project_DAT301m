import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    proxyTimeout: 600000, // 10 minutes (600,000 ms) for slow AI inference
  },
  allowedDevOrigins: [
    'anagrammatically-nonderogative-bibi.ngrok-free.dev',
    '*.ngrok-free.dev',
    '*.ngrok-free.app',
    '*.ngrok.io',
    '*.ngrok.app',
  ],
  async rewrites() {
    const backendUrl = process.env.BACKEND_INTERNAL_URL || "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: "/static/anchors/:path*",
        destination: `${backendUrl}/static/anchors/:path*`,
      },
    ];
  },
};

export default nextConfig;
