import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["better-sqlite3"],
  experimental: {
    useTypeScriptCli: false,
  },
};

export default nextConfig;
