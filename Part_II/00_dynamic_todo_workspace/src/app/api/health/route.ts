import { databaseHealth } from "@/lib/db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const health = databaseHealth();
  return Response.json(health, { status: health.status === "ok" ? 200 : 503 });
}

