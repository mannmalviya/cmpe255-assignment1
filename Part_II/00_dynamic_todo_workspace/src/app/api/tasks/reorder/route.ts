import { reorderTasks } from "@/lib/db";
import { reorderSchema, validationMessage } from "@/lib/validation";
import { ZodError } from "zod";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const { ids } = reorderSchema.parse(await request.json());
    return Response.json({ tasks: reorderTasks(ids) });
  } catch (error) {
    if (error instanceof ZodError) {
      return Response.json({ error: validationMessage(error) }, { status: 400 });
    }
    return Response.json({ error: "Unable to reorder tasks" }, { status: 500 });
  }
}

