import { bulkTasks } from "@/lib/db";
import { bulkSchema, validationMessage } from "@/lib/validation";
import { ZodError } from "zod";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const { ids, action } = bulkSchema.parse(await request.json());
    return Response.json({ tasks: bulkTasks(ids, action) });
  } catch (error) {
    if (error instanceof ZodError) {
      return Response.json({ error: validationMessage(error) }, { status: 400 });
    }
    return Response.json({ error: "Unable to update tasks" }, { status: 500 });
  }
}

