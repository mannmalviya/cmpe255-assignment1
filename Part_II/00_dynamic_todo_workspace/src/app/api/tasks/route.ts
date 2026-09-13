import { createTask, getTasks } from "@/lib/db";
import { taskCreateSchema, validationMessage } from "@/lib/validation";
import { ZodError } from "zod";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return Response.json({ tasks: getTasks() });
}

export async function POST(request: Request) {
  try {
    const input = taskCreateSchema.parse(await request.json());
    return Response.json({ task: createTask(input) }, { status: 201 });
  } catch (error) {
    if (error instanceof ZodError) {
      return Response.json({ error: validationMessage(error) }, { status: 400 });
    }
    return Response.json({ error: "Unable to create the task" }, { status: 500 });
  }
}

