import { deleteTask, updateTask } from "@/lib/db";
import { taskUpdateSchema, validationMessage } from "@/lib/validation";
import { ZodError } from "zod";

export const runtime = "nodejs";

function parseId(value: string) {
  const id = Number(value);
  return Number.isInteger(id) && id > 0 ? id : null;
}

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const id = parseId((await params).id);
  if (!id) return Response.json({ error: "Invalid task id" }, { status: 400 });
  try {
    const patch = taskUpdateSchema.parse(await request.json());
    const task = updateTask(id, patch);
    return task
      ? Response.json({ task })
      : Response.json({ error: "Task not found" }, { status: 404 });
  } catch (error) {
    if (error instanceof ZodError) {
      return Response.json({ error: validationMessage(error) }, { status: 400 });
    }
    return Response.json({ error: "Unable to update the task" }, { status: 500 });
  }
}

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const id = parseId((await params).id);
  if (!id) return Response.json({ error: "Invalid task id" }, { status: 400 });
  const task = deleteTask(id);
  return task
    ? Response.json({ task })
    : Response.json({ error: "Task not found" }, { status: 404 });
}

