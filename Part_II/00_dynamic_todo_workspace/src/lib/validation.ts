import { z } from "zod";

const dateString = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/, "Use a valid date")
  .nullable();

const taskFields = z.object({
  title: z.string().trim().min(1, "Give your task a name").max(140),
  notes: z.string().trim().max(1000),
  priority: z.enum(["low", "medium", "high"]),
  dueDate: dateString,
  tags: z
    .array(z.string().trim().min(1).max(24))
    .max(8)
    .transform((tags) => [...new Set(tags.map((tag) => tag.toLowerCase()))]),
});

export const taskCreateSchema = taskFields.extend({
  notes: taskFields.shape.notes.default(""),
  priority: taskFields.shape.priority.default("medium"),
  dueDate: taskFields.shape.dueDate.default(null),
  tags: taskFields.shape.tags.default([]),
});

export const taskUpdateSchema = taskFields.partial().extend({
  completed: z.boolean().optional(),
});

export const reorderSchema = z.object({
  ids: z.array(z.number().int().positive()).min(1),
});

export const bulkSchema = z.object({
  ids: z.array(z.number().int().positive()).min(1),
  action: z.enum(["complete", "reopen", "delete"]),
});

export function validationMessage(error: z.ZodError) {
  return error.issues[0]?.message ?? "Please check the information and try again";
}
