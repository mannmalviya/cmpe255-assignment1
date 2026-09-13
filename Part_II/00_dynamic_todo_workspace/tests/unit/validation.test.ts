import { describe, expect, it } from "vitest";
import { taskCreateSchema, taskUpdateSchema } from "@/lib/validation";

describe("task validation", () => {
  it("normalizes whitespace and deduplicates lowercase tags", () => {
    const task = taskCreateSchema.parse({
      title: "  Ship the design  ",
      tags: ["Work", "work", " Review "],
    });
    expect(task.title).toBe("Ship the design");
    expect(task.tags).toEqual(["work", "review"]);
    expect(task.priority).toBe("medium");
    expect(task.dueDate).toBeNull();
  });

  it("rejects blank and oversized task titles", () => {
    expect(taskCreateSchema.safeParse({ title: "   " }).success).toBe(false);
    expect(taskCreateSchema.safeParse({ title: "x".repeat(141) }).success).toBe(false);
  });

  it("accepts a focused completion-only update", () => {
    expect(taskUpdateSchema.parse({ completed: true })).toEqual({ completed: true });
  });

  it("rejects ambiguous date formats", () => {
    expect(taskCreateSchema.safeParse({ title: "Task", dueDate: "09/15/2026" }).success).toBe(false);
  });
});

