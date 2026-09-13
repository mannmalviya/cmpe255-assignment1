export const PRIORITIES = ["low", "medium", "high"] as const;

export type Priority = (typeof PRIORITIES)[number];

export type Task = {
  id: number;
  title: string;
  notes: string;
  priority: Priority;
  dueDate: string | null;
  tags: string[];
  completed: boolean;
  position: number;
  createdAt: string;
  updatedAt: string;
  completedAt: string | null;
};

export type TaskDraft = Pick<
  Task,
  "title" | "notes" | "priority" | "dueDate" | "tags"
>;

export type TaskFilter = "all" | "today" | "upcoming" | "completed";
export type TaskSort = "manual" | "dueDate" | "priority" | "createdAt";

