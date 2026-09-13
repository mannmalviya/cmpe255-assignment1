import "server-only";

import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import type { Priority, Task, TaskDraft } from "./types";
import { addDaysKey, localDateKey } from "./date";

type TaskRow = {
  id: number;
  title: string;
  notes: string;
  priority: Priority;
  due_date: string | null;
  tags: string;
  completed: number;
  position: number;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
};

const databasePath =
  process.env.TODO_DB_PATH ?? path.join(process.cwd(), "data", "tempo.sqlite");

fs.mkdirSync(path.dirname(databasePath), { recursive: true });

const db = new Database(databasePath);
db.pragma("journal_mode = WAL");
db.pragma("foreign_keys = ON");
db.pragma("busy_timeout = 5000");

db.exec(`
  CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL CHECK(length(title) BETWEEN 1 AND 140),
    notes TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'medium' CHECK(priority IN ('low', 'medium', 'high')),
    due_date TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    completed INTEGER NOT NULL DEFAULT 0 CHECK(completed IN (0, 1)),
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
  );
  CREATE INDEX IF NOT EXISTS idx_tasks_position ON tasks(position);
  CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
  CREATE INDEX IF NOT EXISTS idx_tasks_completed ON tasks(completed);
  CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
  );
`);

function mapTask(row: TaskRow): Task {
  let tags: string[] = [];
  try {
    tags = JSON.parse(row.tags) as string[];
  } catch {
    tags = [];
  }
  return {
    id: row.id,
    title: row.title,
    notes: row.notes,
    priority: row.priority,
    dueDate: row.due_date,
    tags,
    completed: Boolean(row.completed),
    position: row.position,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    completedAt: row.completed_at,
  };
}

const selectById = db.prepare("SELECT * FROM tasks WHERE id = ?");

export function getTasks() {
  return (db
    .prepare("SELECT * FROM tasks ORDER BY completed ASC, position ASC, created_at DESC")
    .all() as TaskRow[]).map(mapTask);
}

export function getTask(id: number) {
  const row = selectById.get(id) as TaskRow | undefined;
  return row ? mapTask(row) : null;
}

export function createTask(input: TaskDraft) {
  const now = new Date().toISOString();
  const nextPosition = (
    db.prepare("SELECT COALESCE(MAX(position), -1) + 1 AS value FROM tasks").get() as {
      value: number;
    }
  ).value;
  const result = db
    .prepare(
      `INSERT INTO tasks (title, notes, priority, due_date, tags, position, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.title,
      input.notes,
      input.priority,
      input.dueDate,
      JSON.stringify(input.tags),
      nextPosition,
      now,
      now,
    );
  return getTask(Number(result.lastInsertRowid))!;
}

export function updateTask(id: number, patch: Partial<TaskDraft> & { completed?: boolean }) {
  const current = getTask(id);
  if (!current) return null;
  const completed = patch.completed ?? current.completed;
  const now = new Date().toISOString();
  const completedAt = completed
    ? current.completedAt ?? now
    : null;
  db.prepare(
    `UPDATE tasks SET title = ?, notes = ?, priority = ?, due_date = ?, tags = ?,
      completed = ?, completed_at = ?, updated_at = ? WHERE id = ?`,
  ).run(
    patch.title ?? current.title,
    patch.notes ?? current.notes,
    patch.priority ?? current.priority,
    patch.dueDate === undefined ? current.dueDate : patch.dueDate,
    JSON.stringify(patch.tags ?? current.tags),
    completed ? 1 : 0,
    completedAt,
    now,
    id,
  );
  return getTask(id);
}

export function deleteTask(id: number) {
  const task = getTask(id);
  if (!task) return null;
  db.prepare("DELETE FROM tasks WHERE id = ?").run(id);
  return task;
}

export const reorderTasks = db.transaction((ids: number[]) => {
  const update = db.prepare("UPDATE tasks SET position = ?, updated_at = ? WHERE id = ?");
  const now = new Date().toISOString();
  ids.forEach((id, index) => update.run(index, now, id));
  return getTasks();
});

export const bulkTasks = db.transaction(
  (ids: number[], action: "complete" | "reopen" | "delete") => {
    const placeholders = ids.map(() => "?").join(",");
    if (action === "delete") {
      db.prepare(`DELETE FROM tasks WHERE id IN (${placeholders})`).run(...ids);
    } else {
      const completed = action === "complete" ? 1 : 0;
      const now = new Date().toISOString();
      db.prepare(
        `UPDATE tasks SET completed = ?, completed_at = ?, updated_at = ?
         WHERE id IN (${placeholders})`,
      ).run(completed, completed ? now : null, now, ...ids);
    }
    return getTasks();
  },
);

function seedIfEmpty() {
  if (process.env.TODO_SKIP_SEED === "1") return;
  db.exec("BEGIN IMMEDIATE");
  try {
    const seeded = db.prepare("SELECT value FROM app_meta WHERE key = 'seed_version'").get();
    if (seeded) { db.exec("COMMIT"); return; }
    const count = (db.prepare("SELECT COUNT(*) AS value FROM tasks").get() as { value: number }).value;
  const seeds: TaskDraft[] = [
    {
      title: "Shape the week around three meaningful outcomes",
      notes: "Keep the list realistic and leave room for the unexpected.",
      priority: "high",
      dueDate: localDateKey(),
      tags: ["planning"],
    },
    {
      title: "Review the project brief",
      notes: "Capture open questions before starting implementation.",
      priority: "medium",
      dueDate: localDateKey(),
      tags: ["work"],
    },
    {
      title: "Take a proper screen break",
      notes: "A short walk counts.",
      priority: "low",
      dueDate: addDaysKey(1),
      tags: ["wellbeing"],
    },
    {
      title: "Prepare the Friday progress note",
      notes: "Wins, decisions, risks, and next steps.",
      priority: "medium",
      dueDate: addDaysKey(3),
      tags: ["work", "writing"],
    },
  ];
    if (count === 0) seeds.forEach(createTask);
    db.prepare("INSERT INTO app_meta (key, value) VALUES ('seed_version', '1')").run();
    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  }
}

seedIfEmpty();

export function databaseHealth() {
  const result = db.pragma("integrity_check", { simple: true }) as string;
  return { status: result === "ok" ? "ok" : "degraded", database: "sqlite", integrity: result };
}
