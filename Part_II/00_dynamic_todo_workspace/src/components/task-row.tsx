"use client";

import { CSS } from "@dnd-kit/utilities";
import { useSortable } from "@dnd-kit/sortable";
import { CalendarDays, Check, GripVertical, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { useState } from "react";
import { friendlyDate } from "@/lib/date";
import type { Task } from "@/lib/types";

type Props = { task: Task; today: string; selected: boolean; draggingEnabled: boolean; onSelect: (id: number) => void; onToggle: (task: Task) => void; onEdit: (task: Task) => void; onDelete: (task: Task) => void };

export function TaskRow({ task, today, selected, draggingEnabled, onSelect, onToggle, onEdit, onDelete }: Props) {
  const [menuOpen, setMenuOpen] = useState(false);
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: task.id, disabled: !draggingEnabled });
  const dueLabel = friendlyDate(task.dueDate);
  const overdue = Boolean(task.dueDate && task.dueDate < today && !task.completed);
  return (
    <article ref={setNodeRef} className={`task-row ${task.completed ? "is-complete" : ""} ${isDragging ? "is-dragging" : ""}`} style={{ transform: CSS.Transform.toString(transform), transition }} data-testid="task-row">
      <label className="selection-control" title="Select task"><input type="checkbox" checked={selected} onChange={() => onSelect(task.id)} aria-label={`Select ${task.title}`} /></label>
      <button className={`complete-button priority-${task.priority}`} type="button" onClick={() => onToggle(task)} aria-label={task.completed ? `Mark ${task.title} active` : `Complete ${task.title}`} aria-pressed={task.completed}>{task.completed && <Check size={15} strokeWidth={3} aria-hidden="true" />}</button>
      <div className="task-copy">
        <button className="task-title-button" type="button" onClick={() => onEdit(task)}><span className="task-title">{task.title}</span>{task.notes && <span className="task-notes">{task.notes}</span>}</button>
        <div className="task-meta">{dueLabel && <span className={overdue ? "date-chip overdue" : "date-chip"}><CalendarDays size={13} aria-hidden="true" /> {overdue ? `Overdue · ${dueLabel}` : dueLabel}</span>}<span className={`priority-label ${task.priority}`}>{task.priority}</span>{task.tags.map((tag) => <span className="tag-chip" key={tag}>#{tag}</span>)}</div>
      </div>
      <div className="task-actions">
        <button className="drag-handle" type="button" disabled={!draggingEnabled} aria-label={draggingEnabled ? `Reorder ${task.title}` : "Choose manual sort to reorder"} {...attributes} {...listeners}><GripVertical size={18} aria-hidden="true" /></button>
        <div className="menu-wrap"><button className="icon-button subtle" type="button" onClick={() => setMenuOpen((open) => !open)} aria-label={`Actions for ${task.title}`} aria-expanded={menuOpen}><MoreHorizontal size={19} aria-hidden="true" /></button>{menuOpen && <div className="task-menu"><button type="button" onClick={() => { setMenuOpen(false); onEdit(task); }}><Pencil size={15} aria-hidden="true" /> Edit</button><button type="button" className="danger" onClick={() => { setMenuOpen(false); onDelete(task); }}><Trash2 size={15} aria-hidden="true" /> Delete</button></div>}</div>
      </div>
    </article>
  );
}
