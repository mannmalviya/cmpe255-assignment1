"use client";

import { useEffect, useRef, useState } from "react";
import { CalendarDays, Flag, Tag, X } from "lucide-react";
import type { Task, TaskDraft } from "@/lib/types";

type Props = { task: Task | null; onClose: () => void; onSave: (draft: TaskDraft) => Promise<void> };

export function TaskModal({ task, onClose, onSave }: Props) {
  const [title, setTitle] = useState(task?.title ?? "");
  const [notes, setNotes] = useState(task?.notes ?? "");
  const [priority, setPriority] = useState<TaskDraft["priority"]>(task?.priority ?? "medium");
  const [dueDate, setDueDate] = useState(task?.dueDate ?? "");
  const [tags, setTags] = useState(task?.tags.join(", ") ?? "");
  const [saving, setSaving] = useState(false);
  const titleRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    titleRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!title.trim() || saving) return;
    setSaving(true);
    try {
      await onSave({
        title: title.trim(), notes: notes.trim(), priority, dueDate: dueDate || null,
        tags: tags.split(",").map((tag) => tag.trim()).filter(Boolean),
      });
    } finally { setSaving(false); }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="task-modal" role="dialog" aria-modal="true" aria-labelledby="task-modal-title">
        <div className="modal-heading">
          <div><p className="eyebrow">{task ? "Update details" : "Capture what matters"}</p><h2 id="task-modal-title">{task ? "Edit task" : "Create a task"}</h2></div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close task editor"><X size={19} aria-hidden="true" /></button>
        </div>
        <form onSubmit={submit} className="task-form">
          <label className="field"><span>Task name</span><input ref={titleRef} value={title} onChange={(event) => setTitle(event.target.value)} maxLength={140} required placeholder="What needs your attention?" /><small>{title.length}/140</small></label>
          <label className="field"><span>Notes <em>Optional</em></span><textarea value={notes} onChange={(event) => setNotes(event.target.value)} maxLength={1000} rows={4} placeholder="Add context, a definition of done, or a helpful link…" /></label>
          <div className="form-grid">
            <label className="field"><span><Flag size={15} aria-hidden="true" /> Priority</span><select value={priority} onChange={(event) => setPriority(event.target.value as TaskDraft["priority"])}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label>
            <label className="field"><span><CalendarDays size={15} aria-hidden="true" /> Due date</span><input type="date" value={dueDate} onChange={(event) => setDueDate(event.target.value)} /></label>
          </div>
          <label className="field"><span><Tag size={15} aria-hidden="true" /> Tags <em>Comma separated</em></span><input value={tags} onChange={(event) => setTags(event.target.value)} placeholder="work, personal" /></label>
          <div className="modal-actions"><button type="button" className="button secondary" onClick={onClose}>Cancel</button><button type="submit" className="button primary" disabled={!title.trim() || saving}>{saving ? "Saving…" : task ? "Save changes" : "Create task"}</button></div>
        </form>
      </section>
    </div>
  );
}

