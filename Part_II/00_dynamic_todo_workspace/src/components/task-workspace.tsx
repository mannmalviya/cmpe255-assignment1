"use client";

import { closestCenter, DndContext, KeyboardSensor, PointerSensor, useSensor, useSensors, type DragEndEvent } from "@dnd-kit/core";
import { arrayMove, SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { CalendarCheck, CalendarClock, CheckCircle2, ChevronDown, Circle, Command, Inbox, ListFilter, Moon, Plus, Search, Sparkles, Sun, Tags, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { TaskModal } from "./task-modal";
import { TaskRow } from "./task-row";
import type { Task, TaskDraft, TaskFilter, TaskSort } from "@/lib/types";

type Toast = { message: string; undo?: () => void } | null;
const filterDetails: Record<TaskFilter, { title: string; description: string }> = {
  all: { title: "My tasks", description: "A calm view of everything on your plate." },
  today: { title: "Today", description: "Focus on what deserves your attention now." },
  upcoming: { title: "Upcoming", description: "See what is waiting around the corner." },
  completed: { title: "Completed", description: "A record of the progress you have made." },
};

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error ?? "Something went wrong");
  return data as T;
}

export function TaskWorkspace({ initialTasks, today }: { initialTasks: Task[]; today: string }) {
  const mounted = useSyncExternalStore(() => () => {}, () => true, () => false);
  const [tasks, setTasks] = useState(initialTasks);
  const [filter, setFilter] = useState<TaskFilter>("all");
  const [sort, setSort] = useState<TaskSort>("manual");
  const [query, setQuery] = useState("");
  const [quickTitle, setQuickTitle] = useState("");
  const [editing, setEditing] = useState<Task | "new" | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [toast, setToast] = useState<Toast>(null);
  const [busy, setBusy] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const quickRef = useRef<HTMLInputElement>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }), useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }));

  const showToast = useCallback((next: NonNullable<Toast>) => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setToast(next);
    toastTimer.current = setTimeout(() => setToast(null), 5500);
  }, []);

  useEffect(() => () => { if (toastTimer.current) clearTimeout(toastTimer.current); }, []);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      const isTyping = ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
      if (event.key === "/" && !isTyping) { event.preventDefault(); searchRef.current?.focus(); }
      if (event.key.toLowerCase() === "n" && !isTyping) { event.preventDefault(); setEditing("new"); }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const counts = useMemo(() => ({
    all: tasks.filter((task) => !task.completed).length,
    today: tasks.filter((task) => !task.completed && task.dueDate === today).length,
    upcoming: tasks.filter((task) => !task.completed && Boolean(task.dueDate && task.dueDate > today)).length,
    completed: tasks.filter((task) => task.completed).length,
  }), [tasks, today]);
  const tags = useMemo(() => [...new Set(tasks.flatMap((task) => task.tags))].sort().slice(0, 6), [tasks]);
  const visibleTasks = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const matches = tasks.filter((task) => {
      if (filter === "all" && task.completed) return false;
      if (filter === "today" && (task.completed || task.dueDate !== today)) return false;
      if (filter === "upcoming" && (task.completed || !task.dueDate || task.dueDate <= today)) return false;
      if (filter === "completed" && !task.completed) return false;
      return !needle || [task.title, task.notes, ...task.tags].some((value) => value.toLowerCase().includes(needle));
    });
    return [...matches].sort((a, b) => {
      if (sort === "dueDate") return (a.dueDate ?? "9999").localeCompare(b.dueDate ?? "9999");
      if (sort === "priority") return ({ high: 0, medium: 1, low: 2 }[a.priority] - { high: 0, medium: 1, low: 2 }[b.priority]);
      if (sort === "createdAt") return b.createdAt.localeCompare(a.createdAt);
      return a.position - b.position;
    });
  }, [filter, query, sort, tasks, today]);
  const progress = tasks.length ? Math.round((counts.completed / tasks.length) * 100) : 0;
  const draggingEnabled = filter === "all" && sort === "manual" && !query;

  async function create(draft: TaskDraft) {
    const { task } = await requestJson<{ task: Task }>("/api/tasks", { method: "POST", body: JSON.stringify(draft) });
    setTasks((current) => [...current, task]); setEditing(null); showToast({ message: "Task created" });
  }
  async function save(draft: TaskDraft) {
    if (editing === "new" || !editing) return create(draft);
    const { task } = await requestJson<{ task: Task }>(`/api/tasks/${editing.id}`, { method: "PATCH", body: JSON.stringify(draft) });
    setTasks((current) => current.map((item) => item.id === task.id ? task : item)); setEditing(null); showToast({ message: "Changes saved" });
  }
  async function quickAdd(event: React.FormEvent) {
    event.preventDefault(); const title = quickTitle.trim(); if (!title || busy) return; setBusy(true);
    try { await create({ title, notes: "", priority: "medium", dueDate: filter === "today" ? today : null, tags: [] }); setQuickTitle(""); quickRef.current?.focus(); }
    catch (error) { showToast({ message: error instanceof Error ? error.message : "Unable to create task" }); }
    finally { setBusy(false); }
  }
  async function toggleTask(task: Task) {
    const completed = !task.completed;
    setTasks((current) => current.map((item) => item.id === task.id ? { ...item, completed } : item));
    try {
      const result = await requestJson<{ task: Task }>(`/api/tasks/${task.id}`, { method: "PATCH", body: JSON.stringify({ completed }) });
      setTasks((current) => current.map((item) => item.id === task.id ? result.task : item));
      showToast({ message: completed ? "Nice work — task completed" : "Task moved back to active" });
    } catch (error) { setTasks((current) => current.map((item) => item.id === task.id ? task : item)); showToast({ message: error instanceof Error ? error.message : "Unable to update task" }); }
  }
  async function removeTask(task: Task) {
    setTasks((current) => current.filter((item) => item.id !== task.id));
    setSelected((current) => { const next = new Set(current); next.delete(task.id); return next; });
    try {
      await requestJson(`/api/tasks/${task.id}`, { method: "DELETE" });
      showToast({ message: "Task deleted", undo: async () => {
        const { task: restored } = await requestJson<{ task: Task }>("/api/tasks", { method: "POST", body: JSON.stringify({ title: task.title, notes: task.notes, priority: task.priority, dueDate: task.dueDate, tags: task.tags }) });
        if (task.completed) { const result = await requestJson<{ task: Task }>(`/api/tasks/${restored.id}`, { method: "PATCH", body: JSON.stringify({ completed: true }) }); setTasks((current) => [...current, result.task]); }
        else setTasks((current) => [...current, restored]);
        setToast(null);
      } });
    } catch (error) { setTasks((current) => [...current, task]); showToast({ message: error instanceof Error ? error.message : "Unable to delete task" }); }
  }
  function toggleSelected(id: number) { setSelected((current) => { const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next; }); }
  async function runBulk(action: "complete" | "reopen" | "delete") {
    if (!selected.size) return;
    try { const result = await requestJson<{ tasks: Task[] }>("/api/tasks/bulk", { method: "POST", body: JSON.stringify({ ids: [...selected], action }) }); setTasks(result.tasks); showToast({ message: `${selected.size} ${selected.size === 1 ? "task" : "tasks"} updated` }); setSelected(new Set()); }
    catch (error) { showToast({ message: error instanceof Error ? error.message : "Unable to update tasks" }); }
  }
  async function onDragEnd(event: DragEndEvent) {
    if (!event.over || event.active.id === event.over.id) return;
    const oldIndex = tasks.findIndex((task) => task.id === event.active.id); const newIndex = tasks.findIndex((task) => task.id === event.over?.id); if (oldIndex < 0 || newIndex < 0) return;
    const reordered = arrayMove(tasks, oldIndex, newIndex); setTasks(reordered);
    try { const result = await requestJson<{ tasks: Task[] }>("/api/tasks/reorder", { method: "POST", body: JSON.stringify({ ids: reordered.map((task) => task.id) }) }); setTasks(result.tasks); }
    catch { setTasks(tasks); showToast({ message: "Could not save the new order" }); }
  }
  function changeFilter(next: TaskFilter) { setFilter(next); setSelected(new Set()); setMobileNavOpen(false); }
  function toggleTheme() { const dark = document.documentElement.dataset.theme === "dark"; document.documentElement.dataset.theme = dark ? "light" : "dark"; localStorage.setItem("tempo-theme", dark ? "light" : "dark"); }
  const navItems: { id: TaskFilter; label: string; icon: React.ReactNode }[] = [
    { id: "all", label: "My tasks", icon: <Inbox size={18} aria-hidden="true" /> }, { id: "today", label: "Today", icon: <CalendarCheck size={18} aria-hidden="true" /> },
    { id: "upcoming", label: "Upcoming", icon: <CalendarClock size={18} aria-hidden="true" /> }, { id: "completed", label: "Completed", icon: <CheckCircle2 size={18} aria-hidden="true" /> },
  ];

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNavOpen ? "mobile-open" : ""}`} aria-label="Task views">
        <div className="brand"><span className="brand-mark"><Sparkles size={18} aria-hidden="true" /></span><span>tempo</span></div>
        <button className="button primary sidebar-new" type="button" onClick={() => setEditing("new")}><Plus size={18} /> New task <kbd>N</kbd></button>
        <nav className="sidebar-nav"><p className="nav-label">Workspace</p>{navItems.map((item) => <button className={filter === item.id ? "active" : ""} key={item.id} type="button" onClick={() => changeFilter(item.id)}><span>{item.icon}{item.label}</span><strong>{counts[item.id]}</strong></button>)}</nav>
        {tags.length > 0 && <div className="sidebar-tags"><p className="nav-label"><Tags size={13} aria-hidden="true" /> Tags</p>{tags.map((tag) => <button key={tag} type="button" onClick={() => { setQuery(tag); setMobileNavOpen(false); }}>#{tag}</button>)}</div>}
        <div className="progress-card"><div><span>Overall progress</span><strong>{progress}%</strong></div><div className="progress-track" role="progressbar" aria-label="Overall task progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><span style={{ width: `${progress}%` }} /></div><p>{counts.completed} of {tasks.length} tasks complete</p></div>
      </aside>
      <main className="workspace">
        <header className="topbar"><button className="mobile-view-button" type="button" onClick={() => setMobileNavOpen((open) => !open)} aria-expanded={mobileNavOpen}><ListFilter size={17} /> Views <ChevronDown size={15} /></button><p className="today-label">{new Intl.DateTimeFormat("en-US", { weekday: "long", month: "long", day: "numeric" }).format(new Date(`${today}T12:00:00`))}</p><div className="top-actions"><span className="saved-status"><span /> Saved locally</span><button className="icon-button" type="button" onClick={toggleTheme} aria-label="Toggle color theme"><Sun className="sun-icon" size={18} /><Moon className="moon-icon" size={18} /></button><span className="avatar" aria-label="Personal workspace">ME</span></div></header>
        <div className="content">
          <section className="page-heading"><div><p className="eyebrow">Personal workspace</p><h1>{filterDetails[filter].title}</h1><p>{filterDetails[filter].description}</p></div><div className="heading-stat"><strong>{counts.all}</strong><span>open {counts.all === 1 ? "task" : "tasks"}</span></div></section>
          <form className="quick-add" onSubmit={quickAdd}><span className="quick-plus"><Plus size={20} aria-hidden="true" /></span><label className="sr-only" htmlFor="quick-task">Quick-add a task</label><input id="quick-task" ref={quickRef} value={quickTitle} onChange={(event) => setQuickTitle(event.target.value)} placeholder="Add a task, then press Enter…" maxLength={140} /><button type="button" className="expand-add" onClick={() => setEditing("new")}>Add details</button><button type="submit" className="button primary compact" disabled={!quickTitle.trim() || busy}>{busy ? "Adding…" : "Add task"}</button></form>
          <section className="task-panel" aria-labelledby="task-list-heading">
            <div className="task-toolbar"><div><h2 id="task-list-heading">{query ? "Search results" : filterDetails[filter].title}</h2><span>{visibleTasks.length} {visibleTasks.length === 1 ? "task" : "tasks"}</span></div><div className="toolbar-controls"><label className="search-box"><Search size={16} aria-hidden="true" /><span className="sr-only">Search tasks</span><input ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search tasks" />{query ? <button type="button" onClick={() => setQuery("")} aria-label="Clear search"><X size={14} /></button> : <kbd>/</kbd>}</label><label className="sort-select"><span className="sr-only">Sort tasks</span><select value={sort} onChange={(event) => setSort(event.target.value as TaskSort)}><option value="manual">Manual order</option><option value="dueDate">Due date</option><option value="priority">Priority</option><option value="createdAt">Newest</option></select><ChevronDown size={14} aria-hidden="true" /></label></div></div>
            {selected.size > 0 && <div className="bulk-bar" role="region" aria-label="Bulk task actions"><span><strong>{selected.size}</strong> selected</span><div><button type="button" onClick={() => runBulk(filter === "completed" ? "reopen" : "complete")}><CheckCircle2 size={15} />{filter === "completed" ? "Reopen" : "Complete"}</button><button type="button" className="danger" onClick={() => runBulk("delete")}><Trash2 size={15} />Delete</button><button className="icon-button" type="button" onClick={() => setSelected(new Set())} aria-label="Clear selection"><X size={15} /></button></div></div>}
            {visibleTasks.length ? (mounted ? <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}><SortableContext items={visibleTasks.map((task) => task.id)} strategy={verticalListSortingStrategy}><div className="task-list">{visibleTasks.map((task) => <TaskRow key={task.id} task={task} today={today} selected={selected.has(task.id)} draggingEnabled={draggingEnabled} onSelect={toggleSelected} onToggle={toggleTask} onEdit={setEditing} onDelete={removeTask} />)}</div></SortableContext></DndContext> : <div className="task-list">{visibleTasks.map((task) => <TaskRow key={task.id} task={task} today={today} selected={selected.has(task.id)} draggingEnabled={false} onSelect={toggleSelected} onToggle={toggleTask} onEdit={setEditing} onDelete={removeTask} />)}</div>) : <div className="empty-state"><span><Circle size={28} /></span><h3>{query ? "No matching tasks" : filter === "completed" ? "Nothing completed yet" : "A clear horizon"}</h3><p>{query ? "Try a different word or clear your search." : "Add a task when something deserves your attention."}</p>{query ? <button className="button secondary" type="button" onClick={() => setQuery("")}>Clear search</button> : <button className="button primary" type="button" onClick={() => setEditing("new")}><Plus size={16} /> Add a task</button>}</div>}
            {draggingEnabled && visibleTasks.length > 1 && <p className="reorder-hint"><Command size={13} /> Drag the handle or use Space + arrow keys to reorder</p>}
          </section>
        </div>
      </main>
      {editing && <TaskModal key={editing === "new" ? "new" : editing.id} task={editing === "new" ? null : editing} onClose={() => setEditing(null)} onSave={save} />}
      <div className={`toast ${toast ? "visible" : ""}`} role="status" aria-live="polite"><span>{toast?.message}</span>{toast?.undo && <button type="button" onClick={toast.undo}>Undo</button>}</div>
    </div>
  );
}
