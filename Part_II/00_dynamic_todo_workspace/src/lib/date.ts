export function localDateKey(date = new Date()) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function addDaysKey(days: number, from = new Date()) {
  const value = new Date(from);
  value.setDate(value.getDate() + days);
  return localDateKey(value);
}

export function friendlyDate(value: string | null, today = new Date()) {
  if (!value) return null;
  const todayKey = localDateKey(today);
  const tomorrowKey = addDaysKey(1, today);
  if (value === todayKey) return "Today";
  if (value === tomorrowKey) return "Tomorrow";
  const parsed = new Date(`${value}T12:00:00`);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    ...(parsed.getFullYear() !== today.getFullYear() ? { year: "numeric" } : {}),
  }).format(parsed);
}

