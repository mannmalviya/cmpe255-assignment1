import { TaskWorkspace } from "@/components/task-workspace";
import { getTasks } from "@/lib/db";
import { localDateKey } from "@/lib/date";

export const dynamic = "force-dynamic";

export default function Home() {
  return <TaskWorkspace initialTasks={getTasks()} today={localDateKey()} />;
}

