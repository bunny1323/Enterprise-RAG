import { NavLink } from "react-router-dom";
import {
  LayoutGrid,
  FileStack,
  Sparkles,
  Radar,
  Activity,
  Gauge,
  Settings,
} from "lucide-react";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutGrid },
  { to: "/documents", label: "Documents", icon: FileStack },
  { to: "/chat", label: "Ask AI", icon: Sparkles },
  { to: "/retrieval", label: "Retrieval", icon: Radar },
  { to: "/health", label: "System Health", icon: Activity },
  { to: "/evaluation", label: "Evaluation", icon: Gauge },
];

export function Sidebar() {
  return (
    <aside className="hidden md:flex md:w-60 shrink-0 flex-col border-r border-line bg-surface">
      <div className="px-5 py-5 border-b border-line">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-md bg-signal flex items-center justify-center">
            <span className="text-surface text-xs font-semibold font-mono">
              RAG
            </span>
          </div>
          <span className="font-semibold text-ink tracking-tight">
            Enterprise-RAG
          </span>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
                isActive
                  ? "bg-signal-soft text-signal-dark font-medium"
                  : "text-ink-soft hover:bg-surface-wash hover:text-ink"
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-3 py-4 border-t border-line">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            clsx(
              "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
              isActive
                ? "bg-signal-soft text-signal-dark font-medium"
                : "text-ink-soft hover:bg-surface-wash hover:text-ink"
            )
          }
        >
          <Settings className="h-4 w-4" />
          Settings
        </NavLink>
      </div>
    </aside>
  );
}
