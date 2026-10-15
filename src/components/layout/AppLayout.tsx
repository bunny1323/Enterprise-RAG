import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

interface AppLayoutProps {
  title: string;
  children: ReactNode;
  /** Chat and other full-bleed pages can opt out of the default padded shell. */
  fullBleed?: boolean;
}

export function AppLayout({ title, children, fullBleed }: AppLayoutProps) {
  return (
    <div className="flex h-screen overflow-hidden bg-surface-sunk">
      <Sidebar />
      <div className="flex flex-1 flex-col min-w-0">
        <Topbar title={title} />
        <main className={fullBleed ? "flex-1 min-h-0" : "flex-1 overflow-y-auto px-6 py-6"}>
          {children}
        </main>
      </div>
    </div>
  );
}
