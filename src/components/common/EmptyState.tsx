import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: ReactNode;
}

export function EmptyState({ icon: Icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center px-6">
      {Icon && (
        <div className="h-11 w-11 rounded-full bg-surface-wash flex items-center justify-center">
          <Icon className="h-5 w-5 text-ink-faint" />
        </div>
      )}
      <div>
        <p className="font-medium text-ink">{title}</p>
        {description && (
          <p className="text-sm text-ink-soft mt-1 max-w-sm">{description}</p>
        )}
      </div>
      {action}
    </div>
  );
}
