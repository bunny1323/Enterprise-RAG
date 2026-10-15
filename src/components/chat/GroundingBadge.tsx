import { ShieldCheck, ShieldAlert } from "lucide-react";

export function GroundingBadge({ hallucinationStatus }: { hallucinationStatus?: boolean }) {
  if (hallucinationStatus === undefined || hallucinationStatus === null) return null;

  if (hallucinationStatus) {
    return (
      <div className="flex items-start gap-2 rounded-md bg-amber-soft px-3 py-2 text-sm text-amber">
        <ShieldAlert className="h-4 w-4 mt-0.5 shrink-0" />
        <div>
          <p className="font-medium">Verification warning</p>
          <p className="text-xs mt-0.5 text-ink-soft">
            Some generated claims may not be fully supported by the retrieved
            documents.
          </p>
        </div>
      </div>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-good-soft px-2.5 py-1 text-xs font-medium text-good">
      <ShieldCheck className="h-3 w-3" />
      Grounded response
    </span>
  );
}
