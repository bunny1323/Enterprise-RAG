import { UploadCloud } from "lucide-react";
import { useRef, useState } from "react";
import clsx from "clsx";

const ACCEPTED = ".pdf,.png,.tif,.tiff";
const MAX_BYTES = 100 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = new Set([".pdf", ".png", ".tif", ".tiff"]);

interface UploadDropzoneProps {
  file: File | null;
  onSelect: (file: File) => void;
}

export function UploadDropzone({ file, onSelect }: UploadDropzoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const chooseFile = (candidate: File) => {
    const extension = `.${candidate.name.split(".").pop()?.toLowerCase() ?? ""}`;
    if (!ACCEPTED_EXTENSIONS.has(extension)) {
      window.alert("Please choose a PDF, PNG, or TIFF file.");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      window.alert("The file must be 100MB or smaller.");
      return;
    }
    onSelect(candidate);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const dropped = e.dataTransfer.files?.[0];
        if (dropped) chooseFile(dropped);
      }}
      className={clsx(
        "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors",
        dragOver
          ? "border-signal bg-signal-soft"
          : "border-line-strong bg-surface-sunk"
      )}
    >
      <UploadCloud className="h-6 w-6 text-ink-faint" />
      {file ? (
        <p className="text-sm font-medium text-ink">{file.name}</p>
      ) : (
        <>
          <p className="text-sm text-ink">Drag & drop your document</p>
          <p className="text-xs text-ink-soft">or</p>
        </>
      )}
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mt-1 rounded-md border border-line-strong bg-surface px-3 py-1.5 text-sm font-medium text-ink hover:bg-surface-wash transition-colors"
      >
        Browse files
      </button>
      <p className="text-xs text-ink-faint mt-1">
        PDF &bull; PNG &bull; TIFF &mdash; up to 100MB
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        className="hidden"
        onChange={(e) => {
          const selected = e.target.files?.[0];
          if (selected) chooseFile(selected);
        }}
      />
    </div>
  );
}
