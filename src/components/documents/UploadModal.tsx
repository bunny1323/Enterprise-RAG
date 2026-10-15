import { useState } from "react";
import { Modal } from "../common/Modal";
import { UploadDropzone } from "./UploadDropzone";
import { INDUSTRY_OPTIONS } from "../../types/document";

interface UploadModalProps {
  open: boolean;
  onClose: () => void;
  onUpload: (
    file: File,
    industry: string,
    title?: string,
    documentType?: string
  ) => Promise<unknown>;
}

const DOCUMENT_TYPES = [
  "Technical Manual",
  "Policy Document",
  "Financial Report",
  "Contract",
  "SOP",
  "Other",
];

export function UploadModal({ open, onClose, onUpload }: UploadModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [industry, setIndustry] = useState<string>(INDUSTRY_OPTIONS[0]);
  const [title, setTitle] = useState("");
  const [documentType, setDocumentType] = useState(DOCUMENT_TYPES[0]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setFile(null);
    setTitle("");
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Choose a file to upload.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await onUpload(file, industry, title || undefined, documentType);
      reset();
      onClose();
    } catch (err) {
      setError((err as Error).message || "Document upload failed.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={() => {
        reset();
        onClose();
      }}
      title="Upload document"
    >
      <div className="space-y-4">
        <UploadDropzone file={file} onSelect={setFile} />

        <div>
          <label className="block text-sm font-medium text-ink mb-1">
            Industry
          </label>
          <select
            value={industry}
            onChange={(e) => setIndustry(e.target.value)}
            className="w-full rounded-md border border-line-strong bg-surface px-3 py-2 text-sm text-ink focus:border-signal"
          >
            {INDUSTRY_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt.charAt(0).toUpperCase() + opt.slice(1)}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-ink mb-1">
            Document title <span className="text-ink-faint">(optional)</span>
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Excavator Manual"
            className="w-full rounded-md border border-line-strong bg-surface px-3 py-2 text-sm text-ink focus:border-signal"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-ink mb-1">
            Document type
          </label>
          <select
            value={documentType}
            onChange={(e) => setDocumentType(e.target.value)}
            className="w-full rounded-md border border-line-strong bg-surface px-3 py-2 text-sm text-ink focus:border-signal"
          >
            {DOCUMENT_TYPES.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>

        {error && <p className="text-sm text-bad">{error}</p>}

        <div className="flex justify-end gap-2 pt-2">
          <button
            onClick={() => {
              reset();
              onClose();
            }}
            className="rounded-md border border-line-strong px-4 py-2 text-sm font-medium text-ink hover:bg-surface-wash transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="rounded-md bg-signal px-4 py-2 text-sm font-medium text-surface hover:bg-signal-dark transition-colors disabled:opacity-60"
          >
            {submitting ? "Uploading…" : "Upload"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
