import React from "react";

type Props = {
  onFiles: (files: FileList) => void;
};

export default function FileDrop({ onFiles }: Props) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  return (
    <div className="card">
      <div className="row">
        <input
          ref={inputRef}
          type="file"
          multiple
          onChange={(e) => e.target.files && onFiles(e.target.files)}
        />
      </div>
      <div className="mt" style={{ color: "var(--muted)" }}>
        Upload one or more CSV/XLS/PNG/JPG files.
      </div>
    </div>
  );
}
