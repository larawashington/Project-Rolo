import { useEffect, useRef } from "react";

interface ConfirmationBubbleProps {
  files: string[];
  onConfirm: () => void;
  onDecline: () => void;
  timeoutMs: number;
}

/** Truncate a filename to maxLen characters, preserving the extension. */
function truncateFilename(name: string, maxLen = 25): string {
  if (name.length <= maxLen) return name;

  const dotIndex = name.lastIndexOf(".");
  if (dotIndex > 0 && name.length - dotIndex <= 5) {
    // Has an extension (up to 4 chars + dot)
    const ext = name.slice(dotIndex);
    const keep = maxLen - ext.length - 3; // 3 for "..."
    return name.slice(0, Math.max(keep, 4)) + "..." + ext;
  }

  // No extension or very long extension
  return name.slice(0, maxLen - 3) + "...";
}

/** Extract just the filename from a full file path. */
function basename(path: string): string {
  const sep = path.includes("\\") ? "\\" : "/";
  return path.split(sep).pop() ?? path;
}

/** Format the file list for display in the bubble. */
function formatFiles(files: string[]): { prompt: string; fileList: string[] } {
  const names = files.map((f) => truncateFilename(basename(f)));

  if (files.length === 1) {
    return {
      prompt: `Feed me ${names[0]}?`,
      fileList: [],
    };
  }

  const maxDisplay = 4;
  const displayed = names.slice(0, maxDisplay);
  const remaining = files.length - maxDisplay;

  return {
    prompt: "Feed me these?",
    fileList:
      remaining > 0
        ? [...displayed, `...and ${remaining} more`]
        : displayed,
  };
}

export function ConfirmationBubble({
  files,
  onConfirm,
  onDecline,
  timeoutMs,
}: ConfirmationBubbleProps) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const declinedRef = useRef(false);

  useEffect(() => {
    declinedRef.current = false;

    timerRef.current = setTimeout(() => {
      if (!declinedRef.current) {
        declinedRef.current = true;
        onDecline();
      }
    }, timeoutMs);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [timeoutMs, onDecline]);

  const handleYum = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    onConfirm();
  };

  const handleNah = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (!declinedRef.current) {
      declinedRef.current = true;
      onDecline();
    }
  };

  const { prompt, fileList } = formatFiles(files);

  return (
    <div className="bubble-container">
      <div className="bubble-content">
        <div className="bubble-prompt">{prompt}</div>
        {fileList.length > 0 && (
          <div>
            {fileList.map((name, i) => (
              <div key={i} className="bubble-file">
                {name}
              </div>
            ))}
          </div>
        )}
        <div className="bubble-buttons">
          <button className="btn-yum" onClick={handleYum}>
            Yum!
          </button>
          <button className="btn-nah" onClick={handleNah}>
            Nah
          </button>
        </div>
      </div>
      <div className="bubble-tail" />
    </div>
  );
}
