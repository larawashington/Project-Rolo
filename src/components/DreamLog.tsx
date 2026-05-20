/**
 * DreamLog — pixel-art panel surfacing Rolo's recent dream-compile runs.
 *
 * Fetches up to 30 entries from `read_dream_log`, renders them newest-first
 * with timestamp, duration, fact count, and (for compile runs) per-fact rows
 * with [revert] buttons. Lint and manual_revert entries pass through with
 * less detail. PRD §7C / §O4.
 */

import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./DreamLog.css";

interface AcceptedFact {
  file?: string;
  content?: string;
  source_event_ids?: string[];
}

interface DreamRunSummary {
  run_id: string;
  status: string;
  started_at: string | null;
  ended_at: string | null;
  latency_ms: number | null;
  facts_accepted: number | null;
  facts_rejected: number | null;
  /** "lint" / "manual_revert" / null for compile runs. */
  run_type: string | null;
  parent_run_id: string | null;
  reason: string | null;
  /** Loaded from the per-run artifact when available. */
  accepted_facts: AcceptedFact[] | null;
  raw: Record<string, unknown>;
}

function formatTimestamp(iso: string | null, fallback: string): string {
  if (!iso) return fallback;
  // Trim seconds — "2026-05-04T14:30" reads better in a pixel-font row.
  // Local timezones rendered as-is; we trust the backend's RFC3339 string.
  const idx = iso.indexOf("T");
  if (idx === -1) return iso;
  return `${iso.slice(0, idx)} ${iso.slice(idx + 1, idx + 6)}`;
}

function formatDuration(ms: number | null): string | null {
  if (ms == null) return null;
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function runLabel(run: DreamRunSummary): string {
  if (run.run_type === "lint") return "lint";
  if (run.run_type === "manual_revert") return "revert";
  return "dream";
}

export default function DreamLog() {
  const [runs, setRuns] = useState<DreamRunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    try {
      const data = await invoke<DreamRunSummary[]>("read_dream_log", { limit: 30 });
      setRuns(data);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const handleRevert = async (runId: string, factIndex: number) => {
    try {
      await invoke("revert_fact", { runId, factIndex });
      await refresh();
    } catch (e) {
      setError(`Revert failed: ${e}`);
    }
  };

  if (loading) {
    return <div className="dreamlog-empty">Loading...</div>;
  }
  if (error) {
    return <div className="dreamlog-error">{error}</div>;
  }
  if (runs.length === 0) {
    return (
      <div className="dreamlog-empty">
        No dreams yet. Rolo will dream once he has things to remember.
      </div>
    );
  }

  return (
    <div className="dreamlog-panel">
      {runs.map((run) => (
        <div
          key={`${run.run_id}-${run.run_type ?? "compile"}`}
          className={`dreamlog-run dreamlog-${run.status}`}
        >
          <div className="dreamlog-header">
            <span className="dreamlog-time">
              {formatTimestamp(run.started_at, run.run_id || "—")}
            </span>
            <span className="dreamlog-status">
              {runLabel(run)} · {run.status}
            </span>
            {run.latency_ms != null && (
              <span className="dreamlog-latency">
                {formatDuration(run.latency_ms)}
              </span>
            )}
          </div>
          {run.facts_accepted != null && (
            <div className="dreamlog-meta">
              {run.facts_accepted} learned, {run.facts_rejected ?? 0} rejected
            </div>
          )}
          {Array.isArray(run.accepted_facts) && run.accepted_facts.length > 0 && (
            <ul className="dreamlog-facts">
              {run.accepted_facts.map((fact, i) => (
                <li key={i}>
                  <div className="dreamlog-fact-body">
                    <span className="dreamlog-fact-file">{fact.file ?? "?"}</span>
                    <span className="dreamlog-fact-content">{fact.content ?? ""}</span>
                  </div>
                  <button
                    className="dreamlog-revert"
                    onClick={() => handleRevert(run.run_id, i)}
                    title="Revert this fact — it will be commented out in the wiki"
                  >
                    revert
                  </button>
                </li>
              ))}
            </ul>
          )}
          {run.reason && (
            <div className="dreamlog-reason">reason: {run.reason}</div>
          )}
        </div>
      ))}
    </div>
  );
}
