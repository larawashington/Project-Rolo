/**
 * PerceptionTab — diagnostic status for Rolo's senses.
 *
 * Renders the ordered probe list from `cc_run_diagnostics`. Each row is a
 * status dot, label, one-line message (and optional latency), with an
 * expand-on-click chevron revealing the fix hint when red/amber.
 *
 * Auto-refreshes on mount and whenever `rolo://command-center-opened`
 * fires (so a tray-menu open from the user always pulls fresh diagnostics).
 * A "Refresh all" button at the top re-runs the probes on demand.
 */

import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTauriEvent } from "../../hooks/useTauriEvent";

import type { CommandCenterTabProps } from "./CommandCenter";
import type {
  DiagnosticsReport,
  ProbeOutcome,
} from "../../types/commandCenter";

export default function PerceptionTab(_props: CommandCenterTabProps) {
  const [report, setReport] = useState<DiagnosticsReport | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  // Run the diagnostics. On error, keep the previous report visible and
  // surface a banner — Rolo wants the user to retain the last-known state
  // rather than stare at an empty panel.
  const runDiagnostics = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const next = await invoke<DiagnosticsReport>("cc_run_diagnostics");
      setReport(next);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  }, []);

  // Initial run on mount.
  useEffect(() => {
    void runDiagnostics();
  }, [runDiagnostics]);

  // Re-run whenever the window is opened externally (right-click menu,
  // tray, etc.).
  useTauriEvent("rolo://command-center-opened", () => {
    void runDiagnostics();
  });

  const toggleExpanded = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  return (
    <div className="cc-tab cc-tab-perception">
      <div className="cc-perception-header">
        <h2 className="cc-tab-title">Perception</h2>
        <button
          type="button"
          className="cc-refresh-button"
          onClick={() => void runDiagnostics()}
          disabled={refreshing}
        >
          {refreshing ? "Refreshing…" : "Refresh all"}
        </button>
      </div>

      {error !== null && (
        <div className="cc-error-banner" role="alert">
          Diagnostic run failed: {error}
        </div>
      )}

      {loading && report === null ? (
        <p className="cc-tab-placeholder">Running diagnostics…</p>
      ) : (
        <ul className="cc-probe-list">
          {report?.probes.map((probe) => (
            <ProbeRow
              key={probe.id}
              probe={probe}
              expanded={expanded.has(probe.id)}
              onToggle={() => toggleExpanded(probe.id)}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

interface ProbeRowProps {
  probe: ProbeOutcome;
  expanded: boolean;
  onToggle: () => void;
}

function ProbeRow({ probe, expanded, onToggle }: ProbeRowProps) {
  // Red/Amber rows show their fix hint when present; Green rows fall back to
  // the message (no fix needed → just echo).
  const detail: string = probe.fix_hint ?? probe.message;

  return (
    <li className={"cc-probe-row cc-probe-status-" + probe.status}>
      <button
        type="button"
        className="cc-probe-row-button"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <span className={"cc-probe-dot cc-probe-dot-" + probe.status} />
        <span className="cc-probe-label">{probe.label}</span>
        <span className="cc-probe-message">{probe.message}</span>
        {probe.latency_ms !== null && (
          <span className="cc-probe-latency">({probe.latency_ms}ms)</span>
        )}
        <span
          className={
            "cc-probe-chevron" + (expanded ? " cc-probe-chevron-open" : "")
          }
          aria-hidden="true"
        >
          ▸
        </span>
      </button>
      {expanded && <div className="cc-probe-detail">{detail}</div>}
    </li>
  );
}
