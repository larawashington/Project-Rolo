/**
 * <FormRow> — shared label/control/help/error wrapper used by the Command
 * Center's Brain and Weather tabs.
 *
 * Replaces the ~22 hand-rolled `<div className="cc-form-row">…</div>`
 * fragments that each repeated the same label + helper + error structure.
 * The visual output is identical to the inlined version — same class names,
 * same DOM order — so this is a no-op refactor for the rendered page.
 *
 * Consumers pass:
 *   - `label`     : the user-facing label text.
 *   - `htmlFor`   : optional, the id of the control the label points to.
 *   - `help`      : optional helper paragraph (ReactNode — may contain
 *                   inline code, links, etc.).
 *   - `error`     : optional inline error paragraph rendered after `help`.
 *   - `className` : optional extra classes (e.g., `cc-form-row-inline`).
 *   - `children`  : the actual input / select / textarea.
 *
 * Helper and error slots are rendered only when truthy, so existing rows
 * that didn't have them keep their compact two-element shape.
 */

import { ReactNode } from "react";

export interface FormRowProps {
  label: string;
  htmlFor?: string;
  help?: ReactNode;
  error?: ReactNode;
  /** Extra row-level class names (e.g., `cc-form-row-inline`). */
  className?: string;
  children: ReactNode;
}

export function FormRow({
  label,
  htmlFor,
  help,
  error,
  className,
  children,
}: FormRowProps) {
  const rowClass = ["cc-form-row", className].filter(Boolean).join(" ");
  return (
    <div className={rowClass}>
      <label className="cc-form-label" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
      {help && <p className="cc-form-helper">{help}</p>}
      {error && <p className="cc-form-error">{error}</p>}
    </div>
  );
}
