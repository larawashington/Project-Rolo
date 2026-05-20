/**
 * MemoryTab — Phase 7 of the Rolo Command Center PRD.
 *
 * Three textareas (About You, People & Context, How Rolo Should Respond) let
 * the user write things Rolo should know about them. Saving routes through
 * `cc_save_memory_and_sleep` which:
 *   1. atomically writes `user_profile.json`,
 *   2. appends one `UserProfileUpdate` event per non-empty section,
 *   3. drives a one-shot dream cycle through the compiler/linter pipeline.
 *
 * The SleepOverlay watches `rolo://command-center-dream-progress` and shows
 * stage progress until the save promise resolves. On success, Rolo wakes and
 * queues a hardcoded reaction bubble.
 *
 * Per-section input is hard-capped at 2000 characters in the `onChange`
 * handler — input past the cap is dropped, not truncated mid-character (we
 * slice by Unicode code units via `Array.from`). The backend re-enforces the
 * cap for defense in depth.
 */

import { invoke } from "@tauri-apps/api/core";
import { useTauriEvent } from "../../hooks/useTauriEvent";
import { useCallback, useEffect, useRef, useState } from "react";
import type { CommandCenterTabProps } from "./CommandCenter";
import type {
  ClearProfileResult,
  LoadProfileResult,
  SaveMemoryResult,
} from "../../types/commandCenter";
import SleepOverlay from "./SleepOverlay";

/** Hard char cap per section. Mirrors `MEMORY_SECTION_CHAR_CAP` in commands.rs. */
const SECTION_CAP = 2000;
/** Counter goes amber at this threshold. */
const SECTION_AMBER = 1800;

/** Three-section memory form state. */
interface MemoryFormState {
  aboutYou: string;
  peopleAndContext: string;
  howRoloShouldRespond: string;
}

const EMPTY_FORM: MemoryFormState = {
  aboutYou: "",
  peopleAndContext: "",
  howRoloShouldRespond: "",
};

/**
 * Slice an input string to at most `cap` Unicode scalar values. JavaScript's
 * `slice` operates on UTF-16 code units, which would cut multi-code-unit
 * characters (most emoji, some CJK) in half — `Array.from` enumerates by
 * code points and `.slice().join("")` reassembles cleanly.
 */
function capByCodePoints(text: string, cap: number): string {
  // Fast path — ASCII-only text avoids the Array.from allocation.
  if (text.length <= cap) return text;
  return Array.from(text).slice(0, cap).join("");
}

/** Length of `text` measured in Unicode code points. Matches Rust's `chars().count()`. */
function codePointLength(text: string): number {
  // Array.from iterates by code point; `[...text]` does the same.
  let n = 0;
  for (const _ of text) n += 1;
  return n;
}

/** Banner shown above the form. Auto-clears for success after 3 seconds. */
type BannerVariant = "success" | "amber" | "red";
interface Banner {
  variant: BannerVariant;
  text: string;
}

export default function MemoryTab(props: CommandCenterTabProps) {
  const { markDirty, markClean } = props;
  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------
  const [form, setForm] = useState<MemoryFormState>(EMPTY_FORM);
  const [savedProfileEcho, setSavedProfileEcho] =
    useState<MemoryFormState | null>(null);
  const [learnedTitles, setLearnedTitles] = useState<string[]>([]);
  const [showLearned, setShowLearned] = useState<boolean>(false);
  const [sleeping, setSleeping] = useState<boolean>(false);
  const [banner, setBanner] = useState<Banner | null>(null);
  const [confirmClear, setConfirmClear] = useState<boolean>(false);
  const successTimerRef = useRef<number | null>(null);

  // -------------------------------------------------------------------------
  // Load (mount + on `rolo://command-center-opened`)
  // -------------------------------------------------------------------------
  const reloadProfile = useCallback(async () => {
    try {
      const result = await invoke<LoadProfileResult>("cc_load_user_profile");
      if (result.profile) {
        const next: MemoryFormState = {
          aboutYou: result.profile.about_you,
          peopleAndContext: result.profile.people_and_context,
          howRoloShouldRespond: result.profile.how_rolo_should_respond,
        };
        // Populate the form on first load only — once the user starts
        // typing, their edits are the source of truth and we don't want a
        // later reload (e.g. tab re-open) to clobber unsaved work.
        setForm((prev) => {
          const allEmpty =
            prev.aboutYou === "" &&
            prev.peopleAndContext === "" &&
            prev.howRoloShouldRespond === "";
          return allEmpty ? next : prev;
        });
        setSavedProfileEcho(next);
      } else {
        setSavedProfileEcho(null);
      }
      setLearnedTitles(result.learned_dream_titles ?? []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setBanner({
        variant: "red",
        text: `Couldn't load Rolo's memory of you: ${msg}`,
      });
    }
  }, []);

  useEffect(() => {
    void reloadProfile();
  }, [reloadProfile]);

  // External refresh — tray menu, another window, etc. re-open of CC.
  useTauriEvent("rolo://command-center-opened", () => {
    void reloadProfile();
  });

  // Discard event — wipe the in-memory form and pull saved values off
  // disk. Unlike `command-center-opened`, this bypasses the
  // "don't clobber unsaved edits" guard because the user explicitly
  // chose Discard.
  useTauriEvent("rolo://command-center-discard", () => {
    setForm(EMPTY_FORM);
    void reloadProfile().then(() => {
      setSavedProfileEcho((cur) => {
        if (cur) setForm(cur);
        return cur;
      });
    });
  });

  // Clean up the success-banner timer on unmount.
  useEffect(() => {
    return () => {
      if (successTimerRef.current !== null) {
        window.clearTimeout(successTimerRef.current);
        successTimerRef.current = null;
      }
    };
  }, []);

  // -------------------------------------------------------------------------
  // Field changes
  // -------------------------------------------------------------------------
  const setField = (key: keyof MemoryFormState, value: string) => {
    const capped = capByCodePoints(value, SECTION_CAP);
    setForm((prev) => ({ ...prev, [key]: capped }));
    // Phase 9: any textarea edit flips the tab's dirty flag. We mark on
    // every keystroke rather than after a debounce — the close-handler is
    // synchronous and we want it to catch even a single keystroke of edit.
    markDirty("memory");
  };

  const allEmpty =
    form.aboutYou.trim() === "" &&
    form.peopleAndContext.trim() === "" &&
    form.howRoloShouldRespond.trim() === "";

  // -------------------------------------------------------------------------
  // Save and Sleep
  // -------------------------------------------------------------------------
  const handleSave = async () => {
    if (sleeping || allEmpty) return;
    setBanner(null);
    if (successTimerRef.current !== null) {
      window.clearTimeout(successTimerRef.current);
      successTimerRef.current = null;
    }
    setSleeping(true);
    try {
      const result = await invoke<SaveMemoryResult>(
        "cc_save_memory_and_sleep",
        {
          req: {
            about_you: form.aboutYou,
            people_and_context: form.peopleAndContext,
            how_rolo_should_respond: form.howRoloShouldRespond,
          },
        },
      );

      // Refresh the "What Rolo has learned" echo + the learned-titles list
      // regardless of dream outcome — the profile file itself was saved
      // before the dream ran, so the echo should reflect that immediately.
      await reloadProfile();

      if (result.status === "success") {
        setBanner({ variant: "success", text: "Rolo absorbed it." });
        // Phase 9: a successful Save+Sleep cleans the memory tab. The
        // skipped/cancelled/failed branches keep the tab dirty so the
        // user knows their unsaved edits weren't fully absorbed.
        markClean("memory");
        successTimerRef.current = window.setTimeout(() => {
          setBanner(null);
          successTimerRef.current = null;
        }, 3000);
      } else if (
        result.status === "skipped" &&
        result.reason === "already_running"
      ) {
        setBanner({
          variant: "amber",
          text: "Rolo is already dreaming — try again in a moment.",
        });
      } else if (result.status === "cancelled") {
        setBanner({
          variant: "amber",
          text: "Sleep was cancelled — your profile is saved.",
        });
      } else {
        setBanner({
          variant: "red",
          text: "Sleep was interrupted — content saved, but Rolo didn't get to dream about it. Try Save and Sleep again.",
        });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setBanner({ variant: "red", text: msg });
    } finally {
      setSleeping(false);
    }
  };

  // -------------------------------------------------------------------------
  // Clear User Profile
  // -------------------------------------------------------------------------
  const handleClear = async () => {
    setConfirmClear(false);
    setBanner(null);
    try {
      const result = await invoke<ClearProfileResult>("cc_clear_user_profile");
      setForm(EMPTY_FORM);
      setSavedProfileEcho(null);
      setLearnedTitles([]);
      // Phase 9: clearing the profile is the same outcome as a successful
      // save (the user's intent — the form's "saved" state — now matches
      // disk). Drop the dirty mark so close doesn't prompt.
      markClean("memory");
      const suffix =
        result.dreams_removed > 0
          ? ` (removed ${result.dreams_removed} dream line${result.dreams_removed === 1 ? "" : "s"})`
          : "";
      setBanner({
        variant: "success",
        text:
          result.status === "cancel_then_clear"
            ? `Cancelled the active dream and cleared Rolo's memory of you.${suffix}`
            : `Cleared Rolo's memory of you.${suffix}`,
      });
      successTimerRef.current = window.setTimeout(() => {
        setBanner(null);
        successTimerRef.current = null;
      }, 3000);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setBanner({ variant: "red", text: msg });
    }
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="cc-tab cc-tab-memory">
      <h2 className="cc-tab-title">Memory</h2>

      {banner && (
        <div className={`cc-banner cc-banner-${banner.variant}`}>
          {banner.text}
        </div>
      )}

      <div className="cc-memory-form">
        <MemorySection
          id="about-you"
          label="About You"
          placeholder="Tell Rolo who you are."
          value={form.aboutYou}
          onChange={(v) => setField("aboutYou", v)}
          disabled={sleeping}
        />
        <MemorySection
          id="people-and-context"
          label="People & Context"
          placeholder="People in your life Rolo should know."
          value={form.peopleAndContext}
          onChange={(v) => setField("peopleAndContext", v)}
          disabled={sleeping}
        />
        <MemorySection
          id="how-rolo-should-respond"
          label="How Rolo Should Respond"
          placeholder="Anything Rolo should remember about how you like to be talked to."
          value={form.howRoloShouldRespond}
          onChange={(v) => setField("howRoloShouldRespond", v)}
          disabled={sleeping}
        />

        <div className="cc-form-actions">
          <button
            type="button"
            className="cc-button cc-button-primary cc-button-wide"
            onClick={() => void handleSave()}
            disabled={sleeping || allEmpty}
          >
            {sleeping ? "Sleeping…" : "Save and Sleep"}
          </button>
        </div>
      </div>

      {/* What Rolo has learned about you — collapsible. */}
      <section className="cc-collapsible">
        <button
          type="button"
          className="cc-collapsible-toggle"
          onClick={() => setShowLearned((v) => !v)}
          aria-expanded={showLearned}
        >
          <span
            className={
              "cc-collapsible-caret" +
              (showLearned ? " cc-collapsible-caret-open" : "")
            }
          >
            ›
          </span>
          What Rolo has learned about you
        </button>
        {showLearned && (
          <div className="cc-collapsible-body">
            {savedProfileEcho ? (
              <div className="cc-learned-echo">
                {savedProfileEcho.aboutYou && (
                  <EchoBlock label="About You" text={savedProfileEcho.aboutYou} />
                )}
                {savedProfileEcho.peopleAndContext && (
                  <EchoBlock
                    label="People & Context"
                    text={savedProfileEcho.peopleAndContext}
                  />
                )}
                {savedProfileEcho.howRoloShouldRespond && (
                  <EchoBlock
                    label="How Rolo Should Respond"
                    text={savedProfileEcho.howRoloShouldRespond}
                  />
                )}
                {!savedProfileEcho.aboutYou &&
                  !savedProfileEcho.peopleAndContext &&
                  !savedProfileEcho.howRoloShouldRespond && (
                    <p className="cc-learned-empty">
                      Your saved profile is empty.
                    </p>
                  )}
              </div>
            ) : (
              <p className="cc-learned-empty">
                Rolo hasn't been told anything about you yet.
              </p>
            )}
            {learnedTitles.length > 0 && (
              <div className="cc-learned-titles">
                <h4 className="cc-learned-titles-header">Dream entries</h4>
                <ul className="cc-learned-titles-list">
                  {learnedTitles.map((t) => (
                    <li key={t}>{t}</li>
                  ))}
                </ul>
              </div>
            )}
            <div className="cc-form-actions">
              {confirmClear ? (
                <>
                  <span className="cc-learned-confirm">
                    Clear everything Rolo has learned about you? This deletes
                    your saved profile and any dreams tagged from it.
                  </span>
                  <button
                    type="button"
                    className="cc-button cc-button-danger"
                    onClick={() => void handleClear()}
                  >
                    Yes, clear
                  </button>
                  <button
                    type="button"
                    className="cc-button cc-button-secondary"
                    onClick={() => setConfirmClear(false)}
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="cc-button cc-button-secondary"
                  onClick={() => setConfirmClear(true)}
                >
                  Clear User Profile
                </button>
              )}
            </div>
          </div>
        )}
      </section>

      {sleeping && (
        <SleepOverlay
          initialLabel="Writing memory…"
          onCancel={() => {
            // The Save call is still in flight; once its promise resolves
            // with a `cancelled` outcome we tear down via setSleeping(false)
            // in the finally block. We don't flip sleeping here ourselves
            // to avoid showing the form behind the still-pending invoke.
          }}
        />
      )}
    </div>
  );
}

interface MemorySectionProps {
  id: string;
  label: string;
  placeholder: string;
  value: string;
  onChange: (next: string) => void;
  disabled: boolean;
}

function MemorySection({
  id,
  label,
  placeholder,
  value,
  onChange,
  disabled,
}: MemorySectionProps) {
  const len = codePointLength(value);
  const counterClass =
    len >= SECTION_CAP
      ? "cc-counter cc-counter-red"
      : len >= SECTION_AMBER
        ? "cc-counter cc-counter-amber"
        : "cc-counter";
  return (
    <div className="cc-memory-section">
      <label htmlFor={id} className="cc-memory-section-label">
        {label}
      </label>
      <textarea
        id={id}
        className="cc-textarea"
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        rows={4}
        spellCheck
      />
      <div className={counterClass}>
        {len} / ~{SECTION_CAP}
      </div>
    </div>
  );
}

function EchoBlock({ label, text }: { label: string; text: string }) {
  return (
    <div className="cc-echo-block">
      <div className="cc-echo-label">{label}</div>
      <div className="cc-echo-text">{text}</div>
    </div>
  );
}
