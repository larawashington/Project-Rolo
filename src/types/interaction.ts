/**
 * TypeScript mirrors of Rust interaction engine types.
 *
 * These MUST stay in sync with `src-tauri/src/interaction.rs`.
 * The interaction engine is the foundation for all of Rolo's
 * conversational features — mood check-ins today, chatbot tomorrow.
 * A mismatch between these types and their Rust counterparts
 * would leave Rolo unable to communicate, which is unacceptable.
 */

// ---------------------------------------------------------------------------
// Prompt elements — the building blocks of what Rolo says / asks
// ---------------------------------------------------------------------------

/** A single button definition within a ButtonRow. */
export interface ButtonDef {
  label: string;
  value: string;
  style?: "primary" | "secondary";
}

/** Plain text element — Rolo's words. */
export interface TextElement {
  type: "text";
  content: string;
}

/** A row of buttons rendered vertically (full-width each). */
export interface ButtonRowElement {
  type: "button_row";
  buttons: ButtonDef[];
}

/** A text input field for free-form user responses. */
export interface TextInputElement {
  type: "text_input";
  placeholder?: string;
  max_length?: number;
}

/**
 * Discriminated union of all interaction element types.
 * The `kind` field determines which variant is active.
 */
export type InteractionElement = TextElement | ButtonRowElement | TextInputElement;

// ---------------------------------------------------------------------------
// Prompt — the full interaction Rolo presents to the user
// ---------------------------------------------------------------------------

/**
 * A complete interaction prompt emitted by the Rust backend.
 *
 * - `interaction_id`: identifies the *type* of interaction (e.g., "mood_checkin")
 * - `instance_id`: unique ID for this specific occurrence (for response routing)
 * - `elements`: ordered list of UI elements to render vertically
 * - `timeout_ms`: optional auto-dismiss timer (milliseconds)
 */
export interface InteractionPrompt {
  interaction_id: string;
  instance_id: string;
  elements: InteractionElement[];
  timeout_ms?: number;
}

// ---------------------------------------------------------------------------
// Response — what the user sends back
// ---------------------------------------------------------------------------

/**
 * Wire vocabulary for the mood check-in prompt.
 *
 * Mirrors Rust's `interaction::Mood` enum (`#[serde(rename_all = "snake_case")]`).
 * Use this union when narrowing a `ButtonPressResponse.value` to the
 * two-answer mood prompt; non-mood buttons keep `value: string`.
 */
export type Mood = "good" | "ok";

/**
 * Stable identifier for the daily mood check-in prompt
 * (`InteractionPrompt.interaction_id`). Mirrors
 * `INTERACTION_ID_MOOD_CHECKIN` in `src-tauri/src/interaction.rs` so the
 * frontend never compares against a bare `"mood_checkin"` literal.
 */
export const INTERACTION_ID_MOOD_CHECKIN = "mood_checkin";

/** User clicked a button. */
export interface ButtonPressResponse {
  type: "button_press";
  value: string;
}

/** User submitted free text. */
export interface TextSubmitResponse {
  type: "text_submit";
  text: string;
}

/** User dismissed the interaction without responding. */
export interface DismissedResponse {
  type: "dismissed";
}

/**
 * Discriminated union of all response value types.
 * The `kind` field determines which variant is active.
 */
export type ResponseValue = ButtonPressResponse | TextSubmitResponse | DismissedResponse;

/**
 * The full response payload sent back to the Rust backend
 * via the `submit_interaction_response` command.
 */
export interface InteractionResponse {
  instance_id: string;
  interaction_id: string;
  response: ResponseValue;
  timestamp: string;
}
