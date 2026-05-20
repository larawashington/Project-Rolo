//! Rolo's spatial constants — the single source of truth for every
//! measurement that positions Rolo or UI elements relative to him.
//!
//! Import from here whenever you need to place something near Rolo.
//! Never hardcode window sizes, sprite sizes, or clearances elsewhere.

// ---------------------------------------------------------------------------
// Rolo's body
// ---------------------------------------------------------------------------

/// Rolo's window size in logical pixels (oversized transparent container
/// that holds the sprite and leaves room for layout mode transitions).
pub const WINDOW_SIZE: u32 = 320;

/// Rolo's sprite size in logical pixels (the visible character).
pub const SPRITE_SIZE: u32 = 128;

/// Offset from the bottom-right screen edge for Rolo's starting position.
pub const EDGE_OFFSET: u32 = 100;

/// Physical-pixel hysteresis band for Normal ↔ TopAnchored transitions.
/// Prevents flickering when the sprite hovers near the threshold.
pub const LAYOUT_HYSTERESIS: i32 = 32;

// ---------------------------------------------------------------------------
// Speech bubble (separate "speech-bubble" Tauri window)
// ---------------------------------------------------------------------------

/// Gap above Rolo's head for the speech bubble in normal position,
/// measured from the bubble *window's* bottom edge (NOT the visible body)
/// to the sprite top.
///
/// Tuned so the visible bubble body lands 8 logical px above the sprite —
/// matching `CONFIRM_GAP` so the speech bubble and the in-world confirmation
/// bubble share the same visual gap above Rolo.
///
/// Why the negative number: the speech-bubble window pins
/// `.speech-bubble-container` to its bottom edge via `align-items: flex-end`.
/// The container has `padding-bottom: 9px` (see `src/components/SpeechBubble.css`)
/// reserving room for the tail's box-shadow, so the visible body's bottom
/// border sits 9 logical px ABOVE the window's bottom edge. To put that body
/// 8 px above the sprite, the window's bottom must sit 1 px BELOW the sprite
/// top → `8 - 9 = -1`. The tail tip ends up ~1 px inside Rolo's transparent
/// hood-apex pixels (invisible), mirroring how the confirmation tail nearly
/// touches the sprite.
///
/// If `padding-bottom` in `SpeechBubble.css` ever changes, adjust this by
/// the same amount.
pub const SPEECH_GAP_ABOVE: i32 = -1;

/// Gap below Rolo's feet for the speech bubble in flipped position.
pub const SPEECH_GAP_BELOW: i32 = 4;

/// Maximum bubble width in logical pixels.
pub const SPEECH_MAX_WIDTH: i32 = 250;

/// Maximum bubble body height in logical pixels (text area, without tail).
pub const SPEECH_MAX_HEIGHT: i32 = 100;

/// Extra window height to contain the tail's box-shadow overflow.
pub const SPEECH_TAIL_OVERFLOW: i32 = 12;

/// Total bubble window height: body + tail overflow.
pub const SPEECH_WINDOW_HEIGHT: i32 = SPEECH_MAX_HEIGHT + SPEECH_TAIL_OVERFLOW;

// ---------------------------------------------------------------------------
// Confirmation bubble (inside 320x320 main window)
// ---------------------------------------------------------------------------

/// Gap between sprite edge and confirmation bubble (logical px).
pub const CONFIRM_GAP: i32 = 8;

// ---------------------------------------------------------------------------
// Interactive bubble (inside 320x320 main window)
// ---------------------------------------------------------------------------

/// Gap between sprite edge and interactive bubble (logical px).
pub const INTERACT_GAP: i32 = 8;

// ---------------------------------------------------------------------------
// Scaling helpers
// ---------------------------------------------------------------------------

/// Physical-pixel offsets between window top-left and sprite top-left.
/// Returns `(dx, dy)` such that `sprite_pos = window_pos + (dx, dy)`.
pub fn sprite_window_offsets(scale_factor: f64) -> (i32, i32) {
    let phys_window = (f64::from(WINDOW_SIZE) * scale_factor) as i32;
    let phys_pet = (f64::from(SPRITE_SIZE) * scale_factor) as i32;
    ((phys_window - phys_pet) / 2, phys_window - phys_pet)
}

pub fn scaled_speech_width(scale_factor: f64) -> i32 {
    (SPEECH_MAX_WIDTH as f64 * scale_factor).round() as i32
}

pub fn scaled_speech_height(scale_factor: f64) -> i32 {
    (SPEECH_WINDOW_HEIGHT as f64 * scale_factor).round() as i32
}

pub fn scaled_speech_gap_above(scale_factor: f64) -> i32 {
    (SPEECH_GAP_ABOVE as f64 * scale_factor).round() as i32
}

pub fn scaled_speech_gap_below(scale_factor: f64) -> i32 {
    (SPEECH_GAP_BELOW as f64 * scale_factor).round() as i32
}
