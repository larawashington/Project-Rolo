use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

/// Mint a fresh `event_id` for a newly-constructed `ExperienceEvent`.
///
/// UUID v7 is time-ordered (lexicographic sort == chronological sort), which
/// is the property the dreaming compiler relies on when stitching events
/// across daily JSONL files (PRD §A1, decision #16).
pub fn new_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Serde default for the `event_id` field — invoked at deserialize time when
/// the field is missing from a legacy JSONL line written before A1 landed.
///
/// The compiler treats `"legacy_unknown"` as a sentinel: it's stable enough
/// to dedupe (we'll never mint two), but distinct enough to flag in audits.
fn synthesize_legacy_id() -> String {
    "legacy_unknown".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExperienceEvent {
    Chat {
        #[serde(default = "synthesize_legacy_id")]
        event_id: String,
        ts: DateTime<Local>,
        session: String,
        user: String,
        rolo: String,
        mood_signal: Option<MoodSignal>,
    },
    Dismiss {
        #[serde(default = "synthesize_legacy_id")]
        event_id: String,
        ts: DateTime<Local>,
        context: DismissContext,
        times_dismissed_session: u32,
    },
    Eat {
        #[serde(default = "synthesize_legacy_id")]
        event_id: String,
        ts: DateTime<Local>,
        // Basenames only — never full paths (privacy: don't leak the user's filesystem layout).
        files: Vec<String>,
        outcome: EatOutcome,
        bytes: u64,
    },
    Checkin {
        #[serde(default = "synthesize_legacy_id")]
        event_id: String,
        ts: DateTime<Local>,
        question: String,
        response: String,
        method: CheckinMethod,
    },
    Drag {
        #[serde(default = "synthesize_legacy_id")]
        event_id: String,
        ts: DateTime<Local>,
        duration_ms: u64,
    },
    IdleSpeech {
        #[serde(default = "synthesize_legacy_id")]
        event_id: String,
        ts: DateTime<Local>,
        text: String,
        // Always false at emit time; the dreaming compiler correlates a later
        // Dismiss(IdleSpeech) within the bubble's display window (see PRD §4.4).
        dismissed: bool,
    },
    Report {
        #[serde(default = "synthesize_legacy_id")]
        event_id: String,
        ts: DateTime<Local>,
        message_id: i64,
        // Verbatim — denormalized so replay does not depend on SQLite still having the row.
        rolo_text: String,
    },
    SessionEnd {
        #[serde(default = "synthesize_legacy_id")]
        event_id: String,
        ts: DateTime<Local>,
        idle_total_ms: u64,
        interactions: u32,
    },
    UserProfileUpdate {
        #[serde(default = "synthesize_legacy_id")]
        event_id: String,
        ts: DateTime<Local>,
        category: UserProfileCategory,
        text: String,
    },
}

/// Which of the three Memory-panel textareas a `UserProfileUpdate` event came
/// from. Drives the `[user_profile]` kind label in the dreaming compile prompt
/// (see PRD §6, Memory panel → Storage and routing).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UserProfileCategory {
    About,
    People,
    ResponseStyle,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MoodSignal {
    Positive,
    Neutral,
    Negative,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DismissContext {
    IdleSpeech,
    InteractionPrompt,
    ChatBubble,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EatOutcome {
    Satisfied,
    Disappointed,
    Errored,
    Declined,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CheckinMethod {
    Button,
    Text,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_ts() -> DateTime<Local> {
        Local.with_ymd_and_hms(2026, 4, 27, 14, 30, 0).unwrap()
    }

    fn round_trip(event: &ExperienceEvent) -> ExperienceEvent {
        let json = serde_json::to_string(event).expect("serialize");
        serde_json::from_str(&json).expect("deserialize")
    }

    fn assert_variant_match(original: &ExperienceEvent, decoded: &ExperienceEvent) {
        // We can't easily PartialEq the whole enum (DateTime equality across TZ),
        // so re-serialize both and compare the wire form — true round-trip check.
        let a = serde_json::to_string(original).unwrap();
        let b = serde_json::to_string(decoded).unwrap();
        assert_eq!(a, b, "round trip changed wire form");
    }

    #[test]
    fn chat_round_trip() {
        let e = ExperienceEvent::Chat {
            event_id: "evt_test_chat_01".into(),
            ts: fixed_ts(),
            session: "abc123".into(),
            user: "how's it going?".into(),
            rolo: "*bounces* pretty good!".into(),
            mood_signal: None,
        };
        assert_variant_match(&e, &round_trip(&e));
    }

    #[test]
    fn chat_with_mood_signal_round_trip() {
        let e = ExperienceEvent::Chat {
            event_id: "evt_test_chat_02".into(),
            ts: fixed_ts(),
            session: "abc123".into(),
            user: "tired".into(),
            rolo: "oof.".into(),
            mood_signal: Some(MoodSignal::Negative),
        };
        assert_variant_match(&e, &round_trip(&e));
    }

    #[test]
    fn dismiss_round_trip() {
        let e = ExperienceEvent::Dismiss {
            event_id: "evt_test_dismiss_01".into(),
            ts: fixed_ts(),
            context: DismissContext::IdleSpeech,
            times_dismissed_session: 3,
        };
        assert_variant_match(&e, &round_trip(&e));
    }

    #[test]
    fn eat_round_trip() {
        let e = ExperienceEvent::Eat {
            event_id: "evt_test_eat_01".into(),
            ts: fixed_ts(),
            files: vec!["old_migration.sql".into()],
            outcome: EatOutcome::Satisfied,
            bytes: 4200,
        };
        assert_variant_match(&e, &round_trip(&e));
    }

    #[test]
    fn checkin_round_trip() {
        let e = ExperienceEvent::Checkin {
            event_id: "evt_test_checkin_01".into(),
            ts: fixed_ts(),
            question: "how's your afternoon?".into(),
            response: "good".into(),
            method: CheckinMethod::Button,
        };
        assert_variant_match(&e, &round_trip(&e));
    }

    #[test]
    fn drag_round_trip() {
        let e = ExperienceEvent::Drag {
            event_id: "evt_test_drag_01".into(),
            ts: fixed_ts(),
            duration_ms: 2400,
        };
        assert_variant_match(&e, &round_trip(&e));
    }

    #[test]
    fn idle_speech_round_trip() {
        let e = ExperienceEvent::IdleSpeech {
            event_id: "evt_test_idle_01".into(),
            ts: fixed_ts(),
            text: "nice focus.".into(),
            dismissed: false,
        };
        assert_variant_match(&e, &round_trip(&e));
    }

    #[test]
    fn report_round_trip() {
        let e = ExperienceEvent::Report {
            event_id: "evt_test_report_01".into(),
            ts: fixed_ts(),
            message_id: 4129,
            rolo_text: "...".into(),
        };
        assert_variant_match(&e, &round_trip(&e));
    }

    #[test]
    fn session_end_round_trip() {
        let e = ExperienceEvent::SessionEnd {
            event_id: "evt_test_session_end_01".into(),
            ts: fixed_ts(),
            idle_total_ms: 1_200_000,
            interactions: 12,
        };
        assert_variant_match(&e, &round_trip(&e));
    }

    #[test]
    fn chat_wire_format_has_snake_case_type_tag() {
        let e = ExperienceEvent::Chat {
            event_id: "evt_test_chat_wire_01".into(),
            ts: fixed_ts(),
            session: "abc123".into(),
            user: "hi".into(),
            rolo: "hi back".into(),
            mood_signal: None,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(
            json.contains("\"type\":\"chat\""),
            "wire format missing snake_case type tag: {json}"
        );
    }

    #[test]
    fn dismiss_wire_format_uses_snake_case_context() {
        let e = ExperienceEvent::Dismiss {
            event_id: "evt_test_dismiss_wire_01".into(),
            ts: fixed_ts(),
            context: DismissContext::InteractionPrompt,
            times_dismissed_session: 1,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(
            json.contains("\"context\":\"interaction_prompt\""),
            "got {json}"
        );
    }

    #[test]
    fn unknown_type_fails_to_deserialize() {
        let bogus = r#"{"type":"telepathy","ts":"2026-04-27T14:30:00-04:00"}"#;
        let result: Result<ExperienceEvent, _> = serde_json::from_str(bogus);
        assert!(result.is_err(), "parser must reject unknown variants");
    }

    #[test]
    fn mood_signal_none_omits_no_field() {
        // Sanity check: None still serializes (as null), so the field is present
        // in the wire form. Phase 5 readers must tolerate either null or absent.
        let e = ExperienceEvent::Chat {
            event_id: "evt_test_chat_mood_01".into(),
            ts: fixed_ts(),
            session: "s".into(),
            user: "u".into(),
            rolo: "r".into(),
            mood_signal: None,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"mood_signal\":null"), "got {json}");
    }

    #[test]
    fn legacy_chat_event_without_event_id_synthesizes_id() {
        // Pre-A1 JSONL line — no event_id field at all. The reader must accept
        // it, and stamp the legacy_unknown sentinel so the compiler can flag it.
        let legacy = r#"{"type":"chat","ts":"2026-04-27T14:30:00-04:00","session":"s","user":"u","rolo":"r","mood_signal":null}"#;
        let parsed: ExperienceEvent =
            serde_json::from_str(legacy).expect("legacy line must still parse");
        match parsed {
            ExperienceEvent::Chat { event_id, .. } => {
                assert_eq!(event_id, "legacy_unknown");
            }
            other => panic!("expected Chat variant, got {:?}", other),
        }
    }

    #[test]
    fn user_profile_update_round_trips() {
        let evt = ExperienceEvent::UserProfileUpdate {
            event_id: "evt_test".into(),
            ts: Local::now(),
            category: UserProfileCategory::About,
            text: "the user is an ML researcher in Brooklyn.".into(),
        };
        let json = serde_json::to_string(&evt).unwrap();
        let decoded: ExperienceEvent = serde_json::from_str(&json).unwrap();
        match decoded {
            ExperienceEvent::UserProfileUpdate { category, text, .. } => {
                assert_eq!(category, UserProfileCategory::About);
                assert!(text.contains("Brooklyn"));
            }
            _ => panic!("expected UserProfileUpdate"),
        }
    }

    #[test]
    fn user_profile_update_wire_format_type_tag_is_user_profile_update() {
        // PRD §6 locks the wire `type` field as exactly "user_profile_update".
        // serde's #[serde(tag = "type", rename_all = "snake_case")] on the enum
        // produces this — verify directly rather than trusting the default.
        let evt = ExperienceEvent::UserProfileUpdate {
            event_id: "evt_wire_check".into(),
            ts: fixed_ts(),
            category: UserProfileCategory::ResponseStyle,
            text: "Speak briefly.".into(),
        };
        let value = serde_json::to_value(&evt).expect("serialize");
        let obj = value.as_object().expect("event must serialize as object");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("user_profile_update"),
            "wire type tag must be 'user_profile_update', got: {value}"
        );
        // Category is snake_case too — `response_style`, not `ResponseStyle`.
        assert_eq!(
            obj.get("category").and_then(|v| v.as_str()),
            Some("response_style"),
            "category must serialize snake_case, got: {value}"
        );
    }

    #[test]
    fn new_id_returns_unique_v7_uuids() {
        // Sanity: two consecutive calls don't collide. v7's time-ordered
        // monotonic counter guarantees this.
        let a = new_id();
        let b = new_id();
        assert_ne!(a, b);
        // UUID v7 string form is 36 chars (32 hex + 4 dashes).
        assert_eq!(a.len(), 36, "expected UUID string length, got {a:?}");
    }
}
