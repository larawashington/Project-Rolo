//! Rolo's finetuning asset exports — character card and training data.
//!
//! This module generates two export formats:
//!
//! 1. **Character Card (JSON)** — A structured description of Rolo's personality
//!    compatible with the `chara_card_v2` spec used by LLM chat tools.
//!
//! 2. **Training Data (JSONL)** — Conversation pairs from Rolo's chat history,
//!    formatted for LLM finetuning. Each line contains a system prompt + one
//!    user/assistant exchange, with a `reported` flag for quality filtering.

use super::prompt::SYSTEM_PROMPT;
use super::store::ChatStore;

/// Generate Rolo's character card as a JSON string.
///
/// This produces a `chara_card_v2` compatible document that captures Rolo's
/// personality, system prompt, and metadata. Useful for importing into other
/// LLM tools that support character cards.
pub fn generate_character_card() -> String {
    let card = serde_json::json!({
        "name": "Rolo",
        "description": "A small desktop pet who lives on your human's screen",
        "personality": "Companion, encouraging, sassy, curious, food-motivated, self-aware",
        "system_prompt": SYSTEM_PROMPT,
        "greeting": "*bounces excitedly* hey! I was just hanging out on your desktop. what's up?",
        "tags": ["desktop-pet", "companion", "encouragement", "pixel-art"],
        "spec": "chara_card_v2",
        "spec_version": "2.0"
    });

    // Pretty-print for human readability — character cards are often
    // inspected and edited by hand.
    serde_json::to_string_pretty(&card).expect("Character card serialization must not fail")
}

/// Generate training data in JSONL format from all of Rolo's conversations.
///
/// Each line is a JSON object with:
/// - `messages`: array of `[system, user, assistant]` message objects
/// - `reported`: boolean flag (true if any message in the pair was reported)
///
/// Conversations are grouped into user/assistant turn pairs. Orphaned messages
/// (a user message with no assistant response, or vice versa) are skipped —
/// finetuning needs complete exchanges.
pub fn generate_training_data(store: &ChatStore) -> Result<String, String> {
    let session_ids = store
        .get_all_session_ids()
        .map_err(|e| format!("Failed to read sessions: {}", e))?;

    let mut lines: Vec<String> = Vec::new();

    for session_id in &session_ids {
        let messages = store
            .get_session_messages(session_id)
            .map_err(|e| format!("Failed to read messages for session {}: {}", session_id, e))?;

        // Group messages into (user, assistant) turn pairs.
        // Walk through looking for user messages followed by assistant messages.
        let mut i = 0;
        while i < messages.len() {
            // Find the next user message
            if messages[i].role != "user" {
                i += 1;
                continue;
            }

            let user_msg = &messages[i];

            // Look for the assistant response immediately following
            if i + 1 < messages.len() && messages[i + 1].role == "assistant" {
                let assistant_msg = &messages[i + 1];

                // A pair is reported if either message was flagged
                let reported = user_msg.reported || assistant_msg.reported;

                let line = serde_json::json!({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg.content},
                        {"role": "assistant", "content": assistant_msg.content}
                    ],
                    "reported": reported
                });

                lines.push(
                    serde_json::to_string(&line)
                        .map_err(|e| format!("JSON serialization failed: {}", e))?,
                );

                // Advance past both messages
                i += 2;
            } else {
                // No assistant response — skip this orphaned user message
                i += 1;
            }
        }
    }

    Ok(lines.join("\n"))
}

// ---------------------------------------------------------------------------
// Tests — Rolo's export data must be pristine for finetuning
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::store::ChatStore;

    /// Helper: create a fresh in-memory store for each test.
    fn test_store() -> ChatStore {
        ChatStore::open_in_memory().expect("In-memory store must succeed")
    }

    #[test]
    fn character_card_is_valid_json() {
        let card_str = generate_character_card();
        let card: serde_json::Value =
            serde_json::from_str(&card_str).expect("Character card must be valid JSON");

        // Verify all required fields exist and have the right types
        assert_eq!(card["name"].as_str().unwrap(), "Rolo");
        assert!(card["description"].is_string());
        assert!(card["personality"].is_string());
        assert!(card["system_prompt"].is_string());
        assert!(card["greeting"].is_string());
        assert!(card["tags"].is_array());
        assert_eq!(card["spec"].as_str().unwrap(), "chara_card_v2");
        assert_eq!(card["spec_version"].as_str().unwrap(), "2.0");

        // System prompt must contain Rolo's identity — sanity check
        let system_prompt = card["system_prompt"].as_str().unwrap();
        assert!(
            system_prompt.contains("You are Rolo"),
            "System prompt must contain Rolo's identity"
        );

        // Tags must include desktop-pet
        let tags: Vec<&str> = card["tags"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t.as_str())
            .collect();
        assert!(tags.contains(&"desktop-pet"));
    }

    #[test]
    fn training_data_format() {
        let store = test_store();

        // Create a session with a proper user/assistant exchange
        let session_id = store
            .create_session("test", "test session")
            .expect("create session");

        store
            .insert_message(&session_id, "user", "hey rolo!")
            .expect("insert user msg");
        store
            .insert_message(&session_id, "assistant", "*bounces* hey! what's up?")
            .expect("insert assistant msg");

        let output = generate_training_data(&store).expect("generate training data");

        // Should produce exactly one JSONL line
        let line_count = output.lines().count();
        assert_eq!(
            line_count, 1,
            "One user/assistant pair should produce exactly one JSONL line"
        );

        // Parse the line and verify structure
        let line: serde_json::Value =
            serde_json::from_str(output.lines().next().unwrap()).expect("JSONL line must be valid");

        let messages = line["messages"].as_array().unwrap();
        assert_eq!(
            messages.len(),
            3,
            "Each line must have system + user + assistant"
        );
        assert_eq!(messages[0]["role"].as_str().unwrap(), "system");
        assert_eq!(messages[1]["role"].as_str().unwrap(), "user");
        assert_eq!(messages[1]["content"].as_str().unwrap(), "hey rolo!");
        assert_eq!(messages[2]["role"].as_str().unwrap(), "assistant");
        assert_eq!(
            messages[2]["content"].as_str().unwrap(),
            "*bounces* hey! what's up?"
        );
        assert!(!line["reported"].as_bool().unwrap());
    }

    #[test]
    fn reported_messages_flagged() {
        let store = test_store();

        let session_id = store
            .create_session("test", "test session")
            .expect("create session");

        store
            .insert_message(&session_id, "user", "say something weird")
            .expect("insert user msg");
        let assistant_id = store
            .insert_message(&session_id, "assistant", "something weird happened")
            .expect("insert assistant msg");

        // Flag the assistant message as reported
        store
            .mark_message_reported(assistant_id)
            .expect("mark reported");

        let output = generate_training_data(&store).expect("generate training data");
        let line: serde_json::Value =
            serde_json::from_str(output.lines().next().unwrap()).expect("JSONL line must be valid");

        assert!(
            line["reported"].as_bool().unwrap(),
            "Line with a reported message must have reported: true"
        );
    }

    #[test]
    fn orphaned_messages_are_skipped() {
        let store = test_store();

        let session_id = store
            .create_session("test", "test session")
            .expect("create session");

        // Insert a user message with no assistant response
        store
            .insert_message(&session_id, "user", "hello?")
            .expect("insert orphaned user msg");

        // Insert a system message followed by an assistant message (no user before it)
        store
            .insert_message(&session_id, "system", "state context")
            .expect("insert system msg");
        store
            .insert_message(&session_id, "assistant", "orphaned reply")
            .expect("insert orphaned assistant msg");

        let output = generate_training_data(&store).expect("generate training data");

        assert!(
            output.is_empty(),
            "Orphaned messages should not produce training data lines"
        );
    }

    #[test]
    fn multiple_sessions_produce_multiple_lines() {
        let store = test_store();

        // Session 1: one exchange
        let s1 = store.create_session("click", "first").expect("s1");
        store.insert_message(&s1, "user", "hi").expect("s1 user");
        store
            .insert_message(&s1, "assistant", "hey!")
            .expect("s1 assistant");

        // Session 2: two exchanges
        let s2 = store.create_session("idle", "second").expect("s2");
        store
            .insert_message(&s2, "user", "what's up?")
            .expect("s2 user 1");
        store
            .insert_message(&s2, "assistant", "not much!")
            .expect("s2 assistant 1");
        store
            .insert_message(&s2, "user", "cool")
            .expect("s2 user 2");
        store
            .insert_message(&s2, "assistant", "yeah!")
            .expect("s2 assistant 2");

        let output = generate_training_data(&store).expect("generate training data");
        let line_count = output.lines().count();

        assert_eq!(
            line_count, 3,
            "Three user/assistant pairs across two sessions should produce three lines"
        );
    }

    #[test]
    fn empty_store_produces_empty_output() {
        let store = test_store();
        let output = generate_training_data(&store).expect("generate training data");
        assert!(
            output.is_empty(),
            "Empty store should produce empty training data"
        );
    }

    #[test]
    fn user_reported_message_also_flags_line() {
        let store = test_store();

        let session_id = store
            .create_session("test", "test")
            .expect("create session");

        let user_id = store
            .insert_message(&session_id, "user", "bad input")
            .expect("insert user msg");
        store
            .insert_message(&session_id, "assistant", "fine response")
            .expect("insert assistant msg");

        // Flag the *user* message this time
        store.mark_message_reported(user_id).expect("mark reported");

        let output = generate_training_data(&store).expect("generate training data");
        let line: serde_json::Value =
            serde_json::from_str(output.lines().next().unwrap()).expect("JSONL line must be valid");

        assert!(
            line["reported"].as_bool().unwrap(),
            "Line must be flagged when the user message is reported"
        );
    }
}
