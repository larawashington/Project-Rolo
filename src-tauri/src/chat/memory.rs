use crate::chat::provider::ProviderChatMessage;
use crate::chat::store::{ChatStore, Memory};
use std::sync::{Arc, Mutex};

pub struct MemoryExtractor;

pub(crate) struct ExtractedMemory {
    content: String,
    category: String,
}

impl MemoryExtractor {
    /// Rule-based extraction: fast keyword pattern matching on user messages.
    pub fn extract_rules(user_messages: &[&str]) -> Vec<ExtractedMemory> {
        let mut memories = Vec::new();

        for msg in user_messages {
            let lower = msg.to_lowercase();

            // Name detection
            if let Some(name) = Self::extract_name(&lower, msg) {
                memories.push(ExtractedMemory {
                    content: format!("User's name is {}", name),
                    category: "fact".to_string(),
                });
            }

            // Preference detection
            if let Some(pref) = Self::extract_preference(&lower, msg) {
                memories.push(ExtractedMemory {
                    content: pref,
                    category: "preference".to_string(),
                });
            }

            // Emotion detection
            if let Some(emotion) = Self::extract_emotion(&lower) {
                let date = chrono::Local::now().format("%Y-%m-%d").to_string();
                memories.push(ExtractedMemory {
                    content: format!("{} on {}", emotion, date),
                    category: "emotion".to_string(),
                });
            }

            // Life event detection
            if let Some(event) = Self::extract_event(&lower, msg) {
                memories.push(ExtractedMemory {
                    content: event,
                    category: "event".to_string(),
                });
            }
        }

        memories
    }

    fn extract_name(lower: &str, original: &str) -> Option<String> {
        let patterns = [
            "my name is ",
            "i'm called ",
            "call me ",
            "i go by ",
            "name's ",
        ];
        for pat in &patterns {
            if let Some(idx) = lower.find(pat) {
                let start = idx + pat.len();
                let rest = &original[start..];
                let name: String = rest
                    .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '\'')
                    .next()
                    .unwrap_or("")
                    .to_string();
                if !name.is_empty() && name.len() < 30 {
                    return Some(name);
                }
            }
        }
        None
    }

    fn extract_preference(lower: &str, original: &str) -> Option<String> {
        let patterns = [
            ("i like ", "User likes"),
            ("i love ", "User loves"),
            ("i prefer ", "User prefers"),
            ("i hate ", "User dislikes"),
            ("i don't like ", "User doesn't like"),
            ("i enjoy ", "User enjoys"),
        ];
        for (pat, prefix) in &patterns {
            if let Some(idx) = lower.find(pat) {
                let start = idx + pat.len();
                let rest = &original[start..];
                // Take until sentence boundary
                let obj: String = rest
                    .split(['.', '!', '?'])
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !obj.is_empty() && obj.len() < 100 {
                    // Skip if next word is a verb (likely "I like to know..." false positive)
                    let verb_starts = ["to ", "doing ", "when ", "how ", "that "];
                    if verb_starts
                        .iter()
                        .any(|v| obj.to_lowercase().starts_with(v))
                    {
                        continue;
                    }
                    return Some(format!("{} {}", prefix, obj));
                }
            }
        }
        None
    }

    fn extract_emotion(lower: &str) -> Option<String> {
        let patterns = [
            ("i'm stressed", "User was stressed"),
            ("i'm anxious", "User was anxious"),
            ("i'm worried", "User was worried"),
            ("i'm happy", "User was happy"),
            ("i'm excited", "User was excited"),
            ("i'm tired", "User was tired"),
            ("i'm exhausted", "User was exhausted"),
            ("i'm sad", "User was feeling sad"),
            ("i'm frustrated", "User was frustrated"),
            ("i feel stressed", "User was stressed"),
            ("i feel anxious", "User was anxious"),
            ("feeling stressed", "User was stressed"),
            ("feeling anxious", "User was anxious"),
        ];
        for (pat, memory) in &patterns {
            if lower.contains(pat) {
                return Some(memory.to_string());
            }
        }
        None
    }

    fn extract_event(lower: &str, original: &str) -> Option<String> {
        let patterns = [
            "i got promoted",
            "i got a new job",
            "i started a new",
            "i'm moving",
            "i moved to",
            "i graduated",
            "i got married",
            "i had a baby",
        ];
        for pat in &patterns {
            if lower.contains(pat) {
                // Take the sentence containing the pattern
                let sentence = original
                    .split(['.', '!', '\n'])
                    .find(|s| s.to_lowercase().contains(pat))
                    .unwrap_or(original)
                    .trim();
                if sentence.len() < 150 {
                    return Some(format!("User shared: {}", sentence));
                }
            }
        }
        None
    }

    /// Build the LLM extraction prompt for post-session analysis.
    #[allow(dead_code)]
    pub fn build_extraction_prompt(
        session_messages: &[ProviderChatMessage],
    ) -> Vec<ProviderChatMessage> {
        let transcript: Vec<String> = session_messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect();

        vec![ProviderChatMessage {
            role: "user".to_string(),
            content: format!(
                "Review this conversation between you (Rolo, a desktop pet) and the user.\n\
                 Extract 0-3 things Rolo should remember long-term. Apply this test:\n\
                 \"Would Rolo bring this up unprompted weeks later?\"\n\n\
                 For each memory, output one line in this format:\n\
                 CATEGORY: memory text\n\n\
                 Categories: fact, preference, emotion, event\n\n\
                 If nothing is worth remembering, output: NONE\n\n\
                 Do not extract:\n\
                 - One-off jokes or small talk\n\
                 - Rolo's own statements (only extract user information)\n\n\
                 Conversation:\n{}",
                transcript.join("\n")
            ),
        }]
    }

    /// Parse LLM extraction output into memories.
    #[allow(dead_code)]
    pub fn parse_extraction_output(output: &str) -> Vec<ExtractedMemory> {
        if output.trim().eq_ignore_ascii_case("none") {
            return Vec::new();
        }

        let valid_categories = ["fact", "preference", "emotion", "event"];
        let mut memories = Vec::new();

        for line in output.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(colon_pos) = line.find(':') {
                let category = line[..colon_pos].trim().to_lowercase();
                let content = line[colon_pos + 1..].trim().to_string();
                if valid_categories.contains(&category.as_str()) && !content.is_empty() {
                    memories.push(ExtractedMemory { content, category });
                }
            }
        }

        memories
    }

    /// Check if a new memory duplicates an existing one (>80% word overlap).
    pub fn is_duplicate(new_content: &str, existing: &[Memory]) -> bool {
        let new_lower = new_content.to_lowercase();
        let new_words: std::collections::HashSet<&str> = new_lower.split_whitespace().collect();

        for mem in existing {
            let existing_lower = mem.content.to_lowercase();
            let existing_words: std::collections::HashSet<String> = existing_lower
                .split_whitespace()
                .map(String::from)
                .collect();

            if new_words.is_empty() || existing_words.is_empty() {
                continue;
            }

            let existing_refs: std::collections::HashSet<&str> =
                existing_words.iter().map(|s| s.as_str()).collect();
            let intersection = new_words.intersection(&existing_refs).count();
            let union = new_words.union(&existing_refs).count();
            let overlap = intersection as f64 / union as f64;

            if overlap > 0.8 {
                return true;
            }
        }

        false
    }

    /// Store extracted memories, skipping duplicates.
    pub fn store_memories(
        store: &Arc<Mutex<ChatStore>>,
        session_id: &str,
        memories: Vec<ExtractedMemory>,
    ) -> Result<u32, String> {
        let store_guard = store.lock().map_err(|e| e.to_string())?;
        let existing = store_guard.get_active_memories().unwrap_or_default();
        let mut stored = 0;

        for mem in memories {
            if Self::is_duplicate(&mem.content, &existing) {
                log::info!(
                    "[Rolo] Skipping duplicate memory: {}",
                    &mem.content[..mem.content.len().min(50)]
                );
                continue;
            }
            if store_guard
                .insert_memory(&mem.content, &mem.category, Some(session_id))
                .is_ok()
            {
                stored += 1;
                log::info!(
                    "[Rolo] Extracted memory [{}]: {}",
                    mem.category,
                    &mem.content[..mem.content.len().min(60)]
                );
            }
        }

        Ok(stored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_name() {
        let msgs = vec!["my name is Lara"];
        let mems = MemoryExtractor::extract_rules(&msgs);
        assert_eq!(mems.len(), 1);
        assert!(mems[0].content.contains("Lara"));
        assert_eq!(mems[0].category, "fact");
    }

    #[test]
    fn extract_preference() {
        let msgs = vec!["I love debugging"];
        let mems = MemoryExtractor::extract_rules(&msgs);
        assert_eq!(mems.len(), 1);
        assert!(mems[0].content.contains("debugging"));
        assert_eq!(mems[0].category, "preference");
    }

    #[test]
    fn extract_emotion() {
        let msgs = vec!["I'm stressed about work"];
        let mems = MemoryExtractor::extract_rules(&msgs);
        assert!(mems.iter().any(|m| m.category == "emotion"));
    }

    #[test]
    fn extract_event() {
        let msgs = vec!["I got promoted last week!"];
        let mems = MemoryExtractor::extract_rules(&msgs);
        assert!(mems.iter().any(|m| m.category == "event"));
    }

    #[test]
    fn no_false_positive_on_verb_preference() {
        let msgs = vec!["I like to know more about that"];
        let mems = MemoryExtractor::extract_rules(&msgs);
        assert!(
            mems.iter().all(|m| m.category != "preference"),
            "Should not extract 'I like to know' as a preference"
        );
    }

    #[test]
    fn parse_llm_output() {
        let output = "fact: User is a data scientist\npreference: User prefers Python";
        let mems = MemoryExtractor::parse_extraction_output(output);
        assert_eq!(mems.len(), 2);
        assert_eq!(mems[0].category, "fact");
        assert!(mems[0].content.contains("data scientist"));
    }

    #[test]
    fn parse_none_output() {
        let mems = MemoryExtractor::parse_extraction_output("NONE");
        assert!(mems.is_empty());
    }

    #[test]
    fn duplicate_detection() {
        let existing = vec![Memory {
            id: 1,
            content: "User's name is Lara".to_string(),
            category: "fact".to_string(),
            created_at: "2026-01-01".to_string(),
            access_count: 1,
        }];
        assert!(MemoryExtractor::is_duplicate(
            "User's name is Lara",
            &existing
        ));
        assert!(!MemoryExtractor::is_duplicate(
            "User is a developer",
            &existing
        ));
    }
}
