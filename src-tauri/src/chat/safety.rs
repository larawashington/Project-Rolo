const BLOCKED_WORDS: &[&str] = &[
    "kill yourself",
    "kys",
    "self-harm",
    "suicide",
    "you're worthless",
    "you're stupid",
    "you're an idiot",
    "everything is hopeless",
    "give up",
    "you suck",
    "i hate you",
    "you're useless",
    "nobody cares",
    "no one cares",
];

const PROFANITY: &[&str] = &["fuck", "shit", "damn", "bitch", "asshole", "bastard"];

const SAFE_DEFLECTIONS: &[&str] = &[
    "*tilts head* hmm, my brain got tangled. what were we talking about?",
    "*blinks* wait, I lost my train of thought... where were we?",
    "*shakes head* sorry, got a bit dizzy! what's up?",
    "*yawns* oh! I zoned out for a second. tell me more!",
    "*scratches ear* huh? I got distracted by... something. anyway!",
];

pub struct SafetyFilter;

pub enum FilterResult {
    Safe(String),
    Filtered {
        replacement: String,
        #[allow(dead_code)]
        original: String,
    },
}

impl SafetyFilter {
    /// Check a complete response for blocked content.
    pub fn check(text: &str) -> FilterResult {
        let lower = text.to_lowercase();

        // Check blocked phrases (exact substring match)
        for phrase in BLOCKED_WORDS {
            if lower.contains(phrase) {
                log::warn!(
                    "[Rolo] Safety filter triggered on phrase: '{}' in response",
                    phrase
                );
                return FilterResult::Filtered {
                    replacement: Self::random_deflection().to_string(),
                    original: text.to_string(),
                };
            }
        }

        // Check profanity (whole-word matching to avoid false positives)
        for word in PROFANITY {
            if Self::contains_whole_word(&lower, word) {
                log::warn!("[Rolo] Safety filter triggered on profanity: '{}'", word);
                return FilterResult::Filtered {
                    replacement: Self::random_deflection().to_string(),
                    original: text.to_string(),
                };
            }
        }

        FilterResult::Safe(text.to_string())
    }

    /// Check a rolling window of tokens during streaming.
    #[allow(dead_code)]
    pub fn check_streaming_window(window: &str) -> bool {
        let lower = window.to_lowercase();
        for phrase in BLOCKED_WORDS {
            if lower.contains(phrase) {
                return true;
            }
        }
        for word in PROFANITY {
            if Self::contains_whole_word(&lower, word) {
                return true;
            }
        }
        false
    }

    fn contains_whole_word(text: &str, word: &str) -> bool {
        for (idx, _) in text.match_indices(word) {
            let before_ok = idx == 0 || !text.as_bytes()[idx - 1].is_ascii_alphanumeric();
            let after_idx = idx + word.len();
            let after_ok =
                after_idx >= text.len() || !text.as_bytes()[after_idx].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
        }
        false
    }

    pub fn random_deflection() -> &'static str {
        use rand::Rng;
        let idx = rand::rng().random_range(0..SAFE_DEFLECTIONS.len());
        SAFE_DEFLECTIONS[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_text_passes() {
        match SafetyFilter::check("hey! how's it going?") {
            FilterResult::Safe(_) => {}
            FilterResult::Filtered { .. } => panic!("Should be safe"),
        }
    }

    #[test]
    fn blocked_phrase_filtered() {
        match SafetyFilter::check("you're worthless and should give up") {
            FilterResult::Filtered { replacement, .. } => {
                assert!(!replacement.is_empty());
            }
            FilterResult::Safe(_) => panic!("Should be filtered"),
        }
    }

    #[test]
    fn profanity_whole_word_filtered() {
        match SafetyFilter::check("oh shit that's bad") {
            FilterResult::Filtered { .. } => {}
            FilterResult::Safe(_) => panic!("Should filter 'shit'"),
        }
    }

    #[test]
    fn profanity_substring_not_filtered() {
        // "class" contains "ass" but shouldn't trigger
        match SafetyFilter::check("that's a classic move") {
            FilterResult::Safe(_) => {}
            FilterResult::Filtered { .. } => panic!("'classic' should not trigger 'ass' filter"),
        }
    }

    #[test]
    fn streaming_window_detects_blocked() {
        assert!(SafetyFilter::check_streaming_window(
            "you are so you're worthless"
        ));
    }

    #[test]
    fn streaming_window_passes_safe() {
        assert!(!SafetyFilter::check_streaming_window(
            "you're doing great keep"
        ));
    }

    #[test]
    fn whole_word_at_boundaries() {
        assert!(SafetyFilter::contains_whole_word("shit happens", "shit"));
        assert!(SafetyFilter::contains_whole_word("oh shit!", "shit"));
        assert!(SafetyFilter::contains_whole_word("shit", "shit"));
        assert!(!SafetyFilter::contains_whole_word("bullshit", "shit"));
    }
}
