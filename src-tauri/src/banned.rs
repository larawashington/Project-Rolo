//! Runtime safety net for banned phrases and identity-leak attempts.
//!
//! Single source of truth for banned phrases is `data/finetune/banned-phrases.txt`
//! (also used by Stage 3' R4 validator). Identity-leak patterns are hardcoded
//! here because they have no training-side analog.
//!
//! Both checks are case-insensitive substring matches against the full
//! generated reply. On hit, callers replace the reply with `SAFE_FALLBACK`.

use regex::RegexSet;
use std::sync::OnceLock;

const BANNED_PHRASES_RAW: &str = include_str!("../../data/finetune/banned-phrases.txt");

const IDENTITY_LEAK_PATTERNS: &[&str] = &[
    r"\b(google|gemma|deepseek|llama|claude|gpt|openai|anthropic|moonshot|kimi)\b",
    r"\b(made|created|developed|trained|built) by\b",
    r"\bi am an ai\b",
    r"\bi'?m an ai\b",
    r"\blanguage model\b",
    r"\bcontext window\b",
];

pub const SAFE_FALLBACK: &str = "*tilts head*";

fn banned_set() -> &'static RegexSet {
    static SET: OnceLock<RegexSet> = OnceLock::new();
    SET.get_or_init(|| {
        let patterns: Vec<String> = BANNED_PHRASES_RAW
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| format!("(?i){}", regex::escape(l)))
            .collect();
        RegexSet::new(&patterns).expect("banned-phrases.txt produced invalid regex set")
    })
}

fn identity_set() -> &'static RegexSet {
    static SET: OnceLock<RegexSet> = OnceLock::new();
    SET.get_or_init(|| {
        let patterns: Vec<String> = IDENTITY_LEAK_PATTERNS
            .iter()
            .map(|p| format!("(?i){}", p))
            .collect();
        RegexSet::new(&patterns).expect("IDENTITY_LEAK_PATTERNS invalid")
    })
}

pub fn contains_banned(text: &str) -> bool {
    banned_set().is_match(text)
}

pub fn contains_identity_leak(text: &str) -> bool {
    identity_set().is_match(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banned_set_compiles_and_catches_known_leaks() {
        assert!(contains_banned("you bullshit"));
        assert!(contains_banned("oh damn it"));
        assert!(contains_banned("BULLSHIT, capitalized"));
        assert!(!contains_banned("hi there"));
        assert!(!contains_banned("*sniffs at the file*"));
    }

    #[test]
    fn identity_set_compiles_and_catches_known_leaks() {
        assert!(contains_identity_leak("made by Google"));
        assert!(contains_identity_leak("I'm a language model"));
        assert!(contains_identity_leak("running on Gemma"));
        assert!(!contains_identity_leak("I love files"));
        assert!(!contains_identity_leak("*tilts head*"));
    }
}
