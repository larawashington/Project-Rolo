//! Shared token-budget helpers. One chars/4 estimator, one truncator, one
//! pop-oldest-to-fit. Used by `vault::prompt` for slot composition and by
//! `chat::context` for conversation-window summarization checks.
//!
//! The estimator is `text.chars().count() / 4` — floor over Unicode code
//! points. Cheap, deterministic, no allocations; wrong by at most a few
//! tokens on short strings, which is irrelevant against a 4 K+ window.

/// Approximate token count: `chars().count() / 4`.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count() / 4
}

/// Truncate `body` from its tail to fit within `max_tokens` (chars/4).
pub fn truncate_to_budget(body: &str, max_tokens: usize) -> String {
    if estimate_tokens(body) <= max_tokens {
        return body.to_string();
    }
    let max_chars = max_tokens * 4;
    body.char_indices()
        .nth(max_chars)
        .map(|(byte_idx, _)| body[..byte_idx].to_string())
        .unwrap_or_else(|| body.to_string())
}

/// Pop entries off the tail of `lines` until `header + sep.join(lines)` fits
/// within `max_tokens`. Returns `None` if nothing survives.
pub fn pop_oldest_to_fit<I>(lines: I, header: &str, sep: &str, max_tokens: usize) -> Option<String>
where
    I: IntoIterator<Item = String>,
{
    let mut lines: Vec<String> = lines.into_iter().collect();
    let header_chars = header.chars().count();
    let sep_chars = sep.chars().count();
    let mut total_chars = header_chars
        + lines.iter().map(|l| l.chars().count()).sum::<usize>()
        + sep_chars * lines.len().saturating_sub(1);

    while total_chars / 4 > max_tokens {
        match lines.pop() {
            Some(dropped) => {
                total_chars = total_chars
                    .saturating_sub(dropped.chars().count())
                    .saturating_sub(if lines.is_empty() { 0 } else { sep_chars });
            }
            None => return None,
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(format!("{header}{}", lines.join(sep)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_floor() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 0); // 3/4 = 0
        assert_eq!(estimate_tokens("abcd"), 1); // exactly 4
        assert_eq!(estimate_tokens("hello"), 1); // 5/4 = 1
        assert_eq!(estimate_tokens(&"x".repeat(400)), 100);
    }

    #[test]
    fn estimate_tokens_counts_chars_not_bytes() {
        // "ñ" is 2 bytes but 1 char. Two of them = 2 chars / 4 = 0 tokens.
        assert_eq!(estimate_tokens("ññ"), 0);
        // 8 multi-byte chars = 8 chars / 4 = 2 tokens.
        assert_eq!(estimate_tokens("ññññññññ"), 2);
    }

    #[test]
    fn truncate_under_budget_returns_unchanged() {
        let s = "abcdefgh"; // 2 tokens
        assert_eq!(truncate_to_budget(s, 5), s);
    }

    #[test]
    fn truncate_over_budget_cuts_to_max_chars() {
        let s = "abcdefghij"; // 10 chars, 2 tokens
                              // max=1 token -> 4 chars
        assert_eq!(truncate_to_budget(s, 1), "abcd");
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        // 8 multi-byte chars (16 bytes); max=1 token -> first 4 chars.
        let s = "ññññññññ";
        let out = truncate_to_budget(s, 1);
        assert_eq!(out.chars().count(), 4);
        assert!(s.starts_with(&out));
    }

    #[test]
    fn pop_oldest_drops_tail_lines_until_fit() {
        let lines = vec!["one".to_string(), "two".to_string(), "three".to_string()];
        // header=8, lines total = 3+3+5 = 11, sep "\n"=1 * 2 = 2 -> total 21 chars / 4 = 5 tokens.
        // max 5 -> fits, returns full list.
        let out = pop_oldest_to_fit(lines.clone(), "header:\n", "\n", 5).unwrap();
        assert!(out.contains("three"));
        // max 4 tokens -> must drop "three", total = 8 + 3+3 + 1 = 15 chars / 4 = 3 -> fits.
        let out = pop_oldest_to_fit(lines.clone(), "header:\n", "\n", 4).unwrap();
        assert!(!out.contains("three"));
        assert!(out.contains("two"));
    }

    #[test]
    fn pop_oldest_returns_none_when_nothing_fits() {
        let lines = vec!["aaaaaaaaaaaa".to_string()]; // 12 chars
                                                      // header empty, total=12, /4=3 tokens. max=0 -> drop everything -> None.
        assert!(pop_oldest_to_fit(lines, "", "\n", 0).is_none());
    }

    #[test]
    fn pop_oldest_preserves_format_with_separator() {
        let lines = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let out = pop_oldest_to_fit(lines, "head\n", " | ", 100).unwrap();
        assert_eq!(out, "head\nA | B | C");
    }
}
