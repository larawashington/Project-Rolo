use crate::chat::provider::ProviderChatMessage;

const RESPONSE_RESERVE: u32 = 200;

pub struct ContextBudget {
    context_window: u32,
    summarization_count: u32,
}

impl ContextBudget {
    pub fn new(context_window: u32) -> Self {
        Self {
            context_window,
            summarization_count: 0,
        }
    }

    pub fn estimate_tokens(text: &str) -> u32 {
        crate::tokens::estimate_tokens(text) as u32
    }

    pub fn total_tokens(messages: &[ProviderChatMessage]) -> u32 {
        messages
            .iter()
            .map(|m| Self::estimate_tokens(&m.content) + Self::estimate_tokens(&m.role) + 4)
            .sum()
    }

    pub fn needs_summarization(&self, messages: &[ProviderChatMessage]) -> bool {
        if self.summarization_count >= 2 {
            return false; // Cap at 2 rounds
        }
        let total = Self::total_tokens(messages);
        let threshold = self.context_window.saturating_sub(RESPONSE_RESERVE) * 70 / 100;
        total > threshold
    }

    #[allow(dead_code)]
    pub fn should_hard_cutoff(&self) -> bool {
        self.summarization_count >= 2
    }

    pub fn split_for_summarization(
        messages: &[ProviderChatMessage],
    ) -> (Vec<ProviderChatMessage>, Vec<ProviderChatMessage>) {
        // Keep system messages and last 4 exchanges (8 messages)
        let keep_recent = 8;
        let system_count = messages.iter().take_while(|m| m.role == "system").count();

        let conversation = &messages[system_count..];
        if conversation.len() <= keep_recent {
            return (Vec::new(), messages.to_vec());
        }

        let split_point = conversation.len() - keep_recent;
        let to_summarize = conversation[..split_point].to_vec();
        let to_keep: Vec<ProviderChatMessage> = messages[..system_count]
            .iter()
            .chain(conversation[split_point..].iter())
            .cloned()
            .collect();

        (to_summarize, to_keep)
    }

    #[allow(dead_code)]
    pub fn build_summary_prompt(old_messages: &[ProviderChatMessage]) -> Vec<ProviderChatMessage> {
        let transcript: Vec<String> = old_messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect();

        vec![ProviderChatMessage {
            role: "user".to_string(),
            content: format!(
                "Summarize this conversation between you (Rolo) and the user in 2-3 sentences. \
                 Focus on topics discussed, emotional tone, and anything important the user shared.\n\n{}",
                transcript.join("\n")
            ),
        }]
    }

    #[allow(dead_code)]
    pub fn inject_summary(summary: &str, messages: &mut Vec<ProviderChatMessage>) {
        let insert_pos = messages.iter().take_while(|m| m.role == "system").count();

        messages.insert(
            insert_pos,
            ProviderChatMessage {
                role: "system".to_string(),
                content: format!("[Earlier in this conversation: {}]", summary),
            },
        );
    }

    #[allow(dead_code)]
    pub fn record_summarization(&mut self) {
        self.summarization_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ProviderChatMessage {
        ProviderChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn estimate_tokens_basic() {
        // Shared estimator is `chars().count() / 4` (floor).
        assert_eq!(ContextBudget::estimate_tokens("hello"), 1); // 5/4 = 1
        assert_eq!(ContextBudget::estimate_tokens(""), 0);
        assert_eq!(ContextBudget::estimate_tokens("abcd"), 1); // exactly 4
    }

    #[test]
    fn needs_summarization_below_threshold() {
        let budget = ContextBudget::new(4096);
        let messages = vec![msg("user", "hi"), msg("assistant", "hey!")];
        assert!(!budget.needs_summarization(&messages));
    }

    #[test]
    fn needs_summarization_above_threshold() {
        let budget = ContextBudget::new(100);
        let long = "x".repeat(400); // 100 tokens, way over 70% of 100
        let messages = vec![msg("user", &long)];
        assert!(budget.needs_summarization(&messages));
    }

    #[test]
    fn split_preserves_system_messages() {
        let messages = vec![
            msg("system", "system prompt"),
            msg("user", "msg1"),
            msg("assistant", "reply1"),
            msg("user", "msg2"),
            msg("assistant", "reply2"),
            msg("user", "msg3"),
            msg("assistant", "reply3"),
            msg("user", "msg4"),
            msg("assistant", "reply4"),
            msg("user", "msg5"),
            msg("assistant", "reply5"),
        ];
        let (to_summarize, kept) = ContextBudget::split_for_summarization(&messages);

        assert!(!to_summarize.is_empty());
        assert_eq!(kept[0].role, "system");
        assert_eq!(kept[0].content, "system prompt");
        // Last 8 conversation messages should be kept
        assert!(kept.len() >= 8);
    }

    #[test]
    fn split_no_summary_when_short() {
        let messages = vec![
            msg("system", "prompt"),
            msg("user", "hi"),
            msg("assistant", "hey"),
        ];
        let (to_summarize, _kept) = ContextBudget::split_for_summarization(&messages);
        assert!(to_summarize.is_empty());
    }

    #[test]
    fn hard_cutoff_after_two_rounds() {
        let mut budget = ContextBudget::new(4096);
        assert!(!budget.should_hard_cutoff());
        budget.record_summarization();
        assert!(!budget.should_hard_cutoff());
        budget.record_summarization();
        assert!(budget.should_hard_cutoff());
    }

    #[test]
    fn inject_summary_after_system_messages() {
        let mut messages = vec![
            msg("system", "prompt"),
            msg("system", "state"),
            msg("user", "hi"),
        ];
        ContextBudget::inject_summary("We talked about coding.", &mut messages);
        assert_eq!(messages[2].role, "system");
        assert!(messages[2].content.contains("We talked about coding"));
        assert_eq!(messages[3].role, "user");
    }
}
