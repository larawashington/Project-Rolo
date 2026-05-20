//! Shared `reqwest::Client` factories.
//!
//! Every subsystem that talks HTTP (Command Center diagnostics, weather tools,
//! chat providers) used to build its own `reqwest::Client::builder()` chain.
//! That meant five+ near-identical fragments, each with its own timeout and
//! its own `.unwrap_or_else` / `.expect` fallback policy.
//!
//! Now each caller picks the right factory:
//!   * [`http_client`] — short-lived probe/tool client with a single overall
//!     timeout. Used for diagnostics probes, the weather endpoint test, and
//!     the runtime `GetWeather` tool.
//!   * [`chat_client`] — long-running LLM client: 60s overall timeout with a
//!     10s connect timeout. Used by every chat provider.
//!
//! Callers that want a non-fatal fallback on builder failure can pair this
//! with `unwrap_or_else(|_| reqwest::Client::new())` — see the chat providers
//! for the existing pattern.

use std::time::Duration;

use reqwest::Client;

/// Build a probe/tool HTTP client with `timeout` as the overall wall budget.
///
/// Used wherever we need a short, snappy HTTP call (diagnostics probes,
/// weather fetches). The single `.timeout()` becomes the entire request's
/// budget — if you need separate connect / read budgets, write a one-off
/// builder instead.
pub fn http_client(timeout: Duration) -> reqwest::Result<Client> {
    // TLS verification enforced by reqwest defaults — do not disable.
    // All HTTPS callsites in Rolo depend on cert validation; opt-outs must
    // go in a separately-named factory with a written rationale.
    Client::builder().timeout(timeout).build()
}

/// Build a chat-provider HTTP client: 60s overall, 10s to connect.
///
/// Chat completions can stream for tens of seconds even on the happy path,
/// so the overall budget is generous; the connect timeout stays tight so a
/// dead host fails fast rather than hanging the user for a minute.
pub fn chat_client() -> reqwest::Result<Client> {
    // TLS verification enforced by reqwest defaults — do not disable.
    // All HTTPS callsites in Rolo depend on cert validation; opt-outs must
    // go in a separately-named factory with a written rationale.
    Client::builder()
        .timeout(CHAT_CLIENT_OVERALL_TIMEOUT)
        .connect_timeout(CHAT_CLIENT_CONNECT_TIMEOUT)
        .build()
}

/// Exposed so tests (and any future callsite that needs to reason about the
/// chat-provider budget) can refer to the same constant the factory uses.
pub const CHAT_CLIENT_OVERALL_TIMEOUT: Duration = Duration::from_secs(60);

/// Connect-phase budget for `chat_client`. Tight so a dead host fails fast.
pub const CHAT_CLIENT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_client_constants_are_pinned() {
        // The factory's policy is "60s overall, 10s connect." Constant drift
        // would silently affect every chat provider — this test is the
        // canary.
        assert_eq!(CHAT_CLIENT_OVERALL_TIMEOUT, Duration::from_secs(60));
        assert_eq!(CHAT_CLIENT_CONNECT_TIMEOUT, Duration::from_secs(10));
    }

    #[test]
    fn chat_client_builds_successfully() {
        assert!(chat_client().is_ok());
    }

    #[test]
    fn http_client_builds_with_any_positive_timeout() {
        assert!(http_client(Duration::from_secs(5)).is_ok());
        assert!(http_client(Duration::from_millis(250)).is_ok());
    }
}
