//! Privacy filters for perception events.
//!
//! Trivial substring/exact-match denylists. False negatives are accepted in
//! exchange for code clarity (PRD §Decision Log #6).

const SENSITIVE_SUBSTRINGS: &[&str] = &[
    "password",
    "secret",
    "private",
    "bank",
    "1password",
    "keychain",
    ".ssh",
    ".key",
    ".pem",
    ".env",
];

const SENSITIVE_APPS: &[&str] = &[
    "1Password",
    "1Password 7",
    "Keychain Access",
    // macOS Sequoia's built-in replacement for Keychain Access.
    "Passwords",
    "Bitwarden",
    "Dashlane",
];

pub fn is_filename_safe(name: &str) -> bool {
    let lower = name.to_lowercase();
    !SENSITIVE_SUBSTRINGS.iter().any(|s| lower.contains(s))
}

pub fn is_app_sensitive(app_name: &str) -> bool {
    SENSITIVE_APPS.contains(&app_name)
}

pub fn is_window_title_safe(title: &str) -> bool {
    is_filename_safe(title)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filename_safe_passes_innocuous() {
        assert!(is_filename_safe("report.pdf"));
        assert!(is_filename_safe("vacation_photos.zip"));
    }

    #[test]
    fn filename_safe_blocks_sensitive_substrings() {
        assert!(!is_filename_safe("my_passwords.txt"));
        assert!(!is_filename_safe("Secret_Plan.docx"));
        assert!(!is_filename_safe("BankStatement.pdf"));
        assert!(!is_filename_safe("config.env"));
        assert!(!is_filename_safe("id_rsa.pem"));
    }

    #[test]
    fn filename_safe_is_case_insensitive() {
        assert!(!is_filename_safe("PASSWORD.txt"));
        assert!(!is_filename_safe("BANK.pdf"));
    }

    #[test]
    fn app_sensitive_exact_match() {
        assert!(is_app_sensitive("1Password"));
        assert!(is_app_sensitive("1Password 7"));
        assert!(is_app_sensitive("Keychain Access"));
        assert!(is_app_sensitive("Passwords"));
        assert!(!is_app_sensitive("Safari"));
        assert!(!is_app_sensitive("1Password Helper"));
    }

    #[test]
    fn window_title_safe_blocks_bank() {
        assert!(!is_window_title_safe("Inbox - Bank of America"));
        assert!(is_window_title_safe("Inbox - Mail"));
    }
}
