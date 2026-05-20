use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub file_path: PathBuf,
    pub heading: String,
    pub body: String,
    pub tokens: Vec<String>,
}

/// Tokenize per PRD §6.3: split on non-alphanumeric (Unicode-aware), lowercase,
/// no stemming, no stopwords. Empty tokens are dropped.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for low in ch.to_lowercase() {
                current.push(low);
            }
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Strip HTML comments (`<!-- ... -->`) from text in a single pass without regex.
/// Unterminated comments are dropped through end of input — matches the
/// "treat as if it does not exist" contract from PRD §6.1 rule 3.
pub fn strip_html_comments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 4 <= bytes.len() && &bytes[i..i + 4] == b"<!--" {
            // find matching -->
            let mut j = i + 4;
            let mut found = false;
            while j + 3 <= bytes.len() {
                if &bytes[j..j + 3] == b"-->" {
                    found = true;
                    break;
                }
                j += 1;
            }
            if found {
                i = j + 3;
            } else {
                // unterminated: skip to end
                i = bytes.len();
            }
        } else {
            // safe because we only advance on ASCII-byte matches; the byte at i
            // is either an ASCII non-`<` or the start of a UTF-8 sequence we
            // append whole via char iteration below.
            // Re-walk via chars from i for correctness.
            let rest = &text[i..];
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Chunk one file's content per PRD §6.1.
/// `rel_path` is the file's path relative to wiki/, e.g. "user/preferences.md".
pub fn chunk_file(rel_path: &str, content: &str) -> Vec<Chunk> {
    let stripped = strip_html_comments(content);

    // First pass: extract H1 title (first `# ` line) and split body into sections by `## `.
    let mut title: Option<String> = None;
    let mut sections: Vec<(String, Vec<String>)> = Vec::new();
    // Synthetic preamble holder: lines that appear after H1 but before first H2
    // (or, if no H1 yet, before any header).
    let mut preamble_lines: Vec<String> = Vec::new();
    let mut current: Option<(String, Vec<String>)> = None;

    for line in stripped.lines() {
        if let Some(rest) = line.strip_prefix("# ") {
            // H1 — only the first one is the title; subsequent H1s become content lines.
            if title.is_none() {
                title = Some(rest.trim().to_string());
                continue;
            } else if let Some((_, body)) = current.as_mut() {
                body.push(line.to_string());
            } else {
                preamble_lines.push(line.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("## ") {
            // Boundary. Flush current section.
            if let Some(section) = current.take() {
                sections.push(section);
            }
            current = Some((rest.trim().to_string(), Vec::new()));
        } else {
            // Regular line (or H3+, which stays inside parent chunk).
            if let Some((_, body)) = current.as_mut() {
                body.push(line.to_string());
            } else {
                preamble_lines.push(line.to_string());
            }
        }
    }
    if let Some(section) = current.take() {
        sections.push(section);
    }

    let title_str = title.unwrap_or_default();
    let mut all_sections: Vec<(String, Vec<String>)> = Vec::new();

    // Add preamble first (if non-empty after whitespace-only check).
    let preamble_body = preamble_lines.join("\n");
    if !preamble_body.trim().is_empty() {
        all_sections.push(("(file preamble)".to_string(), preamble_lines));
    }
    all_sections.extend(sections);

    // Build chunks with collision-suffixing IDs.
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut chunks = Vec::new();
    for (heading, body_lines) in all_sections {
        let body_text = body_lines.join("\n");
        if body_text.trim().is_empty() {
            continue;
        }
        // body field: include the full chunk text (heading + content) per PRD §6.2.
        let mut body = String::new();
        if !title_str.is_empty() {
            body.push_str(&format!("# {}\n", title_str));
        }
        body.push_str(&format!("## {}\n", heading));
        body.push_str(&body_text);

        let count = seen.entry(heading.clone()).or_insert(0);
        *count += 1;
        let suffixed = if *count == 1 {
            heading.clone()
        } else {
            format!("{}-{}", heading, *count)
        };
        let id = format!("{}#{}", rel_path, suffixed);

        let tokens = tokenize(&body);
        chunks.push(Chunk {
            id,
            file_path: PathBuf::from(rel_path),
            heading: suffixed,
            body,
            tokens,
        });
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_on_non_alphanumeric_and_lowercases() {
        assert_eq!(
            tokenize("Hello, world! foo_bar"),
            vec!["hello", "world", "foo", "bar"]
        );
    }

    #[test]
    fn tokenize_empty_input() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn tokenize_unicode_lowercases() {
        // German sharp S lowercases via to_lowercase to "ss" — verify no panic.
        let toks = tokenize("Straße");
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0], "straße");
    }

    #[test]
    fn strip_html_comments_removes_provenance_block() {
        let input = "<!-- last_compiled_event_ts: null -->\n# Title\nbody";
        let out = strip_html_comments(input);
        assert_eq!(out, "\n# Title\nbody");
    }

    #[test]
    fn strip_html_comments_unterminated_drops_to_end() {
        let input = "before <!-- never closes\nbody";
        let out = strip_html_comments(input);
        assert_eq!(out, "before ");
    }

    #[test]
    fn file_with_only_h1_produces_no_chunks() {
        let chunks = chunk_file("foo.md", "<!-- x -->\n# Title\n");
        assert!(chunks.is_empty(), "got: {:?}", chunks);
    }

    #[test]
    fn file_with_preamble_and_two_h2s_produces_three_chunks() {
        let content = "<!-- x -->\n# Title\nIntro paragraph.\n\n## A\nbody a\n\n## B\nbody b\n";
        let chunks = chunk_file("user/preferences.md", content);
        assert_eq!(chunks.len(), 3, "chunks: {:?}", chunks);
        assert_eq!(chunks[0].id, "user/preferences.md#(file preamble)");
        assert_eq!(chunks[1].id, "user/preferences.md#A");
        assert_eq!(chunks[2].id, "user/preferences.md#B");
        // body includes the H1+H2 prefix
        assert!(chunks[1].body.contains("# Title"));
        assert!(chunks[1].body.contains("## A"));
        assert!(chunks[1].body.contains("body a"));
    }

    #[test]
    fn html_comment_text_does_not_appear_in_tokens() {
        let content = "<!-- secret_marker_xyzzy -->\n# T\n## H\nbody\n";
        let chunks = chunk_file("a.md", content);
        for c in &chunks {
            for t in &c.tokens {
                assert_ne!(t, "secret_marker_xyzzy");
                assert_ne!(t, "xyzzy");
            }
        }
    }

    #[test]
    fn empty_h2_body_dropped() {
        let content = "# T\n## A\n## B\nhas body\n";
        let chunks = chunk_file("a.md", content);
        // A is empty -> dropped. B has body -> kept.
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, "B");
    }

    #[test]
    fn heading_collision_suffixes_with_dash_n() {
        let content = "# T\n## Same\nfirst\n## Same\nsecond\n";
        let chunks = chunk_file("a.md", content);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].id, "a.md#Same");
        assert_eq!(chunks[1].id, "a.md#Same-2");
    }

    #[test]
    fn h3_stays_inside_parent_chunk() {
        let content = "# T\n## A\nintro\n### sub\nmore\n";
        let chunks = chunk_file("a.md", content);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].body.contains("### sub"));
        assert!(chunks[0].body.contains("more"));
    }
}
