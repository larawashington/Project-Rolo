use std::fs;

use desktop_pet_lib::vault::bm25::BM25Index;
use desktop_pet_lib::vault::bootstrap::bootstrap_vault;
use tempfile::TempDir;

const GOLDEN_PREFERENCES: &str = "# User
## Preferences
User prefers Python over JavaScript. Loves debugging mysteries.
## Communication Style
User likes terse responses. Hates corporate-speak.
## Likes
User feeds Rolo SQL files most often. Affectionate response when food metaphors are used.
";

#[test]
fn deterministic_retrieval_matches_golden_table() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("vault");
    bootstrap_vault(&root).unwrap();

    let prefs_path = root.join("wiki/user/preferences.md");
    fs::write(&prefs_path, GOLDEN_PREFERENCES).unwrap();

    let idx = BM25Index::build(&root.join("wiki")).unwrap();

    let cases: &[(&str, Option<(&str, f64)>)] = &[
        ("python", Some(("user/preferences.md#Preferences", 0.5))),
        (
            "corporate",
            Some(("user/preferences.md#Communication Style", 0.5)),
        ),
        ("sql files", Some(("user/preferences.md#Likes", 0.3))),
        ("debugging", Some(("user/preferences.md#Preferences", 0.3))),
        // PRD §11.2 row 5 spec: "no terms match" → empty result. The PRD's literal
        // example query "rolo never seen this term" was authored against a corpus
        // containing only the golden preferences content; under a fully-bootstrapped
        // vault those words ("rolo", "never", "this") all appear in personality/*.
        // We substitute a string of nonsense tokens to honor the test's intent.
        ("zzzqqq xxxqqq vvvqqq", None),
    ];

    for (query, expected) in cases {
        let hits = idx.search(query, 5);
        match expected {
            None => {
                assert!(
                    hits.is_empty(),
                    "query {:?} expected empty result, got {:?}",
                    query,
                    hits.iter()
                        .map(|h| (&h.chunk.id, h.score))
                        .collect::<Vec<_>>()
                );
            }
            Some((expected_id, min_score)) => {
                assert!(
                    !hits.is_empty(),
                    "query {:?} expected hits, got none",
                    query
                );
                let top = &hits[0];
                assert_eq!(
                    top.chunk.id, *expected_id,
                    "query {:?} top-1 mismatch",
                    query
                );
                assert!(
                    top.score > *min_score,
                    "query {:?} score {} not greater than {}",
                    query,
                    top.score,
                    min_score
                );
            }
        }
    }
}
