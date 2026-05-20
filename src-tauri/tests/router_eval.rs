//! Live-Ollama integration smoke for the tool-layer router (PRD/rolo-tool-layer.md T4).
//!
//! Gated on `ROLO_TEST_OLLAMA=1` and `#[ignore]`d so it never blocks CI.
//! Manual run command:
//!
//! ```sh
//! ROLO_TEST_OLLAMA=1 cargo test \
//!     --manifest-path src-tauri/Cargo.toml \
//!     --test router_eval -- --ignored --nocapture
//! ```
//!
//! Optional knobs:
//! - `ROLO_ROUTER_MODEL` — override the default `gemma3:4b`. Risk #2 in the
//!   PRD says if Gemma loops under `format`, flip this to `llama3.2:3b`.
//! - `ROLO_ROUTER_BASE_URL` — point at a non-default Ollama endpoint.
//!
//! Done-when gates (PRD §6 T4):
//! - Overall accuracy ≥ 0.85
//! - Memory-probe (search_vault) accuracy ≥ 0.90
//! - Malformed-JSON rate ≤ 2/N (i.e., 2 out of 10)
//! - p50 latency under `format` ≤ 1.5× unconstrained baseline
//!
//! When live Ollama is genuinely flaky on the chosen model, the test prints
//! every metric to stderr and softens the latency assert (the rest stay hard).
//! the user verifies this manually before flipping the dispatcher flag.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use desktop_pet_lib::ollama_router::{ollama_chat_with_format, OLLAMA_BASE_URL_DEFAULT};
use desktop_pet_lib::tools::router::{router_format_schema, router_system_prompt, RouterEnvelope};
use desktop_pet_lib::tools::ToolRegistry;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct SmokeRow {
    input: String,
    expected_action: String,
    #[serde(default)]
    expected_tool: Option<String>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("router_smoke.jsonl")
}

fn load_fixture() -> Vec<SmokeRow> {
    let raw = std::fs::read_to_string(fixture_path()).expect("read router_smoke fixture");
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("fixture line {} not JSON: {line:?} — {e}", i + 1))
        })
        .collect()
}

/// Median + p95 of a sorted slice. Returns `(p50_ms, p95_ms)` with both
/// values rounded to the nearest millisecond.
fn percentiles(samples_ms: &mut [u128]) -> (u128, u128) {
    samples_ms.sort();
    let n = samples_ms.len();
    if n == 0 {
        return (0, 0);
    }
    let p50 = samples_ms[n / 2];
    let p95_idx = ((n as f64) * 0.95).ceil() as usize;
    let p95 = samples_ms[p95_idx.saturating_sub(1).min(n - 1)];
    (p50, p95)
}

#[tokio::test]
#[ignore = "requires live Ollama; run with ROLO_TEST_OLLAMA=1 cargo test --test router_eval -- --ignored"]
async fn router_smoke_accuracy() {
    if std::env::var("ROLO_TEST_OLLAMA").is_err() {
        eprintln!("ROLO_TEST_OLLAMA not set — skipping live router smoke");
        return;
    }

    let model = std::env::var("ROLO_ROUTER_MODEL").unwrap_or_else(|_| "gemma3:4b".into());
    let base_url =
        std::env::var("ROLO_ROUTER_BASE_URL").unwrap_or_else(|_| OLLAMA_BASE_URL_DEFAULT.into());
    let registry = ToolRegistry::standard();
    let sys_prompt = router_system_prompt(&registry);
    let schema = router_format_schema();
    let timeout = Duration::from_secs(20);

    let rows = load_fixture();
    assert!(
        rows.len() >= 10,
        "fixture must have ≥10 rows, found {}",
        rows.len()
    );

    // ---- Pass 1: constrained baseline (`format` set) ------------------------
    let mut hits = 0usize;
    let mut memory_hits = 0usize;
    let mut memory_total = 0usize;
    let mut malformed = 0usize;
    let mut latencies: Vec<u128> = Vec::with_capacity(rows.len());
    let mut misses: Vec<String> = Vec::new();

    for row in &rows {
        if row.expected_action == "tool" && row.expected_tool.as_deref() == Some("search_vault") {
            memory_total += 1;
        }

        let started = Instant::now();
        let result =
            ollama_chat_with_format(&base_url, &model, &sys_prompt, &row.input, schema, timeout)
                .await;
        let elapsed_ms = started.elapsed().as_millis();
        latencies.push(elapsed_ms);

        let raw: Value = match result {
            Ok(v) => v,
            Err(e) => {
                malformed += 1;
                misses.push(format!("  - input={:?} transport_error={e}", row.input));
                continue;
            }
        };

        let env: RouterEnvelope = match serde_json::from_value(raw.clone()) {
            Ok(env) => env,
            Err(e) => {
                malformed += 1;
                misses.push(format!(
                    "  - input={:?} envelope_decode_error={e} raw={raw}",
                    row.input
                ));
                continue;
            }
        };

        let ok = env.action == row.expected_action
            && match (&row.expected_tool, &env.tool) {
                (Some(want), got) => got.as_deref() == Some(want.as_str()),
                (None, _) => true,
            };

        if ok {
            hits += 1;
            if row.expected_action == "tool" && row.expected_tool.as_deref() == Some("search_vault")
            {
                memory_hits += 1;
            }
        } else {
            misses.push(format!(
                "  - input={:?} expected={}({:?}) got={}({:?})",
                row.input, row.expected_action, row.expected_tool, env.action, env.tool
            ));
        }
    }

    let total = rows.len();
    let accuracy = (hits as f64) / (total as f64);
    let memory_accuracy = if memory_total == 0 {
        1.0
    } else {
        (memory_hits as f64) / (memory_total as f64)
    };
    let (p50_ms, p95_ms) = percentiles(&mut latencies.clone());

    // ---- Pass 2: unconstrained latency baseline -----------------------------
    // The PRD wants ≤1.5× latency under `format`. We measure the same
    // prompts WITHOUT `format` to compute the ratio. Keep this short — we
    // only need a baseline median, not full accuracy.
    let mut baseline_latencies: Vec<u128> = Vec::with_capacity(rows.len());
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("baseline reqwest client builds");
    for row in &rows {
        let body = serde_json::json!({
            "model": model,
            "messages": [
                { "role": "system", "content": sys_prompt },
                { "role": "user",   "content": row.input },
            ],
            "stream": false,
            "options": { "temperature": 0.0 },
        });
        let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
        let started = Instant::now();
        let _ = client.post(&url).json(&body).send().await;
        baseline_latencies.push(started.elapsed().as_millis());
    }
    let (baseline_p50, _) = percentiles(&mut baseline_latencies.clone());
    let latency_ratio = if baseline_p50 == 0 {
        1.0
    } else {
        (p50_ms as f64) / (baseline_p50 as f64)
    };

    // ---- Report -------------------------------------------------------------
    eprintln!("\n[router_eval]");
    eprintln!("  model:              {model}");
    eprintln!("  rows:               {total}");
    eprintln!(
        "  overall accuracy:   {hits}/{total} = {:.0}%",
        accuracy * 100.0
    );
    eprintln!(
        "  memory-probe acc:   {memory_hits}/{memory_total} = {:.0}%",
        memory_accuracy * 100.0
    );
    eprintln!("  malformed JSON:     {malformed}/{total}");
    eprintln!(
        "  latency p50:        {p50_ms}ms (constrained) vs {baseline_p50}ms (baseline) = {:.2}x",
        latency_ratio
    );
    eprintln!("  latency p95:        {p95_ms}ms");
    if !misses.is_empty() {
        eprintln!("  misses:");
        for m in &misses {
            eprintln!("{m}");
        }
    }

    // ---- Assertions (PRD T4 done-when gates) --------------------------------
    assert!(
        accuracy >= 0.85,
        "overall accuracy {:.0}% < 85%",
        accuracy * 100.0
    );
    assert!(
        memory_accuracy >= 0.90,
        "memory-probe accuracy {:.0}% < 90%",
        memory_accuracy * 100.0
    );
    let malformed_budget = ((total as f64) * 0.20).ceil() as usize; // 2/10
    assert!(
        malformed <= malformed_budget,
        "malformed JSON {malformed}/{total} exceeds budget {malformed_budget}"
    );
    // Latency assertion is print-only (soft) when the baseline is suspect:
    // if the baseline p50 was zero (Ollama not responding) or the ratio is
    // wildly off, skip the assert and let the human decide. Otherwise
    // enforce ≤1.5×.
    if baseline_p50 > 0 && latency_ratio.is_finite() {
        assert!(
            latency_ratio <= 1.5,
            "constrained latency {p50_ms}ms is {:.2}× the {baseline_p50}ms baseline (>1.5×)",
            latency_ratio
        );
    } else {
        eprintln!(
            "  (latency assertion skipped — baseline p50 was {baseline_p50}ms, not comparable)"
        );
    }
}
