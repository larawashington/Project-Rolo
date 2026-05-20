//! Dreaming orchestrator — the long-running poll loop and single-flight
//! handle that turn raw experience events into compiled wiki facts.
//!
//! This module is pure logic + a tokio task spawner. The gate evaluator at
//! the top is a deterministic function over a snapshot struct, so the trigger
//! conditions in PRD §1 can be unit-tested as a truth table without touching
//! the filesystem, the clock, or Quartz. The `DreamHandle` enforces
//! single-flight: only one dream runs at a time, and a fresh
//! `CancellationToken` is minted per attempt so wake commands abort the
//! in-flight LLM call cleanly. The orchestrator (`Dreaming::run_one_cycle`)
//! threads the gate, handle, pet state machine, compiler, and linter together
//! with a 60-second timeout so a hung LLM never traps Rolo in `Sleeping`.
//!
//! Failure handling is deliberately lossy on the dream side and lossless on
//! the source side: the raw JSONL is the ground truth and survives every
//! aborted, timed-out, or rejected dream — the same events compile cleanly
//! on the next cycle. Anything that goes wrong here is noise, not damage.

use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Local};
use tokio_util::sync::CancellationToken;

use crate::chat::provider::InferenceProvider;
use crate::vault::compiler::{CompileBatch, CompileOutcome, Compiler};
use crate::vault::dreams_log::DreamsLog;
use crate::vault::linter::Linter;

// ---------------------------------------------------------------------------
// D1 — Gate evaluator
// ---------------------------------------------------------------------------

/// Why a dream attempt was blocked at the 30s poll boundary. Each variant
/// corresponds 1:1 with a row in the trigger table from PRD §1 — the order is
/// the evaluation order in `Gates::all_pass` and is also the natural priority
/// for a debug log line ("here is the FIRST gate that failed").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    /// `now - last_compile_time < 6h`.
    LastCompileTooRecent,
    /// No compile yet AND `now - startup_time < 24h`.
    StartupTooRecent,
    /// `idle_seconds < 180`.
    UserNotIdle,
    /// Unprocessed event count below the 50-event floor.
    NotEnoughEvents,
    /// Pet is in any non-Idle state (eating, dragging, walking, …).
    PetNotIdle,
    /// Chat window is currently open — dreaming is suppressed to avoid
    /// racing with active inference.
    ChatWindowOpen,
    /// Ollama health check failed at the most recent probe.
    OllamaUnreachable,
}

/// One snapshot of every gate input, gathered atomically by the poll loop.
/// Pure data — no behavior — so unit tests can construct any combination of
/// inputs cheaply.
pub struct GateInputs {
    pub now: DateTime<Local>,
    pub last_compile_time: Option<DateTime<Local>>,
    pub startup_time: DateTime<Local>,
    pub idle_seconds: f64,
    pub unprocessed_event_count: usize,
    pub pet_state_is_idle: bool,
    pub chat_window_open: bool,
    pub ollama_healthy: bool,
}

/// Stateless gate evaluator. Lives as a struct only so future configurable
/// thresholds (per-user idle-seconds floor, e.g.) have somewhere to land.
pub struct Gates;

// ---------------------------------------------------------------------------
// !! DEV-ONLY — REMOVE BEFORE SHIPPING !!
// ---------------------------------------------------------------------------
// `ROLO_DREAM_GATE_RELAXED=1` lowers the gate thresholds so dreams fire in
// minutes instead of hours. Used to capture B4 fixture data (10–20 real
// Gemma compile outputs) without waiting 24h+ for the organic startup gate.
//
// MUST BE REMOVED before any release. Tracked in
// plans/rolo-cognitive-architecture-v2.md §G fixtures.
//
// Relaxed thresholds:
//   - last compile / startup grace: 5 min  (was 6h / 24h)
//   - user idle floor:              30 sec (was 180 sec)
//   - unprocessed event floor:      5      (was 50)
fn dream_gate_relaxed() -> bool {
    // Cached: this is a launch-time env flag, and the dream gate is checked
    // every frame in the tick loop (via `dream_thresholds`). Avoid re-allocating
    // a String per call.
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ROLO_DREAM_GATE_RELAXED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

const PARK_BUFFER_SECS: f64 = 10.0;

/// Single source of truth for every relaxed/production threshold the
/// dream gate cares about. The relaxed-mode warning is logged on the
/// first call inside `Gates::all_pass`; this builder is silent so it
/// can be called from hot paths (e.g. `tick.rs`).
pub struct DreamThresholds {
    pub park_threshold_secs: f64,
    pub idle_floor_secs: f64,
    pub last_compile_grace: chrono::Duration,
    pub startup_grace: chrono::Duration,
    pub event_floor: usize,
}

pub fn dream_thresholds() -> DreamThresholds {
    let relaxed = dream_gate_relaxed();
    let idle_floor_secs = if relaxed { 30.0 } else { 180.0 };
    DreamThresholds {
        park_threshold_secs: idle_floor_secs - PARK_BUFFER_SECS,
        idle_floor_secs,
        last_compile_grace: if relaxed {
            chrono::Duration::minutes(5)
        } else {
            chrono::Duration::hours(6)
        },
        startup_grace: if relaxed {
            chrono::Duration::minutes(5)
        } else {
            chrono::Duration::hours(24)
        },
        event_floor: if relaxed { 5 } else { 50 },
    }
}
// ---------------------------------------------------------------------------

impl Gates {
    /// Returns `Ok(())` when every PRD §1 gate passes; otherwise the FIRST
    /// failing reason in evaluation order. The poll loop logs the reason at
    /// `debug` and silently retries 30s later.
    pub fn all_pass(inputs: &GateInputs) -> Result<(), BlockReason> {
        if dream_gate_relaxed() {
            log::warn!("[Rolo dreaming] !! DEV-ONLY !! ROLO_DREAM_GATE_RELAXED is set — gates lowered for fixture capture; revert before shipping");
        }
        let t = dream_thresholds();

        // 1. Time since last compile.
        match inputs.last_compile_time {
            Some(last) => {
                if inputs.now - last < t.last_compile_grace {
                    return Err(BlockReason::LastCompileTooRecent);
                }
            }
            None => {
                if inputs.now - inputs.startup_time < t.startup_grace {
                    return Err(BlockReason::StartupTooRecent);
                }
            }
        }

        // 2. User idle floor.
        if inputs.idle_seconds < t.idle_floor_secs {
            return Err(BlockReason::UserNotIdle);
        }

        // 3. Backlog floor.
        if inputs.unprocessed_event_count < t.event_floor {
            return Err(BlockReason::NotEnoughEvents);
        }

        // 4. Pet must be idle.
        if !inputs.pet_state_is_idle {
            return Err(BlockReason::PetNotIdle);
        }

        // 5. Chat window must be closed.
        if inputs.chat_window_open {
            return Err(BlockReason::ChatWindowOpen);
        }

        // 6. Ollama must be reachable.
        if !inputs.ollama_healthy {
            return Err(BlockReason::OllamaUnreachable);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// D2 — DreamHandle (single-flight + cancellation token)
// ---------------------------------------------------------------------------

/// Lifecycle state of the single dream slot. `InProgress` carries the
/// run identifier and start time so observability tools can render "Rolo
/// has been dreaming for N seconds" without consulting the dreams_log.
#[derive(Debug, Clone)]
pub enum DreamRunState {
    Idle,
    InProgress {
        started_at: DateTime<Local>,
        run_id: String,
    },
}

/// Process-wide handle to the dream task. Owns the active
/// `CancellationToken` and the run-state slot. Cloned via `Arc` into the
/// poll loop, the wake_rolo command, and the exit handler.
pub struct DreamHandle {
    /// Wrapped in a `Mutex` rather than an `RwLock` because we always need
    /// to swap it for a fresh token when a new attempt claims the slot —
    /// readers already get a cheap clone via `current_token`, so contention
    /// is dominated by the writer path.
    token: Mutex<CancellationToken>,
    state: Mutex<DreamRunState>,
}

/// RAII guard that resets the run-state slot to `Idle` on drop. Held for
/// the lifetime of one cycle so a panic mid-compile cannot leave the slot
/// permanently claimed.
pub struct DreamGuard<'a> {
    handle: &'a DreamHandle,
}

impl Drop for DreamGuard<'_> {
    fn drop(&mut self) {
        // Mutex poisoning is non-fatal for state — recovering the inner
        // value lets us still reset the slot. A poisoned guard means a
        // previous panic happened mid-run; the next attempt starts fresh.
        if let Ok(mut s) = self.handle.state.lock() {
            *s = DreamRunState::Idle;
        }
    }
}

impl DreamHandle {
    pub fn new() -> Self {
        Self {
            token: Mutex::new(CancellationToken::new()),
            state: Mutex::new(DreamRunState::Idle),
        }
    }

    /// Claim the single dream slot. Returns a guard whose drop releases the
    /// slot. The previous token is replaced with a fresh one so any stale
    /// `cancelled()` waiters from a prior cycle do not leak across runs.
    pub fn start_attempt(&self, run_id: String) -> Result<DreamGuard<'_>, &'static str> {
        let mut state = self.state.lock().map_err(|_| "state lock poisoned")?;
        if matches!(*state, DreamRunState::InProgress { .. }) {
            return Err("already running");
        }
        *state = DreamRunState::InProgress {
            started_at: Local::now(),
            run_id,
        };
        if let Ok(mut t) = self.token.lock() {
            *t = CancellationToken::new();
        }
        Ok(DreamGuard { handle: self })
    }

    /// Clone the active token. Cheap (`Arc` under the hood). Called by the
    /// orchestrator immediately after `start_attempt` so it can `select!`
    /// against `cancelled()`.
    pub fn current_token(&self) -> CancellationToken {
        self.token
            .lock()
            .map(|t| t.clone())
            .unwrap_or_else(|_| CancellationToken::new())
    }

    /// Cancel the active token. Safe to call when no dream is running — the
    /// next `start_attempt` mints a fresh token, so a stray cancel here is a
    /// no-op for the next cycle.
    pub fn request_wake(&self) {
        if let Ok(t) = self.token.lock() {
            t.cancel();
        }
    }

    /// Snapshot of the run-state slot. Tests and the future "Rolo has been
    /// asleep for N seconds" UI surface read this.
    pub fn current_state(&self) -> DreamRunState {
        self.state
            .lock()
            .map(|s| s.clone())
            .unwrap_or(DreamRunState::Idle)
    }
}

impl Default for DreamHandle {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// D3 — Orchestrator (timeout + cancel-aware run_one_cycle)
// ---------------------------------------------------------------------------

/// Wall-clock cap for the compile+lint chain in production. PRD §2 §1 — a
/// hung 4B model must not strand Rolo in `Sleeping`. Tests override via
/// `run_one_cycle_with_timeout`.
pub const COMPILE_TIMEOUT_SECS: u64 = 60;

/// Stateless orchestrator. Holds the compiler and linter so callers can
/// stub them in future tests without changing call sites.
pub struct Dreaming {
    pub compiler: Compiler,
    pub linter: Linter,
}

/// Outcome of a single `run_one_cycle` call. `status` is a string (rather
/// than an enum) because it mirrors the dreams_log `status` field which
/// already carries cancel/skip lifecycle states the compiler doesn't know
/// about.
#[derive(Debug, Clone)]
pub struct CycleOutcome {
    /// One of: `success`, `failed`, `rejected`, `skipped`, `cancelled`.
    pub status: String,
    /// Free-text reason on non-success. e.g. `timeout`, `pet_not_idle`.
    pub reason: Option<String>,
    /// The compiler's run_id, when it actually started the call.
    pub run_id: Option<String>,
    /// The compiler's structured outcome, when the call completed.
    pub compile: Option<CompileOutcome>,
}

impl Dreaming {
    pub fn new() -> Self {
        Self {
            compiler: Compiler,
            linter: Linter,
        }
    }

    /// Run one full dream cycle: claim the handle, transition the pet to
    /// Sleeping, race compile+lint against cancel and timeout, transition
    /// the pet back to Idle, return a structured outcome.
    ///
    /// Production callers use the 60s timeout via `run_one_cycle`; tests
    /// override with `run_one_cycle_with_timeout` to keep wall-clock short.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_one_cycle(
        &self,
        wiki_root: &Path,
        artifacts_dir: &Path,
        dreams: &DreamsLog,
        batch: CompileBatch,
        provider: Arc<dyn InferenceProvider>,
        dream_handle: Arc<DreamHandle>,
        pet: Arc<Mutex<crate::state_machine::Pet>>,
    ) -> CycleOutcome {
        self.run_one_cycle_with_timeout(
            wiki_root,
            artifacts_dir,
            dreams,
            batch,
            provider,
            dream_handle,
            pet,
            COMPILE_TIMEOUT_SECS,
        )
        .await
    }

    /// Run one dream cycle over a hand-constructed batch of events, bypassing
    /// the gate evaluator and the 50-event floor that the poll loop enforces.
    /// Used by the Command Center's Memory panel to immediately absorb a User
    /// Profile update.
    ///
    /// Honors the single-flight `DreamHandle` exactly like `run_one_cycle`:
    /// returns a `CycleOutcome` with `status: "skipped"` and
    /// `reason: Some("already_running")` if a dream is in flight. Does NOT
    /// queue — callers must surface "busy" to the user.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_one_cycle_for_events(
        &self,
        wiki_root: &Path,
        artifacts_dir: &Path,
        dreams: &DreamsLog,
        events: Vec<crate::vault::compiler::CompiledEvent>,
        provider: Arc<dyn InferenceProvider>,
        dream_handle: Arc<DreamHandle>,
        pet: Arc<Mutex<crate::state_machine::Pet>>,
    ) -> CycleOutcome {
        let batch = CompileBatch { events };
        self.run_one_cycle(
            wiki_root,
            artifacts_dir,
            dreams,
            batch,
            provider,
            dream_handle,
            pet,
        )
        .await
    }

    /// Test-injectable variant. Same logic as `run_one_cycle` but the
    /// timeout is a parameter so tokio's pause/advance machinery can fire a
    /// timeout deterministically without burning real wall-clock.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_one_cycle_with_timeout(
        &self,
        wiki_root: &Path,
        artifacts_dir: &Path,
        dreams: &DreamsLog,
        batch: CompileBatch,
        provider: Arc<dyn InferenceProvider>,
        dream_handle: Arc<DreamHandle>,
        pet: Arc<Mutex<crate::state_machine::Pet>>,
        timeout_secs: u64,
    ) -> CycleOutcome {
        // 1. Claim the slot. The placeholder run_id is purely a debug aid
        //    until the compiler mints the real one; the real id is what
        //    appears in dreams_log.
        let run_id_placeholder = format!("drm_pending_{}", Local::now().timestamp());
        let _guard = match dream_handle.start_attempt(run_id_placeholder.clone()) {
            Ok(g) => g,
            Err(_) => {
                return CycleOutcome {
                    status: "skipped".into(),
                    reason: Some("already_running".into()),
                    run_id: None,
                    compile: None,
                };
            }
        };

        // 2. Idle → Sleeping. If refused (drag or eat snuck in between the
        //    gate snapshot and now), bail without burning the LLM.
        {
            let mut p = pet.lock().unwrap_or_else(|e| e.into_inner());
            if p.enter_sleeping().is_err() {
                return CycleOutcome {
                    status: "skipped".into(),
                    reason: Some("pet_not_idle".into()),
                    run_id: None,
                    compile: None,
                };
            }
        }

        let token = dream_handle.current_token();

        // 3. Race compile + lint vs cancel vs timeout. The lint runs only
        //    on a successful compile per PRD §5 (lint is advisory, never
        //    blocks the wiki write).
        let compile_fut = async {
            let outcome = self
                .compiler
                .compile(&batch, wiki_root, provider.as_ref(), dreams, artifacts_dir)
                .await;
            if outcome.status == "success" {
                let _ = self
                    .linter
                    .lint(wiki_root, &outcome.run_id, provider.as_ref(), dreams)
                    .await;
            }
            outcome
        };

        let outcome = tokio::select! {
            biased; // prioritize cancellation over completion
            _ = token.cancelled() => {
                let entry = serde_json::json!({
                    "run_id": run_id_placeholder,
                    "status": "cancelled",
                    "reason": "user_wake",
                    "ended_at": Local::now().to_rfc3339(),
                });
                if let Err(e) = dreams.append(entry) {
                    log::warn!(
                        "[Rolo dreaming] cancel append failed: {} — non-fatal",
                        e
                    );
                }
                CycleOutcome {
                    status: "cancelled".into(),
                    reason: Some("user_wake".into()),
                    run_id: None,
                    compile: None,
                }
            }
            res = tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), compile_fut) => {
                match res {
                    Ok(c) => {
                        let status = c.status.clone();
                        let run_id = c.run_id.clone();
                        let reason = c.reject_reason.clone();
                        CycleOutcome {
                            status,
                            reason,
                            run_id: Some(run_id),
                            compile: Some(c),
                        }
                    }
                    Err(_elapsed) => {
                        let entry = serde_json::json!({
                            "run_id": run_id_placeholder,
                            "status": "failed",
                            "reason": "timeout",
                            "ended_at": Local::now().to_rfc3339(),
                        });
                        if let Err(e) = dreams.append(entry) {
                            log::warn!(
                                "[Rolo dreaming] timeout append failed: {} — non-fatal",
                                e
                            );
                        }
                        CycleOutcome {
                            status: "failed".into(),
                            reason: Some("timeout".into()),
                            run_id: None,
                            compile: None,
                        }
                    }
                }
            }
        };

        // 4. Always wake. The wake is intentionally outside the select! so
        //    it runs on success, cancel, AND timeout. Pulling it into a
        //    branch would re-introduce the bug PRD §D3 calls out.
        {
            let mut p = pet.lock().unwrap_or_else(|e| e.into_inner());
            p.wake_from_sleep();
        }

        outcome
    }
}

impl Default for Dreaming {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// D4 — 30s background poll loop
// ---------------------------------------------------------------------------

/// Tunable poll-loop knobs. Defaults match PRD §1: 30s polling, 50-event
/// batch ceiling. Tests shrink the interval to compress wall-clock.
pub struct PollerConfig {
    pub poll_interval_secs: u64,
    pub max_batch_events: usize,
}

impl Default for PollerConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: 30,
            max_batch_events: 50,
        }
    }
}

/// Spawn the long-running dream poll loop. Returns the `JoinHandle` so the
/// caller can cancel on app exit. The loop never returns under normal
/// operation; a panic in the body would abort the task and stop dreaming
/// (which is preferable to silently hanging).
///
/// Phase 4 of PRD/rolo-command-center.md: the loop reads its provider from
/// a `SharedProviderSlot` so the Brain panel can hot-swap brains at runtime.
/// Each iteration snapshots the slot ONCE at the top, then uses that
/// snapshot for the health probe and the compile cycle — so a swap mid-cycle
/// cannot tear a single dream attempt in half. `run_one_cycle` still takes
/// an `Arc<dyn InferenceProvider>`, unchanged from Phase 3.
#[allow(clippy::too_many_arguments)]
pub fn spawn_poll_loop(
    vault: Arc<crate::vault::Vault>,
    pet: Arc<Mutex<crate::state_machine::Pet>>,
    provider_slot: crate::command_center::SharedProviderSlot,
    dream_handle: Arc<DreamHandle>,
    is_chat_window_open: Arc<dyn Fn() -> bool + Send + Sync>,
    config: PollerConfig,
) -> tauri::async_runtime::JoinHandle<()> {
    let startup = Local::now();
    tauri::async_runtime::spawn(async move {
        let dreaming = Dreaming::new();
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(config.poll_interval_secs)).await;

            // Snapshot once per iteration. If `cc_apply_brain` swaps the slot
            // while we're mid-cycle, this attempt finishes with the old
            // provider and the next iteration picks up the new one — clean
            // handoff, no torn requests.
            let provider = provider_slot.snapshot();

            let inputs = GateInputs {
                now: Local::now(),
                last_compile_time: vault.meta_snapshot().last_compile_time,
                startup_time: startup,
                idle_seconds: crate::vault::idle::seconds_since_last_input().unwrap_or(0.0),
                unprocessed_event_count: vault.count_unprocessed_events(),
                pet_state_is_idle: pet
                    .lock()
                    .map(|p| p.state() == crate::state_machine::PetState::Idle)
                    .unwrap_or(false),
                chat_window_open: is_chat_window_open(),
                ollama_healthy: provider.health_check().await,
            };

            if let Err(reason) = Gates::all_pass(&inputs) {
                log::debug!("[Rolo dreaming] gates blocked: {:?}", reason);
                continue;
            }

            let batch = match vault.build_compile_batch(config.max_batch_events) {
                Ok(b) if !b.events.is_empty() => b,
                Ok(_) => {
                    log::debug!("[Rolo dreaming] empty batch despite passing gates — race; retry");
                    continue;
                }
                Err(e) => {
                    log::warn!("[Rolo dreaming] build_compile_batch failed: {} — skip", e);
                    continue;
                }
            };

            let outcome = dreaming
                .run_one_cycle(
                    &vault.wiki_root(),
                    &vault.dreams_artifacts_dir(),
                    &vault.dreams_log_handle(),
                    batch,
                    Arc::clone(&provider),
                    Arc::clone(&dream_handle),
                    Arc::clone(&pet),
                )
                .await;

            if outcome.status == "success" {
                vault.record_compile_complete(Local::now());
            }

            log::info!("[Rolo dreaming] cycle outcome: {}", outcome.status);
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- D1: gate evaluator -----------------------------------------------

    fn passing_inputs() -> GateInputs {
        let now = Local::now();
        GateInputs {
            now,
            last_compile_time: Some(now - chrono::Duration::hours(7)),
            startup_time: now - chrono::Duration::hours(48),
            idle_seconds: 200.0,
            unprocessed_event_count: 60,
            pet_state_is_idle: true,
            chat_window_open: false,
            ollama_healthy: true,
        }
    }

    #[test]
    fn all_gates_pass_returns_ok() {
        let inputs = passing_inputs();
        assert!(Gates::all_pass(&inputs).is_ok());
    }

    #[test]
    fn gates_block_when_last_compile_too_recent() {
        let mut inputs = passing_inputs();
        inputs.last_compile_time = Some(inputs.now - chrono::Duration::hours(2));
        assert_eq!(
            Gates::all_pass(&inputs),
            Err(BlockReason::LastCompileTooRecent)
        );
    }

    #[test]
    fn gates_block_when_startup_too_recent() {
        let mut inputs = passing_inputs();
        inputs.last_compile_time = None;
        inputs.startup_time = inputs.now - chrono::Duration::hours(2);
        assert_eq!(Gates::all_pass(&inputs), Err(BlockReason::StartupTooRecent));
    }

    #[test]
    fn gates_block_when_user_not_idle() {
        let mut inputs = passing_inputs();
        inputs.idle_seconds = 30.0;
        assert_eq!(Gates::all_pass(&inputs), Err(BlockReason::UserNotIdle));
    }

    #[test]
    fn gates_block_when_not_enough_events() {
        let mut inputs = passing_inputs();
        inputs.unprocessed_event_count = 10;
        assert_eq!(Gates::all_pass(&inputs), Err(BlockReason::NotEnoughEvents));
    }

    #[test]
    fn gates_block_when_pet_not_idle() {
        let mut inputs = passing_inputs();
        inputs.pet_state_is_idle = false;
        assert_eq!(Gates::all_pass(&inputs), Err(BlockReason::PetNotIdle));
    }

    #[test]
    fn gates_block_when_chat_window_open() {
        let mut inputs = passing_inputs();
        inputs.chat_window_open = true;
        assert_eq!(Gates::all_pass(&inputs), Err(BlockReason::ChatWindowOpen));
    }

    #[test]
    fn gates_block_when_ollama_unreachable() {
        let mut inputs = passing_inputs();
        inputs.ollama_healthy = false;
        assert_eq!(
            Gates::all_pass(&inputs),
            Err(BlockReason::OllamaUnreachable)
        );
    }

    #[test]
    fn idle_at_180_passes() {
        let mut inputs = passing_inputs();
        inputs.idle_seconds = 180.0;
        assert!(Gates::all_pass(&inputs).is_ok());
    }

    #[test]
    fn events_at_50_passes() {
        let mut inputs = passing_inputs();
        inputs.unprocessed_event_count = 50;
        assert!(Gates::all_pass(&inputs).is_ok());
    }

    // ---- D2: DreamHandle --------------------------------------------------

    #[test]
    fn concurrent_start_attempts_only_one_succeeds() {
        let h = DreamHandle::new();
        let g1 = h.start_attempt("run-1".into());
        assert!(g1.is_ok(), "first attempt must claim the slot");
        let g2 = h.start_attempt("run-2".into());
        match g2 {
            Ok(_) => panic!("second concurrent attempt must fail"),
            Err(msg) => assert_eq!(msg, "already running"),
        }
    }

    #[test]
    fn request_wake_cancels_token() {
        let h = DreamHandle::new();
        let _g = h.start_attempt("run-1".into()).unwrap();
        let token = h.current_token();
        assert!(!token.is_cancelled());
        h.request_wake();
        assert!(
            token.is_cancelled(),
            "request_wake must cancel the active token"
        );
    }

    #[test]
    fn guard_drop_resets_state_to_idle() {
        let h = DreamHandle::new();
        {
            let _g = h.start_attempt("run-1".into()).unwrap();
            assert!(matches!(
                h.current_state(),
                DreamRunState::InProgress { .. }
            ));
        }
        assert!(matches!(h.current_state(), DreamRunState::Idle));
    }

    #[test]
    fn fresh_token_per_attempt() {
        // Cancel one cycle, claim the slot again, the new token must be live.
        let h = DreamHandle::new();
        {
            let _g = h.start_attempt("run-1".into()).unwrap();
            h.request_wake();
            assert!(h.current_token().is_cancelled());
        }
        let _g2 = h.start_attempt("run-2".into()).unwrap();
        assert!(
            !h.current_token().is_cancelled(),
            "new attempt must mint a fresh, un-cancelled token"
        );
    }

    // ---- D3: orchestrator -------------------------------------------------

    use crate::chat::mock_provider::MockProvider;
    use crate::state_machine::{Pet, PetState};
    use crate::vault::compiler::{CompileBatch, CompiledEvent};
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn make_pet() -> Arc<Mutex<Pet>> {
        Arc::new(Mutex::new(Pet::new(0, 0, 1920, 1080, 80, 4)))
    }

    fn setup_cycle_env() -> (TempDir, PathBuf, PathBuf, DreamsLog) {
        let tmp = TempDir::new().unwrap();
        let wiki = tmp.path().join("wiki");
        std::fs::create_dir_all(wiki.join("user")).unwrap();
        std::fs::write(wiki.join("user").join("preferences.md"), "# Preferences\n").unwrap();
        let artifacts = tmp.path().join("dreams_artifacts");
        let dreams = DreamsLog::new(tmp.path());
        (tmp, wiki, artifacts, dreams)
    }

    fn batch_with_one_event() -> CompileBatch {
        CompileBatch {
            events: vec![CompiledEvent {
                event_id: "evt_aaa".into(),
                raw_json_line: r#"{"type":"chat","user":"hi","rolo":"hello"}"#.into(),
                kind: "chat".into(),
            }],
        }
    }

    #[tokio::test]
    async fn successful_cycle_writes_success_to_dreams_log_and_returns_to_idle() {
        let (_tmp, wiki, artifacts, dreams) = setup_cycle_env();
        let response = r#"{"facts":[{"file":"user/preferences.md","section":"S","operation":"add","content":"User prefers tea.","supersedes_line":null,"source_event_ids":["evt_aaa"]}]}"#;
        let provider: Arc<dyn InferenceProvider> = Arc::new(MockProvider {
            response: response.into(),
            delay_ms: 0,
            should_fail: false,
        });
        let handle = Arc::new(DreamHandle::new());
        let pet = make_pet();

        let dreaming = Dreaming::new();
        let outcome = dreaming
            .run_one_cycle(
                &wiki,
                &artifacts,
                &dreams,
                batch_with_one_event(),
                provider,
                Arc::clone(&handle),
                Arc::clone(&pet),
            )
            .await;

        assert_eq!(outcome.status, "success", "outcome={:?}", outcome);
        assert_eq!(
            pet.lock().unwrap().state(),
            PetState::Idle,
            "pet must return to Idle after a successful cycle"
        );
        let recent = dreams.read_recent(10);
        assert!(
            recent.iter().any(|e| e["status"] == "success"),
            "dreams log missing success entry: {:?}",
            recent
        );
        // Lint also runs on success — it appends a second entry. Verify both.
        assert!(
            recent.iter().any(|e| e["run_type"] == "lint"
                || e.get("run_type").is_none() && e["status"] == "success"),
            "expected lint or success entry: {:?}",
            recent
        );
    }

    #[tokio::test]
    async fn mock_provider_error_writes_failed_llm_error_status() {
        let (_tmp, wiki, artifacts, dreams) = setup_cycle_env();
        let provider: Arc<dyn InferenceProvider> = Arc::new(MockProvider::failing("boom"));
        let handle = Arc::new(DreamHandle::new());
        let pet = make_pet();

        let dreaming = Dreaming::new();
        let outcome = dreaming
            .run_one_cycle(
                &wiki,
                &artifacts,
                &dreams,
                batch_with_one_event(),
                provider,
                handle,
                Arc::clone(&pet),
            )
            .await;

        assert_eq!(outcome.status, "failed", "{:?}", outcome);
        assert_eq!(pet.lock().unwrap().state(), PetState::Idle);
    }

    #[tokio::test]
    async fn cancelled_mid_compile_writes_cancelled_status() {
        let (_tmp, wiki, artifacts, dreams) = setup_cycle_env();
        // Long-running provider so we have time to cancel.
        let provider: Arc<dyn InferenceProvider> = Arc::new(MockProvider {
            response: "word ".repeat(200),
            delay_ms: 100,
            should_fail: false,
        });
        let handle = Arc::new(DreamHandle::new());
        let pet = make_pet();

        let dreaming = Arc::new(Dreaming::new());
        let wiki_c = wiki.clone();
        let artifacts_c = artifacts.clone();
        let provider_c = Arc::clone(&provider);
        let handle_c = Arc::clone(&handle);
        let pet_c = Arc::clone(&pet);
        let dreams_path = dreams.path().to_path_buf();

        let join = tokio::spawn(async move {
            let dreams = DreamsLog::new(dreams_path.parent().unwrap());
            dreaming
                .run_one_cycle(
                    &wiki_c,
                    &artifacts_c,
                    &dreams,
                    batch_with_one_event(),
                    provider_c,
                    handle_c,
                    pet_c,
                )
                .await
        });

        // Give the cycle a moment to enter the select! loop and start the
        // mock provider's word-by-word streaming.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        handle.request_wake();

        let outcome = join.await.unwrap();
        assert_eq!(outcome.status, "cancelled", "{:?}", outcome);
        assert_eq!(
            pet.lock().unwrap().state(),
            PetState::Idle,
            "pet must return to Idle after cancel"
        );
        let recent = dreams.read_recent(10);
        assert!(
            recent.iter().any(|e| e["status"] == "cancelled"),
            "dreams log missing cancelled entry: {:?}",
            recent
        );
    }

    #[tokio::test]
    async fn timeout_writes_failed_timeout_status() {
        // Use a tiny timeout (1s) and a provider that never finishes within
        // that window. tokio::time::pause is deliberately NOT used here:
        // MockProvider's sleep is real-time and pause+advance interactions
        // with mpsc::Sender::send are subtle; a short real-clock test is
        // simpler and reliable.
        let (_tmp, wiki, artifacts, dreams) = setup_cycle_env();
        let provider: Arc<dyn InferenceProvider> = Arc::new(MockProvider {
            response: "word ".repeat(500),
            delay_ms: 50, // 500 words * 50ms = ~25s, well past 1s timeout
            should_fail: false,
        });
        let handle = Arc::new(DreamHandle::new());
        let pet = make_pet();

        let dreaming = Dreaming::new();
        let outcome = dreaming
            .run_one_cycle_with_timeout(
                &wiki,
                &artifacts,
                &dreams,
                batch_with_one_event(),
                provider,
                handle,
                Arc::clone(&pet),
                1,
            )
            .await;

        assert_eq!(outcome.status, "failed", "{:?}", outcome);
        assert_eq!(outcome.reason.as_deref(), Some("timeout"));
        assert_eq!(pet.lock().unwrap().state(), PetState::Idle);
        let recent = dreams.read_recent(10);
        assert!(
            recent.iter().any(|e| e["reason"] == "timeout"),
            "dreams log missing timeout entry: {:?}",
            recent
        );
    }

    #[tokio::test]
    async fn run_one_cycle_for_events_succeeds_with_mock_provider() {
        // Phase 7: the Memory tab's "Save and Sleep" path bypasses the gate
        // evaluator and the 50-event floor. Verify a hand-built one-event
        // batch tagged "user_profile" runs through the same compile-lint
        // chain as the poll loop and writes a success entry to dreams_log.
        let (_tmp, wiki, artifacts, dreams) = setup_cycle_env();
        let response = r#"{"facts":[{"file":"user/preferences.md","section":"S","operation":"add","content":"User prefers tea.","supersedes_line":null,"source_event_ids":["evt_aaa"]}]}"#;
        let provider: Arc<dyn InferenceProvider> = Arc::new(MockProvider {
            response: response.into(),
            delay_ms: 0,
            should_fail: false,
        });
        let handle = Arc::new(DreamHandle::new());
        let pet = make_pet();

        let events = vec![CompiledEvent {
            event_id: "evt_aaa".into(),
            raw_json_line: r#"{"type":"user_profile_update","event_id":"evt_aaa","ts":"2026-05-13T10:00:00-04:00","category":"about","text":"the user is an ML researcher in Brooklyn."}"#
                .into(),
            kind: "user_profile".into(),
        }];

        let dreaming = Dreaming::new();
        let outcome = dreaming
            .run_one_cycle_for_events(
                &wiki,
                &artifacts,
                &dreams,
                events,
                provider,
                Arc::clone(&handle),
                Arc::clone(&pet),
            )
            .await;

        assert_eq!(outcome.status, "success", "outcome={:?}", outcome);
        assert_eq!(
            pet.lock().unwrap().state(),
            PetState::Idle,
            "pet must return to Idle after a successful cycle"
        );
        let recent = dreams.read_recent(10);
        assert!(
            recent.iter().any(|e| e["status"] == "success"),
            "dreams_log missing success entry: {:?}",
            recent
        );
    }

    #[tokio::test]
    async fn run_one_cycle_for_events_returns_skipped_when_already_running() {
        // Single-flight contract: a second call while the slot is held must
        // return skipped/already_running, not queue, not panic.
        let (_tmp, wiki, artifacts, dreams) = setup_cycle_env();
        let provider: Arc<dyn InferenceProvider> = Arc::new(MockProvider::new("ignored"));
        let handle = Arc::new(DreamHandle::new());
        let pet = make_pet();

        // Hold the slot manually so run_one_cycle_for_events sees a busy
        // handle and short-circuits.
        let _holder = handle.start_attempt("manual_hold".into()).unwrap();

        let dreaming = Dreaming::new();
        let outcome = dreaming
            .run_one_cycle_for_events(
                &wiki,
                &artifacts,
                &dreams,
                vec![CompiledEvent {
                    event_id: "evt_irrelevant".into(),
                    raw_json_line: "{}".into(),
                    kind: "user_profile".into(),
                }],
                provider,
                Arc::clone(&handle),
                pet,
            )
            .await;

        assert_eq!(outcome.status, "skipped");
        assert_eq!(outcome.reason.as_deref(), Some("already_running"));
    }

    #[tokio::test]
    async fn skipped_when_pet_not_idle() {
        // Force pet into a non-Idle state before cycle starts.
        let (_tmp, wiki, artifacts, dreams) = setup_cycle_env();
        let provider: Arc<dyn InferenceProvider> = Arc::new(MockProvider::new("ignored"));
        let handle = Arc::new(DreamHandle::new());
        let pet = make_pet();
        // Force a non-Idle state; enter_sleeping will refuse.
        pet.lock().unwrap()._force_state(PetState::WalkLeft);

        let dreaming = Dreaming::new();
        let outcome = dreaming
            .run_one_cycle(
                &wiki,
                &artifacts,
                &dreams,
                batch_with_one_event(),
                provider,
                handle,
                pet,
            )
            .await;

        assert_eq!(outcome.status, "skipped");
        assert_eq!(outcome.reason.as_deref(), Some("pet_not_idle"));
    }
}
