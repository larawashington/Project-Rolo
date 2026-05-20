//! `SharedProviderSlot` — an atomically-swappable holder for the active
//! `InferenceProvider` Rolo is currently using as his brain.
//!
//! PRD/rolo-command-center.md — "Brain panel → Hot-reload":
//! the Command Center's Brain tab lets the user re-pick Rolo's provider at
//! runtime. Without a swap mechanism we'd have to restart the whole app to
//! change brains, which would lose every in-flight conversation and dream.
//! Instead, `ChatEngine` and the dreaming poll loop both read their provider
//! through this slot; `cc_apply_brain` (Phase 5) calls `replace(new)` to
//! hot-swap. In-flight inferences keep an owned `Arc<dyn InferenceProvider>`
//! snapshot taken via `snapshot()`, so the old provider survives until those
//! calls drop their references — no torn requests, no panics.

use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::chat::provider::InferenceProvider;

/// Sized newtype around `Arc<dyn InferenceProvider>` so arc-swap can store it
/// atomically.
///
/// `ArcSwap<T>` ultimately uses `AtomicPtr<T>` under the hood, and
/// `AtomicPtr<T>` requires `T: Sized`. Trait objects are unsized, so we wrap
/// the fat `Arc<dyn …>` pointer inside a regular struct (one word: just the
/// `Arc`). `ArcSwap<ProviderRef>` now stores a thin pointer to this sized
/// wrapper, which is what the atomic primitive needs.
///
/// The double-`Arc` indirection costs one extra heap allocation per swap and
/// one extra dereference per snapshot — both negligible compared to the
/// inference call that follows.
pub struct ProviderRef(pub Arc<dyn InferenceProvider>);

/// Atomically-swappable holder for the active inference provider.
///
/// Both the chat engine and the dreaming poll loop read through this slot.
/// `cc_apply_brain` (Phase 5) calls `replace(new_provider)` to hot-swap.
/// In-flight inferences keep a local `Arc<dyn InferenceProvider>` snapshot
/// taken via `snapshot()`, so the old provider survives until those calls
/// drop their references.
#[derive(Clone)]
pub struct SharedProviderSlot {
    inner: Arc<ArcSwap<ProviderRef>>,
}

impl SharedProviderSlot {
    /// Build a new slot, seeded with the given provider.
    pub fn new(initial: Arc<dyn InferenceProvider>) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(ProviderRef(initial))),
        }
    }

    /// Take an owned `Arc` snapshot of the current provider. Safe to call
    /// from any thread; the returned `Arc` keeps the provider alive even if
    /// the slot is swapped while the caller is using it.
    ///
    /// We prefer `load_full()` over `load()` because `load()` returns a
    /// `Guard` whose borrow-like lifetime makes it awkward to hold across
    /// `.await` points. `load_full()` returns a plain `Arc`, which the
    /// caller can then unwrap to the inner provider `Arc`.
    pub fn snapshot(&self) -> Arc<dyn InferenceProvider> {
        // Two-step: pull the sized wrapper out of the swap, then clone the
        // inner provider Arc out of it. Both are cheap atomic ops.
        let wrapper: Arc<ProviderRef> = self.inner.load_full();
        Arc::clone(&wrapper.0)
    }

    /// Replace the active provider. Returns the previous `Arc<dyn …>`, which
    /// the caller is free to drop or keep. The swap is atomic — concurrent
    /// `snapshot()` calls either see the old or the new provider, never a
    /// partial state.
    pub fn replace(&self, new_provider: Arc<dyn InferenceProvider>) -> Arc<dyn InferenceProvider> {
        let prev: Arc<ProviderRef> = self.inner.swap(Arc::new(ProviderRef(new_provider)));
        // Pull the inner Arc<dyn …> out so callers don't need to know about
        // the ProviderRef wrapper.
        Arc::clone(&prev.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::mock_provider::MockProvider;

    fn mock(response: &str) -> Arc<dyn InferenceProvider> {
        Arc::new(MockProvider::new(response))
    }

    #[test]
    fn snapshot_returns_current() {
        let initial = mock("first");
        let slot = SharedProviderSlot::new(Arc::clone(&initial));
        let snap = slot.snapshot();
        assert!(
            Arc::ptr_eq(&snap, &initial),
            "snapshot must return the same Arc that was installed"
        );
        assert_eq!(snap.provider_name(), "Mock");
    }

    #[test]
    fn replace_swaps_atomically() {
        let initial = mock("first");
        let slot = SharedProviderSlot::new(Arc::clone(&initial));

        let new_provider = mock("second");
        let old = slot.replace(Arc::clone(&new_provider));

        assert!(
            Arc::ptr_eq(&old, &initial),
            "replace must return the previous provider Arc"
        );
        let after = slot.snapshot();
        assert!(
            Arc::ptr_eq(&after, &new_provider),
            "subsequent snapshot must return the new provider"
        );
    }

    #[test]
    fn snapshot_outlives_replace() {
        // The whole point of the slot — an in-flight inference that grabbed a
        // snapshot before a swap must keep working against the old provider.
        let initial = mock("first");
        let slot = SharedProviderSlot::new(Arc::clone(&initial));

        // Caller takes a snapshot first…
        let in_flight = slot.snapshot();

        // …then a swap happens.
        let new_provider = mock("second");
        let _ = slot.replace(Arc::clone(&new_provider));

        // The snapshot must still point at the original provider Arc.
        assert!(
            Arc::ptr_eq(&in_flight, &initial),
            "an outstanding snapshot must survive a swap unchanged"
        );
        // And the slot must now hand out the new provider on fresh snapshots.
        let fresh = slot.snapshot();
        assert!(
            Arc::ptr_eq(&fresh, &new_provider),
            "fresh snapshot after swap must return the new provider"
        );
    }
}
