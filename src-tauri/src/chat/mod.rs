//! Rolo's chat infrastructure — the neural pathways that will let him converse.
//!
//! This module provides:
//! - `store` — SQLite-backed persistence for sessions, messages, and memories
//! - `provider` — trait abstraction for LLM inference backends
//! - `mock_provider` — deterministic provider for testing
//! - `openai_provider` — OpenAI-compatible HTTP streaming client

pub mod anthropic_provider;
pub mod config;
pub mod context;
pub mod engine;
pub mod export;
pub mod gemini_provider;
pub mod memory;
pub mod mock_provider;
pub mod openai_provider;
pub mod prompt;
pub mod provider;
pub mod safety;
pub mod store;
