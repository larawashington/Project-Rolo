//! Rolo's memory banks — SQLite-backed persistence for chat sessions, messages,
//! and long-term memories.
//!
//! The store uses WAL mode for concurrent reads and versioned migrations so
//! Rolo's memories survive upgrades without data loss.

use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Current schema version. Bump this when adding new migrations.
const SCHEMA_VERSION: i64 = 1;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single message in a chat session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: i64,
    pub session_id: String,
    /// "user" | "assistant" | "system"
    pub role: String,
    pub content: String,
    pub created_at: String,
    pub reported: bool,
}

/// A long-term memory extracted from conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: i64,
    pub content: String,
    pub category: String,
    pub created_at: String,
    pub access_count: i64,
}

/// A chat session — one continuous conversation with Rolo.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub source: String,
    pub trigger_text: String,
    pub summary: Option<String>,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during chat store operations.
#[derive(Debug)]
pub enum StoreError {
    /// SQLite operation failed.
    Sqlite(rusqlite::Error),
    /// Data directory could not be resolved.
    DataDir(String),
    /// IO error (directory creation, etc.).
    Io(std::io::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Sqlite(e) => write!(f, "SQLite error: {}", e),
            StoreError::DataDir(msg) => write!(f, "Data directory error: {}", msg),
            StoreError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StoreError::Sqlite(e) => Some(e),
            StoreError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for StoreError {
    fn from(e: rusqlite::Error) -> Self {
        StoreError::Sqlite(e)
    }
}

impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        StoreError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// ChatStore
// ---------------------------------------------------------------------------

/// SQLite-backed persistence for Rolo's chat sessions and memories.
///
/// Thread-safe via `Arc<Mutex<Connection>>` — matches the project's existing
/// `SharedPet = Arc<Mutex<Pet>>` pattern.
pub struct ChatStore {
    conn: Arc<Mutex<Connection>>,
}

impl ChatStore {
    /// Open (or create) the chat database at Rolo's platform data directory.
    ///
    /// Path: `{data_dir}/com.rolo.desktop-pet/chat.db`
    ///
    /// Enables WAL mode for better concurrent access and runs any pending
    /// schema migrations.
    pub fn open() -> Result<Self, StoreError> {
        let data_dir = dirs::data_dir().ok_or_else(|| {
            StoreError::DataDir(
                "Could not determine platform data directory — Rolo has no home!".to_string(),
            )
        })?;

        let app_dir = data_dir.join("com.rolo.desktop-pet");
        std::fs::create_dir_all(&app_dir)?;

        let db_path = app_dir.join("chat.db");
        log::info!("[Rolo] Opening chat store at {:?}", db_path);

        let conn = Connection::open(&db_path)?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.initialize()?;
        Ok(store)
    }

    /// Create an in-memory chat store — used as a fallback when the filesystem
    /// is unavailable. Rolo's conversations won't persist across restarts, but
    /// at least he can still talk.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        log::warn!("[Rolo] Using in-memory chat store — conversations will not persist!");
        let conn = Connection::open_in_memory()?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.initialize()?;
        Ok(store)
    }

    /// Enable WAL mode and run schema migrations.
    fn initialize(&self) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());

        // WAL mode for concurrent reads — essential for a desktop pet that
        // might be writing messages while the UI reads them.
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;

        // Create the meta table first (used by migration versioning)
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )?;

        let current_version = self.get_schema_version_with_conn(&conn);
        if current_version < SCHEMA_VERSION {
            self.run_migrations(&conn, current_version)?;
        }

        Ok(())
    }

    /// Read the schema version from the meta table. Returns 0 if not yet set.
    fn get_schema_version_with_conn(&self, conn: &Connection) -> i64 {
        conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| {
                let v: String = row.get(0)?;
                Ok(v.parse::<i64>().unwrap_or(0))
            },
        )
        .unwrap_or(0)
    }

    /// Run migrations from `from_version` up to `SCHEMA_VERSION`.
    fn run_migrations(&self, conn: &Connection, from_version: i64) -> Result<(), StoreError> {
        if from_version < 1 {
            log::info!("[Rolo] Running chat store migration v1 — creating tables");

            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS sessions (
                    id           TEXT PRIMARY KEY,
                    started_at   TEXT NOT NULL,
                    ended_at     TEXT,
                    source       TEXT NOT NULL,
                    trigger_text TEXT NOT NULL DEFAULT '',
                    summary      TEXT,
                    mood_reading TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reported   BOOLEAN NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    content           TEXT NOT NULL,
                    category          TEXT NOT NULL,
                    source_session_id TEXT,
                    created_at        TEXT NOT NULL,
                    last_accessed_at  TEXT NOT NULL,
                    access_count      INTEGER NOT NULL DEFAULT 0,
                    active            BOOLEAN NOT NULL DEFAULT 1,
                    FOREIGN KEY (source_session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS mood_readings (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    reading    TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
                CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
                CREATE INDEX IF NOT EXISTS idx_mood_readings_session_id ON mood_readings(session_id);",
            )?;
        }

        // Future migrations would go here:
        // if from_version < 2 { ... }

        // Update schema version
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )?;

        log::info!("[Rolo] Chat store schema up to date (v{})", SCHEMA_VERSION);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Session operations
    // -----------------------------------------------------------------------

    /// Create a new chat session. Returns the session UUID.
    ///
    /// `source` indicates what triggered the conversation (e.g., "user_click",
    /// "idle_prompt", "mood_checkin"). `trigger_text` is the initial context.
    pub fn create_session(&self, source: &str, trigger_text: &str) -> Result<String, StoreError> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        conn.execute(
            "INSERT INTO sessions (id, started_at, source, trigger_text) VALUES (?1, ?2, ?3, ?4)",
            params![id, now, source, trigger_text],
        )?;

        log::info!("[Rolo] Chat session created: {} (source: {})", id, source);
        Ok(id)
    }

    /// Close a chat session, optionally storing a summary.
    pub fn close_session(&self, session_id: &str, summary: Option<&str>) -> Result<(), StoreError> {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());

        conn.execute(
            "UPDATE sessions SET ended_at = ?1, summary = ?2 WHERE id = ?3",
            params![now, summary, session_id],
        )?;

        log::info!("[Rolo] Chat session closed: {}", session_id);
        Ok(())
    }

    /// Get all sessions that were never closed — used for crash recovery.
    /// If Rolo went down mid-conversation, these sessions need to be cleaned up.
    #[allow(dead_code)]
    pub fn get_unclosed_sessions(&self) -> Result<Vec<String>, StoreError> {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        let mut stmt = conn.prepare("SELECT id FROM sessions WHERE ended_at IS NULL")?;

        let ids = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ids)
    }

    /// Get all session IDs, ordered by start time (oldest first).
    ///
    /// Used by the training data exporter to iterate every conversation Rolo
    /// has ever had.
    pub fn get_all_session_ids(&self) -> Result<Vec<String>, StoreError> {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        let mut stmt = conn.prepare("SELECT id FROM sessions ORDER BY started_at ASC")?;

        let ids = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ids)
    }

    // -----------------------------------------------------------------------
    // Message operations
    // -----------------------------------------------------------------------

    /// Insert a message into a session. Returns the message row ID.
    pub fn insert_message(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
    ) -> Result<i64, StoreError> {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());

        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![session_id, role, content, now],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Get all messages for a session, ordered chronologically.
    pub fn get_session_messages(&self, session_id: &str) -> Result<Vec<ChatMessage>, StoreError> {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        let mut stmt = conn.prepare(
            "SELECT id, session_id, role, content, created_at, reported
             FROM messages
             WHERE session_id = ?1
             ORDER BY id ASC",
        )?;

        let messages = stmt
            .query_map(params![session_id], |row| {
                Ok(ChatMessage {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    created_at: row.get(4)?,
                    reported: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(messages)
    }

    /// Mark a message as reported (flagged by the user).
    pub fn mark_message_reported(&self, message_id: i64) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        conn.execute(
            "UPDATE messages SET reported = 1 WHERE id = ?1",
            params![message_id],
        )?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Memory operations
    // -----------------------------------------------------------------------

    /// Insert a long-term memory extracted from a conversation.
    pub fn insert_memory(
        &self,
        content: &str,
        category: &str,
        source_session_id: Option<&str>,
    ) -> Result<i64, StoreError> {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());

        conn.execute(
            "INSERT INTO memories (content, category, source_session_id, created_at, last_accessed_at)
             VALUES (?1, ?2, ?3, ?4, ?4)",
            params![content, category, source_session_id, now],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Get all active memories, ordered by most recently accessed.
    pub fn get_active_memories(&self) -> Result<Vec<Memory>, StoreError> {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        let mut stmt = conn.prepare(
            "SELECT id, content, category, created_at, access_count
             FROM memories
             WHERE active = 1
             ORDER BY last_accessed_at DESC",
        )?;

        let memories = stmt
            .query_map([], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    category: row.get(2)?,
                    created_at: row.get(3)?,
                    access_count: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(memories)
    }

    // -----------------------------------------------------------------------
    // Mood readings
    // -----------------------------------------------------------------------

    pub fn insert_mood_reading(
        &self,
        session_id: &str,
        reading: &str,
        confidence: f64,
    ) -> Result<i64, StoreError> {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        conn.execute(
            "INSERT INTO mood_readings (session_id, reading, confidence, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![session_id, reading, confidence, now],
        )?;
        Ok(conn.last_insert_rowid())
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Clean up old message data while preserving session metadata and memories.
    ///
    /// Messages older than `days` are deleted. Sessions and memories are kept
    /// so Rolo retains his long-term context even as conversation details fade.
    #[allow(dead_code)]
    pub fn cleanup_old_sessions(&self, days: u32) -> Result<u64, StoreError> {
        let cutoff = Utc::now()
            .checked_sub_signed(chrono::Duration::days(i64::from(days)))
            .unwrap_or_else(Utc::now)
            .to_rfc3339();

        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());

        let deleted = conn.execute(
            "DELETE FROM messages WHERE created_at < ?1",
            params![cutoff],
        )?;

        if deleted > 0 {
            log::info!(
                "[Rolo] Cleaned up {} old messages (older than {} days)",
                deleted,
                days
            );
        }

        Ok(deleted as u64)
    }
}

// ---------------------------------------------------------------------------
// Tests — Rolo's memory must be reliable
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a fresh in-memory store for each test.
    fn test_store() -> ChatStore {
        ChatStore::open_in_memory().expect("In-memory store must succeed")
    }

    #[test]
    fn create_and_query_session() {
        let store = test_store();

        let session_id = store
            .create_session("user_click", "Hello Rolo!")
            .expect("create_session must succeed");

        assert!(!session_id.is_empty(), "Session ID must not be empty");

        // UUID format check: 8-4-4-4-12 hex digits
        assert_eq!(session_id.len(), 36, "Session ID must be a valid UUID");
        assert_eq!(
            session_id.chars().filter(|c| *c == '-').count(),
            4,
            "UUID must have 4 hyphens"
        );
    }

    #[test]
    fn insert_and_query_messages() {
        let store = test_store();
        let session_id = store
            .create_session("test", "test session")
            .expect("create_session");

        let msg1_id = store
            .insert_message(&session_id, "user", "Hi Rolo!")
            .expect("insert user message");
        let msg2_id = store
            .insert_message(&session_id, "assistant", "Hello! I'm happy to see you!")
            .expect("insert assistant message");

        assert!(msg1_id > 0);
        assert!(msg2_id > msg1_id, "Message IDs must be sequential");

        let messages = store
            .get_session_messages(&session_id)
            .expect("get_session_messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "Hi Rolo!");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "Hello! I'm happy to see you!");
        assert!(!messages[0].reported);
    }

    #[test]
    fn messages_empty_for_nonexistent_session() {
        let store = test_store();
        let messages = store
            .get_session_messages("nonexistent-uuid")
            .expect("query must not error for empty results");
        assert!(messages.is_empty());
    }

    #[test]
    fn mark_message_reported() {
        let store = test_store();
        let session_id = store.create_session("test", "").expect("create session");
        let msg_id = store
            .insert_message(&session_id, "assistant", "Something weird")
            .expect("insert");

        store
            .mark_message_reported(msg_id)
            .expect("mark_message_reported");

        let messages = store.get_session_messages(&session_id).expect("query");
        assert!(messages[0].reported, "Message must be marked as reported");
    }

    #[test]
    fn memory_crud() {
        let store = test_store();
        let session_id = store.create_session("test", "").expect("create session");

        let mem_id = store
            .insert_memory("User likes cats", "preference", Some(&session_id))
            .expect("insert_memory");

        assert!(mem_id > 0);

        let memories = store.get_active_memories().expect("get_active_memories");
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].content, "User likes cats");
        assert_eq!(memories[0].category, "preference");
        assert_eq!(memories[0].access_count, 0);
    }

    #[test]
    fn memory_without_session() {
        let store = test_store();

        // Memories can be created without a source session
        let mem_id = store
            .insert_memory("General fact", "knowledge", None)
            .expect("insert_memory without session");

        assert!(mem_id > 0);

        let memories = store.get_active_memories().expect("get_active_memories");
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].content, "General fact");
    }

    #[test]
    fn close_session() {
        let store = test_store();
        let session_id = store
            .create_session("test", "hello")
            .expect("create session");

        // Unclosed sessions should include ours
        let unclosed = store.get_unclosed_sessions().expect("get_unclosed");
        assert!(unclosed.contains(&session_id));

        // Close it
        store
            .close_session(&session_id, Some("A nice conversation"))
            .expect("close_session");

        // Now it should not appear in unclosed
        let unclosed = store.get_unclosed_sessions().expect("get_unclosed");
        assert!(
            !unclosed.contains(&session_id),
            "Closed session must not appear in unclosed list"
        );
    }

    #[test]
    fn close_session_without_summary() {
        let store = test_store();
        let session_id = store.create_session("test", "").expect("create session");

        // Close without a summary — Rolo had nothing noteworthy to say
        store
            .close_session(&session_id, None)
            .expect("close_session with None summary");

        let unclosed = store.get_unclosed_sessions().expect("get_unclosed");
        assert!(!unclosed.contains(&session_id));
    }

    #[test]
    fn cleanup_old_sessions_removes_old_messages() {
        let store = test_store();
        let session_id = store.create_session("test", "").expect("create session");

        // Insert a message and then manually backdate it
        store
            .insert_message(&session_id, "user", "ancient message")
            .expect("insert");

        // Backdate the message to 100 days ago
        {
            let conn = store.conn.lock().unwrap();
            let old_date = Utc::now()
                .checked_sub_signed(chrono::Duration::days(100))
                .unwrap()
                .to_rfc3339();
            conn.execute(
                "UPDATE messages SET created_at = ?1 WHERE session_id = ?2",
                params![old_date, session_id],
            )
            .expect("backdate message");
        }

        // Insert a recent message in a different session
        let recent_session = store.create_session("test", "").expect("create session");
        store
            .insert_message(&recent_session, "user", "fresh message")
            .expect("insert");

        // Clean up messages older than 30 days
        let deleted = store.cleanup_old_sessions(30).expect("cleanup");
        assert_eq!(deleted, 1, "Should have deleted exactly 1 old message");

        // Old session's messages should be gone
        let old_messages = store
            .get_session_messages(&session_id)
            .expect("query old session");
        assert!(old_messages.is_empty(), "Old messages should be deleted");

        // Recent session's messages should survive
        let recent_messages = store
            .get_session_messages(&recent_session)
            .expect("query recent session");
        assert_eq!(recent_messages.len(), 1, "Recent messages must survive");
    }

    #[test]
    fn multiple_sessions_and_messages() {
        let store = test_store();

        let s1 = store.create_session("click", "first").expect("s1");
        let s2 = store.create_session("idle", "second").expect("s2");

        store.insert_message(&s1, "user", "msg in s1").expect("m1");
        store.insert_message(&s2, "user", "msg in s2").expect("m2");
        store
            .insert_message(&s2, "assistant", "reply in s2")
            .expect("m3");

        let s1_msgs = store.get_session_messages(&s1).expect("s1 messages");
        let s2_msgs = store.get_session_messages(&s2).expect("s2 messages");

        assert_eq!(s1_msgs.len(), 1);
        assert_eq!(s2_msgs.len(), 2);
        assert_eq!(s2_msgs[0].content, "msg in s2");
        assert_eq!(s2_msgs[1].content, "reply in s2");
    }

    #[test]
    fn schema_version_is_set() {
        let store = test_store();
        let conn = store.conn.lock().unwrap();
        let version = store.get_schema_version_with_conn(&conn);
        assert_eq!(
            version, SCHEMA_VERSION,
            "Schema version must match SCHEMA_VERSION"
        );
    }

    #[test]
    fn idempotent_initialization() {
        // Opening the same in-memory store twice (simulated by re-initializing)
        // must not error — migrations are idempotent.
        let store = test_store();
        store
            .initialize()
            .expect("Re-initialization must be idempotent");
    }
}
