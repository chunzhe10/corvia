//! In-process kernel event bus.
//!
//! Provides [`KernelEvent`] and [`EventBus`] for publish/subscribe within the
//! kernel. Events are informational — they layer on top of existing direct calls
//! and are not load-bearing. If a consumer lags or drops, nothing breaks.
//!
//! # Design Decisions
//!
//! - **D1 (Hybrid):** Events supplement, never replace, direct method calls.
//! - **D2 (Lean variants):** Events carry IDs only; consumers look up details.
//! - **D3 (`tokio::broadcast`):** Multi-consumer, drop-tolerant, no external deps.
//! - **D4 (Publish after success):** `publish()` called only after the direct call
//!   returns `Ok`, preventing phantom events for failed operations.
//! - **D5 (`CancellationToken`):** Shared shutdown signal for producers and consumers.

use std::path::PathBuf;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// KernelEvent
// ---------------------------------------------------------------------------

/// Kernel-level state transition events.
///
/// Variants carry IDs only (not full data) to keep broadcast cloning cheap.
/// Consumers that need details look them up by ID.
///
/// `#[non_exhaustive]` so new variants are additive and consumers must use
/// wildcard arms.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum KernelEvent {
    MergeCompleted {
        entry_id: Uuid,
        session_id: String,
        scope_id: String,
    },
    MergeFailed {
        entry_id: Uuid,
        scope_id: String,
    },
    EntryCommitted {
        entry_id: Uuid,
        session_id: String,
        scope_id: String,
    },
    SessionOpened {
        agent_id: String,
        session_id: String,
        scope_id: String,
    },
    SessionClosed {
        session_id: String,
        scope_id: String,
    },
    GcCompleted {
        scope_id: String,
        entries_removed: usize,
    },
    IngestCompleted {
        scope_id: String,
        entries_added: usize,
        duration_ms: u64,
    },
}

// ---------------------------------------------------------------------------
// EventBus
// ---------------------------------------------------------------------------

/// In-process event bus backed by `tokio::sync::broadcast`.
///
/// Owned by [`AgentCoordinator`](crate::agent_coordinator::AgentCoordinator),
/// passed as `Arc<EventBus>` to subsystems.
pub struct EventBus {
    tx: broadcast::Sender<KernelEvent>,
    cancel: CancellationToken,
}

impl EventBus {
    /// Create a new event bus with the given channel capacity.
    ///
    /// Default capacity is 256 (see `AgentLifecycleConfig::event_bus_capacity`).
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity.max(1));
        Self {
            tx,
            cancel: CancellationToken::new(),
        }
    }

    /// Publish an event to all subscribers.
    ///
    /// Never panics. Logs at debug level on every publish. Ignores `SendError`
    /// (no receivers) — fire and forget.
    pub fn publish(&self, event: KernelEvent) {
        debug!(event = ?event, "event_bus.publish");
        // SendError means no receivers — that's fine.
        let _ = self.tx.send(event);
    }

    /// Subscribe to the event bus. Returns a new receiver.
    pub fn subscribe(&self) -> broadcast::Receiver<KernelEvent> {
        self.tx.subscribe()
    }

    /// Get a clone of the cancellation token for shutdown coordination.
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Convenience constructor for tests: returns bus + pre-subscribed receiver.
    #[cfg(test)]
    pub fn test() -> (Self, broadcast::Receiver<KernelEvent>) {
        let bus = Self::new(64);
        let rx = bus.subscribe();
        (bus, rx)
    }
}

// ---------------------------------------------------------------------------
// EventLogger (JSONL consumer)
// ---------------------------------------------------------------------------

/// Subscribes to the event bus and writes events to `events.jsonl` via
/// [`corvia_common::events::append_event`].
///
/// Handles `RecvError::Lagged` by logging a warning and continuing.
/// Handles `RecvError::Closed` by exiting the loop cleanly.
pub struct EventLogger;

impl EventLogger {
    /// Spawn the event logger as a background tokio task.
    ///
    /// The task runs until the broadcast channel is closed or the cancellation
    /// token is triggered.
    pub fn spawn(event_bus: &EventBus, events_path: PathBuf) -> tokio::task::JoinHandle<()> {
        let mut rx = event_bus.subscribe();
        let cancel = event_bus.cancel_token();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        debug!("event_logger: shutdown via cancellation token");
                        break;
                    }
                    result = rx.recv() => {
                        match result {
                            Ok(event) => {
                                Self::log_event(&events_path, &event);
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                warn!(lagged = n, "event_bus.consumer_lagged");
                                // Continue receiving — do NOT attempt to replay.
                            }
                            Err(broadcast::error::RecvError::Closed) => {
                                debug!("event_logger: channel closed, exiting");
                                break;
                            }
                        }
                    }
                }
            }
        })
    }

    /// Map a KernelEvent to a corvia_common Event and append to JSONL.
    fn log_event(path: &std::path::Path, event: &KernelEvent) {
        use corvia_common::events::{append_event, Event, EventType};

        let (event_type, data) = match event {
            KernelEvent::MergeCompleted { entry_id, session_id, scope_id } => (
                EventType::MergeCompleted,
                serde_json::json!({
                    "entry_id": entry_id.to_string(),
                    "session_id": session_id,
                    "scope_id": scope_id,
                }),
            ),
            KernelEvent::MergeFailed { entry_id, scope_id } => (
                EventType::MergeFailed,
                serde_json::json!({
                    "entry_id": entry_id.to_string(),
                    "scope_id": scope_id,
                }),
            ),
            KernelEvent::EntryCommitted { entry_id, session_id, scope_id } => (
                EventType::EntryCommitted,
                serde_json::json!({
                    "entry_id": entry_id.to_string(),
                    "session_id": session_id,
                    "scope_id": scope_id,
                }),
            ),
            KernelEvent::SessionOpened { agent_id, session_id, scope_id } => (
                EventType::SessionOpened,
                serde_json::json!({
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "scope_id": scope_id,
                }),
            ),
            KernelEvent::SessionClosed { session_id, scope_id } => (
                EventType::SessionClosed,
                serde_json::json!({
                    "session_id": session_id,
                    "scope_id": scope_id,
                }),
            ),
            KernelEvent::GcCompleted { scope_id, entries_removed } => (
                EventType::GcCompleted,
                serde_json::json!({
                    "scope_id": scope_id,
                    "entries_removed": entries_removed,
                }),
            ),
            KernelEvent::IngestCompleted { scope_id, entries_added, duration_ms } => (
                EventType::IngestionCompleted,
                serde_json::json!({
                    "scope_id": scope_id,
                    "entries_added": entries_added,
                    "duration_ms": duration_ms,
                }),
            ),
            // Non-exhaustive: future variants logged as unhandled.
            // Within this crate the compiler sees all variants; the arm is needed
            // for when new variants are added (non_exhaustive guarantee).
            #[allow(unreachable_patterns)]
            _ => {
                debug!(event = ?event, "event_logger: unhandled event variant");
                return;
            }
        };

        let common_event = Event::new(event_type, data);
        if let Err(e) = append_event(path, &common_event) {
            warn!(error = %e, "event_logger: failed to append event");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_publish_and_receive() {
        let (bus, mut rx) = EventBus::test();

        bus.publish(KernelEvent::MergeCompleted {
            entry_id: Uuid::nil(),
            session_id: "sess-1".into(),
            scope_id: "test".into(),
        });

        let event = rx.recv().await.unwrap();
        match event {
            KernelEvent::MergeCompleted { entry_id, session_id, scope_id } => {
                assert_eq!(entry_id, Uuid::nil());
                assert_eq!(session_id, "sess-1");
                assert_eq!(scope_id, "test");
            }
            _ => panic!("unexpected event variant"),
        }
    }

    #[tokio::test]
    async fn test_publish_with_no_receivers() {
        // Should not panic even with no subscribers.
        let bus = EventBus::new(16);
        bus.publish(KernelEvent::GcCompleted {
            scope_id: "test".into(),
            entries_removed: 5,
        });
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let bus = EventBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        bus.publish(KernelEvent::SessionClosed {
            session_id: "sess-1".into(),
            scope_id: "test".into(),
        });

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert!(matches!(e1, KernelEvent::SessionClosed { .. }));
        assert!(matches!(e2, KernelEvent::SessionClosed { .. }));
    }

    #[tokio::test]
    async fn test_cancellation_token() {
        let bus = EventBus::new(16);
        let token = bus.cancel_token();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn test_event_logger_writes_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let events_path = dir.path().join("events.jsonl");

        let bus = EventBus::new(16);
        let handle = EventLogger::spawn(&bus, events_path.clone());

        // Publish a few events
        bus.publish(KernelEvent::MergeCompleted {
            entry_id: Uuid::nil(),
            session_id: "sess-1".into(),
            scope_id: "test".into(),
        });
        bus.publish(KernelEvent::EntryCommitted {
            entry_id: Uuid::nil(),
            session_id: "sess-1".into(),
            scope_id: "test".into(),
        });

        // Give the logger a moment to process
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Cancel and wait for shutdown
        bus.cancel_token().cancel();
        handle.await.unwrap();

        // Verify JSONL was written
        let content = std::fs::read_to_string(&events_path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("MergeCompleted"));
        assert!(lines[1].contains("EntryCommitted"));
    }
}
