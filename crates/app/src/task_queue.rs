//! # `app::task_queue`
//!
//! Universal Background Task Queue Service.
//!
//! Handles asynchronous execution, progress tracking, pausing, resuming, cancelling,
//! and user confirmation checkpoints for:
//! 1. Storage Asset Preloading ("Preload All Assets")
//! 2. Data Contribution uploads (multi-author submission to R2/D1)
//! 3. Offline Lossless/Lossy Audio Export (FLAC-8, Opus, MP3 with ID3v2.4)

use serde::{Deserialize, Serialize};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// Lifecycle state of an enqueued task item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskStatus {
    Queued,
    Running {
        progress: f32,
        speed_bps: f32,
        status_text: String,
    },
    Paused {
        progress: f32,
        status_text: String,
    },
    AwaitingConfirmation {
        message: String,
    },
    Done {
        result_message: String,
    },
    Failed {
        error_message: String,
    },
    Cancelled,
}

impl TaskStatus {
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Queued
                | Self::Running { .. }
                | Self::Paused { .. }
                | Self::AwaitingConfirmation { .. }
        )
    }

    pub fn progress(&self) -> f32 {
        match self {
            Self::Queued => 0.0,
            Self::Running { progress, .. } | Self::Paused { progress, .. } => *progress,
            Self::AwaitingConfirmation { .. } => 0.95,
            Self::Done { .. } => 1.0,
            Self::Failed { .. } | Self::Cancelled => 0.0,
        }
    }
}

/// Category / Kind of task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskKind {
    AssetPreload { asset_id: String },
    DataContribution { title: String },
    AudioExport { format: String },
}

/// An individual task entry in the queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskItem {
    pub id: String,
    pub title: String,
    pub kind: TaskKind,
    pub status: TaskStatus,
    pub total_bytes: u64,
    pub processed_bytes: u64,
    pub created_at_unix: u64,
}

impl TaskItem {
    pub fn new(id: impl Into<String>, title: impl Into<String>, kind: TaskKind) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            kind,
            status: TaskStatus::Queued,
            total_bytes: 0,
            processed_bytes: 0,
            created_at_unix: 0,
        }
    }
}

/// Universal Thread-safe Task Queue.
#[derive(Clone, Debug, Default)]
pub struct TaskQueue {
    tasks: Arc<Mutex<Vec<TaskItem>>>,
    paused_flags: Arc<Mutex<std::collections::HashMap<String, Arc<AtomicBool>>>>,
}

impl TaskQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a new background task.
    pub fn enqueue(&self, task: TaskItem) {
        let mut tasks = self.tasks.lock().unwrap();
        // Prevent duplicate enqueuing of the same task ID
        if !tasks.iter().any(|t| t.id == task.id) {
            tasks.push(task);
        }
    }

    /// Number of incomplete/active tasks in queue.
    pub fn active_count(&self) -> usize {
        let tasks = self.tasks.lock().unwrap();
        tasks.iter().filter(|t| t.status.is_active()).count()
    }

    /// Snapshot of all tasks.
    pub fn snapshot(&self) -> Vec<TaskItem> {
        let tasks = self.tasks.lock().unwrap();
        tasks.clone()
    }

    /// List of all tasks (alias to snapshot).
    pub fn list(&self) -> Vec<TaskItem> {
        self.snapshot()
    }

    /// Update status of a task by ID.
    pub fn update_status(&self, id: &str, status: TaskStatus) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.iter_mut().find(|t| t.id == id) {
            task.status = status;
        }
    }

    /// Update progress of a task.
    pub fn update_progress(
        &self,
        id: &str,
        progress: f32,
        processed_bytes: u64,
        total_bytes: u64,
        status_text: String,
    ) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.iter_mut().find(|t| t.id == id) {
            task.processed_bytes = processed_bytes;
            task.total_bytes = total_bytes;
            task.status = TaskStatus::Running {
                progress: progress.clamp(0.0, 1.0),
                speed_bps: 0.0,
                status_text,
            };
        }
    }

    /// Pause task by ID.
    pub fn pause(&self, id: &str) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.iter_mut().find(|t| t.id == id) {
            let cur_prog = task.status.progress();
            task.status = TaskStatus::Paused {
                progress: cur_prog,
                status_text: "Paused by user".to_string(),
            };
        }
        let flags = self.paused_flags.lock().unwrap();
        if let Some(flag) = flags.get(id) {
            flag.store(true, Ordering::SeqCst);
        }
    }

    /// Resume task by ID.
    pub fn resume(&self, id: &str) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.iter_mut().find(|t| t.id == id) {
            let cur_prog = task.status.progress();
            task.status = TaskStatus::Running {
                progress: cur_prog,
                speed_bps: 0.0,
                status_text: "Resuming...".to_string(),
            };
        }
        let flags = self.paused_flags.lock().unwrap();
        if let Some(flag) = flags.get(id) {
            flag.store(false, Ordering::SeqCst);
        }
    }

    /// Cancel task by ID.
    pub fn cancel(&self, id: &str) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.iter_mut().find(|t| t.id == id) {
            task.status = TaskStatus::Cancelled;
        }
    }

    /// Confirm task waiting in AwaitingConfirmation.
    pub fn confirm(&self, id: &str) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.iter_mut().find(|t| t.id == id) {
            if matches!(task.status, TaskStatus::AwaitingConfirmation { .. }) {
                task.status = TaskStatus::Running {
                    progress: 0.95,
                    speed_bps: 0.0,
                    status_text: "Finalizing submission...".to_string(),
                };
            }
        }
    }

    /// Confirm and proceed alias for confirmation dialogs.
    pub fn confirm_and_proceed(&self, id: &str) {
        self.confirm(id);
    }

    /// Clear completed, cancelled, or failed tasks.
    pub fn clear_finished(&self) {
        let mut tasks = self.tasks.lock().unwrap();
        tasks.retain(|t| t.status.is_active());
    }
}
