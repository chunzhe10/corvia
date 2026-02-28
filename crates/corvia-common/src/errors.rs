use thiserror::Error;

#[derive(Error, Debug)]
pub enum CorviaError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Ingestion error: {0}")]
    Ingestion(String),

    #[error("Docker error: {0}")]
    Docker(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Infrastructure error: {0}")]
    Infra(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Agent error: {0}")]
    Agent(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, CorviaError>;
