// Should export Neural network model

pub mod arch;
pub mod model4;

pub use crate::model::arch::{UrlClassifier, UrlClassifierConfig};
pub use crate::model::model4::{UrlClassifierModel4, UrlClassifierModel4Config};
