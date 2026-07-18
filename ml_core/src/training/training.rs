// Basic supervised training driver for URL classifier models.
//
// Implements the Burn `TrainStep` / `InferenceStep` traits for `UrlClassifierModel4`, then wires
// the model into `SupervisedTraining` with an Adam optimizer, a manually-computed binary
// cross-entropy loss, and Loss/Accuracy/AUROC metrics.

use std::path::Path;

use burn::prelude::Int;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::{MetricEarlyStoppingStrategy, StoppingCondition};
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    tensor::Tensor,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, AurocMetric, LossMetric},
    },
};

use crate::data::batcher::Batcher;
use crate::data::{Dataset, TrainingBatch};
use crate::model::{
    UrlClassifier, UrlClassifierConfig, UrlClassifierModel4, UrlClassifierModel4Config,
};

use burn::record::CompactRecorder;

/// Configuration for a supervised training run of a URL classifier.
#[derive(Clone)]
pub struct TrainingConfig {
    /// Model architecture configuration.
    pub model: UrlClassifierModel4Config,
    /// Optimizer configuration.
    pub optimizer: AdamConfig,
    /// Learning rate used by the optimizer each step.
    pub learning_rate: f64,
    /// Number of epochs to train for.
    pub num_epochs: usize,
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Seed used to shuffle the training set each epoch.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: UrlClassifierModel4Config::default(),
            optimizer: AdamConfig::new(),
            learning_rate: 1e-3,
            num_epochs: 10,
            batch_size: 8,
            seed: 42,
        }
    }
}

/// Binary cross-entropy between probabilities `p` (shape `[batch, 1]`) and integer targets `y`
/// (shape `[batch]` holding 0/1). Returns the mean loss as a shape `[1]` tensor.
fn binary_cross_entropy<B: Backend>(p: &Tensor<B, 2>, y: &Tensor<B, 1, Int>) -> Tensor<B, 1> {
    let yf = y.clone().float().unsqueeze_dim(1); // [batch, 1]
    let eps = 1e-7;
    let pos = yf.clone().mul(p.clone().clamp_min(eps).log());
    let neg = (yf - 1.0).mul(Tensor::ones_like(p).sub(p.clone()).clamp_min(eps).log());
    pos.add(neg).neg().mean().unsqueeze() // [1]
}

/// Forward pass + loss + metric output shared by the train and inference steps.
fn forward_with_loss<B: Backend>(
    model: &UrlClassifierModel4<B>,
    item: &TrainingBatch<B>,
) -> (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1, Int>) {
    let p = model.forward(item.inputs.clone()); // [batch, 1]
    let targets = item.targets.clone();
    let loss = binary_cross_entropy(&p, &targets);
    // Metrics expect a 2-column `[1-p, p]` view of the binary probabilities.
    // The loss above is always the real BCE computed on the raw sigmoid output.
    let output = Tensor::cat(vec![(1.0 - p.clone()), p], 1); // [batch, 2]
    (loss, output, targets)
}

impl<B: AutodiffBackend> TrainStep for UrlClassifierModel4<B> {
    type Input = TrainingBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let (loss, output, targets) = forward_with_loss(self, &item);
        let grads = loss.backward();
        let loss = loss.detach();
        let output = output.detach();
        let item = ClassificationOutput::new(loss, output, targets);
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for UrlClassifierModel4<B> {
    type Input = TrainingBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TrainingBatch<B>) -> ClassificationOutput<B> {
        let (loss, output, targets) = forward_with_loss(self, &item);
        ClassificationOutput::new(loss, output, targets)
    }
}

/// Train a [`UrlClassifierModel4`] and return the inference-ready model on the inner (non-autodiff)
/// backend.
///
/// # Arguments
///
/// * `config` - Training hyper-parameters.
/// * `dataset` - Pre-encoded dataset (see `crate::data::Dataset`).
/// * `device` - Device the autodiff backend runs on.
/// * `artifact_dir` - Directory used for training logs / checkpoints. Must be creatable.
///
/// # Errors
///
/// Returns an error if `artifact_dir` cannot be created.
pub fn train<B>(
    config: &TrainingConfig,
    dataset: Dataset,
    device: &B::Device,
    artifact_dir: &Path,
) -> anyhow::Result<UrlClassifierModel4<B::InnerBackend>>
where
    B: AutodiffBackend,
{
    std::fs::create_dir_all(artifact_dir)?;

    let model: UrlClassifierModel4<B> = config.model.init(device);
    let optim = config.optimizer.init::<B, UrlClassifierModel4<B>>();

    let batcher_train = Batcher::<B>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(0)
        .set_device(device.clone())
        .build(dataset.clone());

    let batcher_valid = Batcher::<B::InnerBackend>::new(device.clone());
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(0)
        .set_device(device.clone())
        .build(dataset);

    let learner = Learner::new(model, optim, config.learning_rate);
    let loss = LossMetric::<B::InnerBackend>::new();
    let early_stopping: MetricEarlyStoppingStrategy = MetricEarlyStoppingStrategy::new(
        &loss,
        Aggregate::Mean,   // Track the mean value over the epoch
        Direction::Lowest, // For Loss, lower is better (use Highest for Accuracy)
        Split::Valid,      //tor the validation data split
        StoppingCondition::NoImprovementSince {
            n_epochs: 3, // "Patience": stop if no improvement for 3 epochs
        },
    );

    // 1. Initialize your recorder and explicitly tell it to overwrite files
    let recorder = CompactRecorder::new();
    let result = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .early_stopping(early_stopping)
        .num_epochs(config.num_epochs)
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train(AccuracyMetric::new())
        .metric_valid(AccuracyMetric::new())
        .metric_train(AurocMetric::new())
        .metric_valid(AurocMetric::new())
        .with_file_checkpointer(recorder)
        .launch(learner);

    Ok(result.model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::Tensor;
    use std::path::PathBuf;

    type TrainBackend = Autodiff<NdArray<f32>>;

    fn test_csv() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("data");
        p.push("test.csv");
        p
    }

    #[test]
    fn test_training_runs_and_emits_inference_model() {
        let device = Default::default();
        let config = TrainingConfig {
            num_epochs: 1,
            batch_size: 8,
            ..Default::default()
        };
        let dataset = Dataset::load_dataset_csv(test_csv()).expect("load dataset");
        let artifact_dir = std::env::temp_dir().join("thesis_model_train_smoke");

        let trained = train::<TrainBackend>(&config, dataset, &device, &artifact_dir)
            .expect("training should complete");

        // The returned model runs on the inner (NdArray) backend.
        let input = Tensor::<NdArray<f32>, 2>::zeros([3, config.model.input_dim], &device);
        let out = trained.forward(input);
        assert_eq!(out.shape().dims(), [3, config.model.output_dim]);
    }
}
