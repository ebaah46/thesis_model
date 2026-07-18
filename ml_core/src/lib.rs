pub mod data;
pub mod model;
pub mod training;

use std::path::PathBuf;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};

use burn::store::{ModuleSnapshot, SafetensorsStore};
use data::Dataset;
use training::{TrainingConfig, train};

fn test_csv() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("data");
    p.push("balanced_urls.csv");
    p
}

pub fn run_training() {
    type TrainBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    let config = TrainingConfig {
        num_epochs: 1,
        batch_size: 128,
        ..Default::default()
    };
    println!("{:?}", test_csv());
    let dataset = Dataset::load_dataset_csv(test_csv()).expect("load dataset");
    let artifact_dir = std::env::temp_dir().join("thesis_model_train_smoke");
    let mut model_file = artifact_dir.clone();
    model_file.push("model4_1.safetensors");
    println!("{:?}", artifact_dir);
    let trained = train::<TrainBackend>(&config, dataset, &device, &artifact_dir)
        .expect("training should complete");
    let mut store = SafetensorsStore::from_file(model_file).overwrite(true);
    trained
        .clone()
        .save_into(&mut store)
        .expect("model should be saved");
}
