// responsible for defining data transformation from raw dataset items to
// tensors that will be inputs for the model

use burn::{
    Tensor,
    data::dataloader::batcher::Batcher as BurnBatcher,
    prelude::*,
    tensor::{Int, backend::Backend},
};

use crate::data::RawItem;

// Training batch that defines how each dataset item provided
// to the model will be represented
#[derive(Debug, Clone)]
pub struct TrainingBatch<B: Backend> {
    // input refers to the features, and it is a 2D matrix where each row
    // represents a RawItem and B represents the batch size selected.
    pub inputs: Tensor<B, 2>,

    // target refers to the labels on the features, and it is a 1D vector
    // indicating the individual labels are of type Int
    pub targets: Tensor<B, 1, Int>,
}

// This is the custom batcher that holds instructions on how to
// translate a RawItem to a TrainingBatch element on the device(GPU)
// Holds access to the specific device
#[derive(Debug, Clone)]
pub struct Batcher<B: Backend> {
    pub device: B::Device,
}

impl<B: Backend> Batcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device: device }
    }
}
impl<B: Backend> BurnBatcher<B, RawItem, TrainingBatch<B>> for Batcher<B> {
    fn batch(&self, items: Vec<RawItem>, device: &B::Device) -> TrainingBatch<B> {
        let (inputs, target): (Vec<Tensor<B, 2>>, Vec<Tensor<B, 1, Int>>) = items
            .iter()
            .map(|elem| {
                (
                    Tensor::<B, 1>::from_floats(elem.url.as_slice(), &device).unsqueeze(),
                    Tensor::<B, 1, Int>::from_ints([elem.is_malicious as i32], &device),
                )
            })
            .unzip();

        TrainingBatch {
            inputs: Tensor::cat(inputs, 0),
            targets: Tensor::cat(target, 0),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::data::Dataset;
    use burn::backend::NdArray;
    use std::path::PathBuf;

    type TestBackend = NdArray<f32>;

    struct TestContext<B: Backend> {
        pub data: Dataset,
        pub device: B::Device,
    }

    impl<B: Backend> TestContext<B> {
        fn new() -> Self {
            // test data path
            let mut data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            data_path.push("data");
            data_path.push("test.csv");

            let dataset = Dataset::load_dataset_csv(data_path)
                .expect("ml_core::data::batcher::TestContext - Unable to load dataset");
            let device = Default::default();
            Self {
                data: dataset,
                device,
            }
        }
    }

    #[test]
    fn test_batcher_construction() {
        let ctx = TestContext::<TestBackend>::new();

        let batcher: Batcher<TestBackend> = Batcher::new(ctx.device);

        let batch = batcher.batch(ctx.data.records, &ctx.device);

        // confirm the shape of each input tensor
        assert_eq!(batch.inputs.shape().dims(), [8, 8415]);

        // confirm the shape of each label tensor
        assert_eq!(batch.targets.shape().dims(), [8]);
    }
}
