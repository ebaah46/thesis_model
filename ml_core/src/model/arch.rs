// Should define architecture of Neural network model described here

use burn::{
    module::Module,
    nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig},
    tensor::{activation, backend::Backend, Tensor},
};

/// Trait for all URL classifier model architectures.
///
/// Implement this trait on your own model structs to make them swappable
/// in the training pipeline. Any model implementing this trait can be used
/// interchangeably wherever `UrlClassifier` is expected.
///
/// # Example
///
/// ```ignore
/// #[derive(Module, Debug)]
/// pub struct MyModel<B: Backend> {
///     layer1: Linear<B>,
///     output: Linear<B>,
/// }
///
/// impl<B: Backend> UrlClassifier<B> for MyModel<B> {
///     fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
///         let x = self.layer1.forward(input).relu();
///         self.output.forward(x)
///     }
///     fn input_dim(&self) -> usize { 8415 }
///     fn output_dim(&self) -> usize { 2 }
/// }
/// ```
pub trait UrlClassifier<B: Backend>: Module<B> {
    /// Forward pass through the model.
    ///
    /// - `input`: Tensor of shape `[batch_size, feature_dim]`
    /// - Returns: Tensor of shape `[batch_size, num_classes]`
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Expected input dimension (size of the feature vector).
    fn input_dim(&self) -> usize;

    /// Output dimension (number of classes, typically 2 for binary).
    fn output_dim(&self) -> usize;
}

/// Trait for configs that can initialize a URL classifier model.
///
/// Implement this on your own config structs to define how your model
/// is created and parameterized. This enables swappable model architectures
/// controlled by configuration.
///
/// # Example
///
/// ```ignore
/// #[derive(Clone)]
/// pub struct MyModelConfig {
///     pub input_dim: usize,
///     pub hidden_dim: usize,
///     pub output_dim: usize,
/// }
///
/// impl<B: Backend> UrlClassifierConfig<B> for MyModelConfig {
///     type Model = MyModel<B>;
///     fn init(&self, device: &B::Device) -> Self::Model {
///         MyModel {
///             layer1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
///             output: LinearConfig::new(self.hidden_dim, self.output_dim).init(device),
///         }
///     }
/// }
/// ```
pub trait UrlClassifierConfig<B: Backend>: Send + Clone {
    /// The model type this config initializes.
    type Model: UrlClassifier<B>;

    /// Initialize the model on the given device.
    fn init(&self, device: &B::Device) -> Self::Model;
}

// ---------------------------------------------------------------------------
// Reference implementation: UrlClassifierModel4
// ---------------------------------------------------------------------------

/// A reference model with 4 hidden layers and batch normalization.
///
/// Architecture:
///   Linear(8415, 2048) → ReLU → Linear(2048, 1024) → BN → ReLU
///   → Linear(1024, 512) → BN → ReLU → Linear(512, 256) → ReLU
///   → Linear(256, 2)
#[derive(Module, Debug)]
pub struct UrlClassifierModel4<B: Backend> {
    pub dense1: Linear<B>,
    pub dense2: Linear<B>,
    pub bn1: BatchNorm<B>,
    pub dense3: Linear<B>,
    pub bn2: BatchNorm<B>,
    pub dense4: Linear<B>,
    pub output_layer: Linear<B>,
    input_dim: usize,
    output_dim: usize,
}

/// Config for [`UrlClassifierModel4`].
///
/// Defaults are tuned for the URL encoding (8415 features) and binary
/// classification (2 output classes). Adjust `hidden_dim_N` to change
/// layer sizes.
#[derive(Clone)]
pub struct UrlClassifierModel4Config {
    pub input_dim: usize,
    pub hidden_dim_1: usize,
    pub hidden_dim_2: usize,
    pub hidden_dim_3: usize,
    pub hidden_dim_4: usize,
    pub output_dim: usize,
}

impl Default for UrlClassifierModel4Config {
    fn default() -> Self {
        Self {
            input_dim: 8415,
            hidden_dim_1: 2048,
            hidden_dim_2: 1024,
            hidden_dim_3: 512,
            hidden_dim_4: 256,
            output_dim: 2,
        }
    }
}

impl<B: Backend> UrlClassifierModel4<B> {
    /// Create a new model from a config, initialized on the given device.
    pub fn new(config: &UrlClassifierModel4Config, device: &B::Device) -> Self {
        Self {
            dense1: LinearConfig::new(config.input_dim, config.hidden_dim_1).init(device),
            dense2: LinearConfig::new(config.hidden_dim_1, config.hidden_dim_2).init(device),
            bn1: BatchNormConfig::new(config.hidden_dim_2).init(device),
            dense3: LinearConfig::new(config.hidden_dim_2, config.hidden_dim_3).init(device),
            bn2: BatchNormConfig::new(config.hidden_dim_3).init(device),
            dense4: LinearConfig::new(config.hidden_dim_3, config.hidden_dim_4).init(device),
            output_layer: LinearConfig::new(config.hidden_dim_4, config.output_dim).init(device),
            input_dim: config.input_dim,
            output_dim: config.output_dim,
        }
    }
}

impl<B: Backend> UrlClassifier<B> for UrlClassifierModel4<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.dense1.forward(input));
        let x = self.dense2.forward(x);
        let x = activation::relu(self.bn1.forward(x));
        let x = self.dense3.forward(x);
        let x = activation::relu(self.bn2.forward(x));
        let x = activation::relu(self.dense4.forward(x));
        self.output_layer.forward(x)
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }
}

impl<B: Backend> UrlClassifierConfig<B> for UrlClassifierModel4Config {
    type Model = UrlClassifierModel4<B>;

    fn init(&self, device: &B::Device) -> Self::Model {
        UrlClassifierModel4::new(self, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    /// Test that the reference model can be created from config and forward
    /// produces the expected output shape.
    #[test]
    fn test_model4_forward_shape() {
        let device = Default::default();
        let config = UrlClassifierModel4Config::default();
        let model: UrlClassifierModel4<TestBackend> = config.init(&device);

        let batch_size = 4;
        let input = Tensor::<TestBackend, 2>::zeros([batch_size, config.input_dim], &device);
        let output = model.forward(input);

        assert_eq!(output.shape().dims(), [batch_size, config.output_dim]);
    }

    /// Test that custom dimensions in the config are actually used by the model.
    #[test]
    fn test_model4_custom_config_controls_forward_shape() {
        let device = Default::default();
        let config = UrlClassifierModel4Config {
            input_dim: 12,
            hidden_dim_1: 8,
            hidden_dim_2: 6,
            hidden_dim_3: 4,
            hidden_dim_4: 3,
            output_dim: 5,
        };
        let model: UrlClassifierModel4<TestBackend> = config.init(&device);

        let batch_size = 3;
        let input = Tensor::<TestBackend, 2>::zeros([batch_size, config.input_dim], &device);
        let output = model.forward(input);

        assert_eq!(model.input_dim(), config.input_dim);
        assert_eq!(model.output_dim(), config.output_dim);
        assert_eq!(output.shape().dims(), [batch_size, config.output_dim]);
    }

    /// Test that custom model types can implement the traits.
    #[test]
    fn test_custom_model_implements_trait() {
        use burn::nn::LinearConfig;

        #[derive(Module, Debug)]
        struct CustomModel<B: Backend> {
            linear: Linear<B>,
        }

        impl<B: Backend> UrlClassifier<B> for CustomModel<B> {
            fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
                self.linear.forward(input)
            }
            fn input_dim(&self) -> usize {
                10
            }
            fn output_dim(&self) -> usize {
                2
            }
        }

        #[derive(Clone)]
        struct CustomConfig {
            input_dim: usize,
            output_dim: usize,
        }

        impl<B: Backend> UrlClassifierConfig<B> for CustomConfig {
            type Model = CustomModel<B>;
            fn init(&self, device: &B::Device) -> Self::Model {
                CustomModel {
                    linear: LinearConfig::new(self.input_dim, self.output_dim).init(device),
                }
            }
        }

        // Verify it works end-to-end
        let device = Default::default();
        let config = CustomConfig {
            input_dim: 10,
            output_dim: 2,
        };
        let model = config.init(&device);

        let input = Tensor::<TestBackend, 2>::zeros([1, 10], &device);
        let output = model.forward(input);

        assert_eq!(output.shape().dims(), [1, 2]);
    }
}