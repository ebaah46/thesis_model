// Should define architecture of Neural network model described here

use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::NdArray,
        nn::{Linear, LinearConfig},
        tensor::Tensor,
    };

    type TestBackend = NdArray<f32>;

    /// Test that custom model types can implement the traits.
    #[test]
    fn test_custom_model_implements_trait() {
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