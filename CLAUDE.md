
## Project Overview

Thesis project: Deep learning model for malicious URL detection using the Burn framework in Rust. As an expert rust engineer proficient in the burn framework, you will work on the remaining features of the project.

## Technical Decisions
- Burn framework with 2 main backends for unit test and production. ndarray for CPU and wgpu for GPU.
- Data format: CSV with url, label fields. In future, we may explore other formats when we begin to fetch from platforms like huggingface.

## Development Commands

```bash
# Check compilation
cargo check

# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p ml_core

# Run a specific test
cargo test test_charset_length -p ml_core

# Build workspace
cargo build

# Build release binary
cargo build --release
```

## Architecture

Two-crate workspace:

- **ml_core**: Library crate containing data pipeline and model definitions. Also reponsible for creating the training functionality.
- **ml_app**: Binary crate that serves as the entry point for the application. Responsible for providing an inference server and handling user requests for URL classification.

### URL Encoding

The encoding module is reponsible for the tokenization of URLs. The first tokenizer is the one-hot encoding strategy that creates a one-hot representation per character position, producing a sparse vector suitable for feed-forward networks. Other tokenizers will be added in the future.

### Model (ml_core/src/model/)

The model module is reponsible for defining neural network models. It should contain parametized model architectures.

### Dataset Format

CSV files require columns: `url`, `label`, `result`

- `result` (0 or 1) is used as the training label
- `label` (benign/malicious) is currently ignored by the Rust model
- Main dataset: `ml_core/data/balanced_urls.csv` (~632K rows, balanced)
- Test fixture: `ml_core/data/test.csv` (8 rows). Should be prefered in all test cases.

## Rules

- Prefer Test Driven Development approach to creating new features. Tests must be approved by reviewer before implementing logic for any new feature.
- Never use the `ml_core/data/balanced_urls.csv` dataset in any testing or evaluation code.
- Always use the test.csv fixture for testing.
- Burn's API changes across versions — check the version pinned in Cargo.toml and consult docs.rs/burn for the exact API rather than assuming a remembered signature, especially for Tensor/Module/Learner APIs.
- Don't duplicate feature-extraction logic that already exists in crates/data-processing/src/features.rs — extend it instead.
- No dataset files committed to git; data lives in data/ (gitignored).
- Prefer `Result<T, E>` over `panic!`/`unwrap`/`expect` in library code. Panics are allowed in tests, but should be handled with `#[should_panic]`.
- Don't suppress a clippy warning with `#[allow(...)]` as the default fix — fix the underlying pattern.
