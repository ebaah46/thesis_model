// Must contain all the data
// Should also provide an api for preprocessing
// Should convert data into acceptable multi-dimensional array for NN model.
// Preprocessing must include all tokenization primitives.
// Should also provide possibility to extend tokenization API strategy
// use burn::data::dataset::Dataset as BurnDataset;
use anyhow;
use burn::{data::dataset::Dataset as BurnDataset, prelude::*};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::data::{OHEStrategy, Preprocessing, load_records_from_csv};

// This is a preprocessed representation of each record
// in the dataset. At this point, each record is model ready
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RawItem {
    // label shows whether the record is malicious or not
    // 0 means NOT benign and 1 means otherwise
    pub is_malicious: u8,

    // the internal representation for each url in this data set
    // in this IR, the maximum character limit for each url is set at
    // 99 and the character set length is set at 85. This character set includes
    // all ASCII characters. Non-ascii characters are ignored in this representation
    // Here is the total ascii character set used
    // abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;=
    pub url: Vec<f32>,
}

// Dataset
pub struct Dataset {
    // model-ready dataset records
    pub records: Vec<RawItem>,
}

impl Dataset {
    // retrieve dataset from source
    pub fn load_dataset_csv<P: AsRef<Path>>(file: P) -> anyhow::Result<Self> {
        let recs = load_records_from_csv(file)?;
        let records = recs
            .iter()
            .map(|entry| OHEStrategy::encode_url(entry))
            .collect::<Vec<RawItem>>();
        Ok(Self { records })
    }
}

impl BurnDataset<RawItem> for Dataset {
    fn get(&self, index: usize) -> Option<RawItem> {
        self.records.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.records.len()
    }
}
