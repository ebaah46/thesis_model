// must provide access to reading data from datasource and represent data in an internal
// data structure
//
use anyhow::{Ok, Result as AnyhowResult};
use csv::Reader;
use std::{fs::File, path::Path};

use serde::{
    Deserialize, Deserializer,
    de::{self},
};
// instance of a record in dataset
#[derive(Debug, Deserialize)]
pub struct RawRecord {
    pub url: String,
    #[serde(rename = "result")]
    #[serde(deserialize_with = "bool_from_u8")]
    pub label: bool,
}

fn bool_from_u8<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    match i32::deserialize(deserializer)? {
        0 => std::result::Result::Ok(false),
        1 => std::result::Result::Ok(true),
        other => Err(de::Error::invalid_value(
            de::Unexpected::Signed(other as i64),
            &"0 or 1",
        )),
    }
}

pub fn load_records_from_csv<P: AsRef<Path>>(file_path: P) -> AnyhowResult<Vec<RawRecord>> {
    let file = File::open(file_path)?;

    let mut reader = Reader::from_reader(file);

    let mut records = vec![];
    for record_result in reader.deserialize::<RawRecord>() {
        let record  = record_result.inspect_err(|err|{
            println!("{}", err.to_string());
            log::warn!("core::data::source::load_data_from_file - failed to parse line into record object. Error: {} ", err.to_string());
        })?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_load_records_success() {
        let mut data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        data_path.push("data");
        data_path.push("test.csv");

        let records = load_records_from_csv(data_path).unwrap();
        assert!(!records.is_empty());
        assert_eq!(records.len(), 8);
        assert_eq!(records[0].label, false); // false represents benign urls
        assert_eq!(records[0].url, "https://www.google.com"); // url validation

        assert_eq!(records[5].label, true); // true represents malicious urls
        assert_eq!(records[5].url, "http://frozo.ru/wp-admin/user/"); //  url validation
    }

    // csv file not found scenario
    #[test]
    // #[should_panic]
    fn test_load_records_failure() {
        let mut data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        data_path.push("data");
        data_path.push("dummy_test.csv");

        let records_result = load_records_from_csv(data_path);
        assert!(records_result.is_err());
        let err = records_result.unwrap_err();
        let err_str = err.to_string();
        assert!(err_str.contains("No such file or directory"));
    }
}
