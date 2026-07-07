// responsible for generating preprocessing function

use crate::data::{RawItem, RawRecord};

// Main trait for preprocessing of dataset into model-ready
// format
// Accepts data in the raw record format and transforms it into
// model-ready format
pub trait Preprocessing<Item> {
    fn encode_url(record: &RawRecord) -> Item;
}

// The one-hot encoding preprocessing strategy is
pub struct OHEStrategy {}

impl OHEStrategy {
    const CHARSET: &[u8] =
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;=";
    pub const CHARSET_LEN: usize = 84;
    pub const CHARSET_WORD_LEN: usize = Self::CHARSET_LEN + 1; // adding null byte character
    pub const FIXED_URL_LEN: usize = 99;
    pub const ENCODED_URL_LEN: usize = Self::FIXED_URL_LEN * Self::CHARSET_WORD_LEN; // 8415
    pub const LOOK_UP_TABLE: [u8; 256] = {
        let mut table = [255u8; 256];
        let mut i = 0;

        while i < OHEStrategy::CHARSET_LEN {
            table[OHEStrategy::CHARSET[i] as usize] = i as u8;
            i += 1;
        }
        table
    };
}

impl Preprocessing<RawItem> for OHEStrategy {
    fn encode_url(record: &RawRecord) -> RawItem {
        let mut encoded = [0.0f32; OHEStrategy::ENCODED_URL_LEN];
        let url_bytes = record.url.as_bytes();
        let url_len = url_bytes.len().min(OHEStrategy::FIXED_URL_LEN);
        for j in 0..url_len {
            let idx = OHEStrategy::LOOK_UP_TABLE[url_bytes[j] as usize];
            if idx != 255 {
                encoded[j * OHEStrategy::CHARSET_WORD_LEN + idx as usize] = 1.0;
            }
        }
        RawItem {
            is_malicious: record.label,
            url: encoded.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charset_length() {
        assert_eq!(OHEStrategy::CHARSET.len(), OHEStrategy::CHARSET_LEN);
        assert_eq!(OHEStrategy::CHARSET_LEN, 84);
        assert_eq!(OHEStrategy::CHARSET_WORD_LEN, 85);
    }

    #[test]
    fn test_encoding_length() {
        let record = RawRecord {
            url: "https://example.com".to_owned(),
            label: 0,
        };
        let enc = OHEStrategy::encode_url(&record);
        assert_eq!(enc.url.len(), OHEStrategy::ENCODED_URL_LEN);
        assert_eq!(enc.url.len(), 8415);
    }

    #[test]
    fn test_known_char_positions() {
        // 'a' is at charset index 0
        assert_eq!(OHEStrategy::LOOK_UP_TABLE[b'a' as usize], 0);
        // 'z' is at charset index 25
        assert_eq!(OHEStrategy::LOOK_UP_TABLE[b'z' as usize], 25);
        // 'A' is at charset index 26
        assert_eq!(OHEStrategy::LOOK_UP_TABLE[b'A' as usize], 26);
        // '0' is at charset index 52
        assert_eq!(OHEStrategy::LOOK_UP_TABLE[b'0' as usize], 52);
        // '=' is at charset index 83
        assert_eq!(OHEStrategy::LOOK_UP_TABLE[b'=' as usize], 83);
    }

    #[test]
    fn test_unknown_chars_are_255() {
        assert_eq!(OHEStrategy::LOOK_UP_TABLE[b' ' as usize], 255);
        assert_eq!(OHEStrategy::LOOK_UP_TABLE[b'\n' as usize], 255);
        assert_eq!(OHEStrategy::LOOK_UP_TABLE[b'\0' as usize], 255);
        assert_eq!(OHEStrategy::LOOK_UP_TABLE[b'%' as usize], 255);
    }

    #[test]
    fn test_simple_encoding() {
        let record = RawRecord {
            url: "a".into(),
            label: 1,
        };
        let enc = OHEStrategy::encode_url(&record);
        // Position 0, charset index 0 should be 1.0
        assert_eq!(enc.url[0 * OHEStrategy::CHARSET_WORD_LEN + 0], 1.0);
        // All other positions in row 0 should be 0
        for i in 1..OHEStrategy::CHARSET_WORD_LEN {
            assert_eq!(enc.url[0 * OHEStrategy::CHARSET_WORD_LEN + i], 0.0);
        }
        // Remaining rows should be all zeros (padding)
        for j in 1..OHEStrategy::FIXED_URL_LEN {
            for i in 0..OHEStrategy::CHARSET_WORD_LEN {
                assert_eq!(
                    enc.url[j * OHEStrategy::CHARSET_WORD_LEN + i],
                    0.0,
                    "unexpected 1.0 at row {j}, col {i}"
                );
            }
        }
    }

    #[test]
    fn test_truncation_to_99() {
        let long_url = RawRecord {
            url: "a".repeat(200),
            label: 0,
        };
        let enc = OHEStrategy::encode_url(&long_url);
        // All 99 positions should have 'a' encoded
        for j in 0..OHEStrategy::FIXED_URL_LEN {
            assert_eq!(
                enc.url[j * OHEStrategy::CHARSET_WORD_LEN + 0],
                1.0,
                "position {j} should have 'a'"
            );
        }
        // Total 1.0 count should be exactly 99
        let count: usize = enc.url.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(count, 99);
    }

    #[test]
    fn test_empty_url() {
        let enc = OHEStrategy::encode_url(&RawRecord {
            url: "".into(),
            label: 0,
        });
        // All zeros (empty URL -> all padding)
        assert!(enc.url.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_unknown_chars_produce_zero_rows() {
        // '%' is not in charset
        let enc = OHEStrategy::encode_url(&RawRecord {
            url: "%".into(),
            label: 0,
        });
        // Position 0 should be all zeros
        for i in 0..OHEStrategy::CHARSET_WORD_LEN {
            assert_eq!(enc.url[0 * OHEStrategy::CHARSET_WORD_LEN + i], 0.0);
        }
    }

    #[test]
    fn test_sum_per_row() {
        let enc = OHEStrategy::encode_url(&RawRecord {
            url: "https://example.com".into(),
            label: 0,
        });
        let url_bytes = b"https://example.com";
        for j in 0..url_bytes.len() {
            let row_sum: f32 = (0..OHEStrategy::CHARSET_WORD_LEN)
                .map(|i| enc.url[j * OHEStrategy::CHARSET_WORD_LEN + i])
                .sum();
            if OHEStrategy::LOOK_UP_TABLE[url_bytes[j] as usize] != 255 {
                assert_eq!(
                    row_sum, 1.0,
                    "row {j} (char '{}') should sum to 1.0",
                    url_bytes[j] as char
                );
            } else {
                assert_eq!(
                    row_sum, 0.0,
                    "row {j} (char '{}') should sum to 0.0",
                    url_bytes[j] as char
                );
            }
        }
        // Padding rows should sum to 0
        for j in url_bytes.len()..OHEStrategy::FIXED_URL_LEN {
            let row_sum: f32 = (0..OHEStrategy::CHARSET_WORD_LEN)
                .map(|i| enc.url[j * OHEStrategy::CHARSET_WORD_LEN + i])
                .sum();
            assert_eq!(row_sum, 0.0, "padding row {j} should sum to 0.0");
        }
    }
}
