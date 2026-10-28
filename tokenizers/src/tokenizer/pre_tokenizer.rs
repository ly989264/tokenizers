use crate::{
    normalizer::Range, Encoding, NormalizedString, OffsetReferential, Offsets, Result, Token,
};

/// Various possible types of offsets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetType {
    Byte,
    Char,
    None,
}

/// Wrapper for a subpart of a `NormalizedString`.
///
/// This Split contains the underlying `NormalizedString` as well as its offsets
/// in the original string. These offsets are in the `original` referential.
/// It also contains any `Token` associated to the current split
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    /// The underlying `NormalizedString`. Each SubString is represented by a `NormalizedString`
    /// and in the end we might be carrying a lot of SubString representing various parts of the
    /// original input string.
    normalized: NormalizedString,
    /// Optional Tokens associated to this Split
    tokens: Option<Vec<Token>>,
}

impl From<NormalizedString> for Split {
    fn from(n: NormalizedString) -> Self {
        Self {
            normalized: n,
            tokens: None,
        }
    }
}

impl From<(NormalizedString, Option<Vec<Token>>)> for Split {
    fn from(f: (NormalizedString, Option<Vec<Token>>)) -> Self {
        Self {
            normalized: f.0,
            tokens: f.1,
        }
    }
}

/// The `PreTokenizedString` is in charge of splitting an underlying string,
/// making sure everything is fine while doing so, and providing ways to normalize
/// and tokenize these splits.
/// Once everything has been normalized and tokenized, the `PreTokenizedString` is able
/// to build an `Encoding` with all the relevant offsets and word ids, relative to the
/// original string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreTokenizedString {
    original: String,
    splits: Vec<Split>,
}

impl PreTokenizedString {
    /// Split the `PreTokenizedString` by providing a `split_fn` in charge of splitting
    /// each substring (`NormalizedString`) into multiple parts.
    ///
    /// `split_fn` takes a `NormalizedString` and is in charge of returning an iterator
    /// over the produced `NormalizedString`. `split_fn` is free of modifying these
    /// `NormalizedString` as relevant, as long as it respects the constraint stated below.
    ///
    /// There are only one constraint that *MUST* be respected:
    /// > The produced `NormalizedString`, if combined back together, must have the
    /// > same `original` string as the original one given to `split_fn`. This concretely
    /// > means that for the offset tracking to work as expected, `split_fn` must produce
    /// > "splits" of the original string.
    pub fn split<F, U, R>(&mut self, mut split_fn: F) -> Result<()>
    where
        F: FnMut(usize, NormalizedString) -> Result<U>,
        U: IntoIterator<Item = R>,
        R: Into<Split>,
    {
        // new_splits is at least as big as self.splits
        let mut new_splits = Vec::with_capacity(self.splits.len());
        for (i, original_split) in self.splits.drain(..).enumerate() {
            if original_split.tokens.is_some() {
                new_splits.push(original_split);
                continue;
            }

            new_splits.extend(
                split_fn(i, original_split.normalized)?
                    .into_iter()
                    .filter_map(|split| {
                        let split: Split = split.into();
                        if split.normalized.is_empty() {
                            None
                        } else {
                            Some(split)
                        }
                    }),
            );
        }
        self.splits = new_splits;

        Ok(())
    }

    /// Normalized all the splits that do not have attached `Tokens`, using the provided
    /// `normalize` function.
    pub fn normalize<F>(&mut self, normalize: F) -> Result<()>
    where
        F: Fn(&mut NormalizedString) -> Result<()>,
    {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            normalize(&mut split.normalized)?;
        }
        Ok(())
    }

    /// Tokenize all the splits that do not have attached `Tokens`, using the provided
    /// `tokenize` function
    pub fn tokenize<F>(&mut self, tokenize: F) -> Result<()>
    where
        F: Fn(&NormalizedString) -> Result<Vec<Token>>,
    {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            split.tokens = Some(tokenize(&split.normalized)?);
        }

        Ok(())
    }

    /// Transform the current `PreTokenizedString` into an `Encoding`.
    ///
    /// If a `word_idx` is provided, any word in the generated `Encoding`
    /// will be set to this value. This is generally used with pre-tokenized
    /// input, that do not need the `PreTokenizedString` to generate word ids.
    ///
    /// This method will fail if some splits do not have associated `Token`.
    pub fn into_encoding(
        self,
        word_idx: Option<u32>,
        type_id: u32,
        offset_type: OffsetType,
    ) -> Result<Encoding> {
        if self.splits.is_empty() {
            Ok(Encoding::default())
        } else if !self.splits.iter().all(|split| split.tokens.is_some()) {
            Err("Split has not been tokenized, call `PreTokenizedString::tokenize` first".into())
        } else {
            let offset_converter = match offset_type {
                OffsetType::Char => Some(BytesToCharOffsetConverter::new(&self.original)),
                OffsetType::Byte => None,
                OffsetType::None => {
                    let tokens = self
                        .splits
                        .into_iter()
                        .flat_map(|split| {
                            split.tokens.unwrap().into_iter().map(|token| {
                                // Replace this with the actual fields you need for the Encoding type
                                (token.id, String::with_capacity(0), (0, 0), None, 0)
                            })
                        })
                        .collect();
                    return Ok(tokens);
                }
            };

            Ok(self
                .splits
                .into_iter()
                .enumerate()
                .flat_map(|(idx, split)| {
                    let normalized = split.normalized;
                    let offsets = normalized.offsets_original();
                    let offset_converter = &offset_converter;

                    split.tokens.unwrap().into_iter().map(move |token| {
                        let mut offsets = normalized
                            .convert_offsets(Range::Normalized(token.offsets.0..token.offsets.1))
                            .map_or(token.offsets, |range| {
                                (offsets.0 + range.start, offsets.0 + range.end)
                            });

                        // Convert to char offsets if relevant
                        if let Some(converter) = offset_converter {
                            offsets = converter.convert(offsets).unwrap_or(offsets);
                        }

                        (
                            token.id,
                            token.value,
                            offsets,
                            if word_idx.is_some() {
                                word_idx
                            } else {
                                Some(idx as u32)
                            },
                            type_id,
                        )
                    })
                })
                .collect())
        }
    }

    /// Returns a list of splits, each of them being a slice of the normalized
    /// string, the associated offsets either in original or normalized
    /// referential, as well as the potention tokens
    pub fn get_splits(
        &self,
        offset_ref: OffsetReferential,
        offset_type: OffsetType,
    ) -> Vec<(&str, Offsets, &Option<Vec<Token>>)> {
        let offset_converter = match offset_type {
            OffsetType::Char => Some(BytesToCharOffsetConverter::new(&self.original)),
            OffsetType::Byte => None,
            OffsetType::None => None,
        };

        let mut offset = 0;
        self.splits
            .iter()
            .map(|split| {
                let mut offsets = match offset_ref {
                    OffsetReferential::Original => split.normalized.offsets_original(),
                    OffsetReferential::Normalized => {
                        let len = split.normalized.len();
                        offset += len;
                        (offset - len, offset)
                    }
                };

                // Convert to char offsets if relevant
                if let Some(ref converter) = offset_converter {
                    offsets = converter.convert(offsets).unwrap_or(offsets);
                }

                (split.normalized.get(), offsets, &split.tokens)
            })
            .collect()
    }
}

impl From<NormalizedString> for PreTokenizedString {
    fn from(s: NormalizedString) -> Self {
        Self {
            original: s.get_original().to_owned(),
            splits: vec![Split {
                normalized: s,
                tokens: None,
            }],
        }
    }
}

impl From<&str> for PreTokenizedString {
    fn from(s: &str) -> Self {
        let normalized: NormalizedString = s.into();
        normalized.into()
    }
}

impl From<String> for PreTokenizedString {
    fn from(s: String) -> Self {
        let normalized: NormalizedString = s.into();
        normalized.into()
    }
}

/// Maps a byte position of the original string to the index of the char that
/// contains it. Positions at or past the end have no entry.
enum BytesToCharOffsetConverter {
    /// Every char is one byte: the char index is the byte index.
    Ascii(usize),
    /// One entry per byte.
    Table(Vec<usize>),
}

impl BytesToCharOffsetConverter {
    pub fn new(sequence: &str) -> Self {
        if sequence.is_ascii() {
            return Self::Ascii(sequence.len());
        }
        let mut map = Vec::with_capacity(sequence.len());
        for (i, c) in sequence.chars().enumerate() {
            map.extend(std::iter::repeat_n(i, c.len_utf8()));
        }
        Self::Table(map)
    }

    fn get(&self, byte: usize) -> Option<usize> {
        match self {
            Self::Ascii(len) => (byte < *len).then_some(byte),
            Self::Table(map) => map.get(byte).copied(),
        }
    }

    pub fn convert(&self, offsets: Offsets) -> Option<Offsets> {
        match (self.get(offsets.0), self.get(offsets.1)) {
            (Some(start), Some(end)) => Some((start, end)),
            // If we reached the end, `end` is not in the map
            (Some(start), None) => {
                // But the one just before should be
                let last = self.get(offsets.1 - 1).unwrap_or(start + 1);
                Some((start, last + 1))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The map the converter used to build: one entry per byte of the string.
    fn reference(s: &str, offsets: Offsets) -> Option<Offsets> {
        let map: HashMap<usize, usize> = s
            .char_indices()
            .enumerate()
            .flat_map(|(i, (b, c))| (b..b + c.len_utf8()).map(move |p| (p, i)))
            .collect();
        match (map.get(&offsets.0), map.get(&offsets.1)) {
            (Some(start), Some(end)) => Some((*start, *end)),
            (Some(start), None) => {
                let last = map.get(&(offsets.1 - 1)).copied().unwrap_or(start + 1);
                Some((*start, last + 1))
            }
            _ => None,
        }
    }

    #[test]
    fn bytes_to_char_offsets() {
        for s in ["", "a", "hello world", "é", "aé", "日本語", "a👍🏽b", "e\u{301}x", "\u{10000}"] {
            let converter = BytesToCharOffsetConverter::new(s);
            for start in 0..=s.len() + 2 {
                for end in start..=s.len() + 2 {
                    assert_eq!(
                        converter.convert((start, end)),
                        reference(s, (start, end)),
                        "{s:?} {start}..{end}"
                    );
                }
            }
        }
    }
}
