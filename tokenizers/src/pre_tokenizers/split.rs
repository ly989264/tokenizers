use crate::utils::SysRegex;
use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::{
    pattern::{Invert, Pattern},
    Offsets, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};
use std::cell::RefCell;
use std::sync::OnceLock;

/// Represents the different patterns that `Split` can use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub enum SplitPattern {
    String(String),
    Regex(String),
}

impl From<String> for SplitPattern {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for SplitPattern {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub struct Split {
    pub pattern: SplitPattern,
    #[serde(skip)]
    pub regex: SysRegex,
    pub behavior: SplitDelimiterBehavior,
    pub invert: bool,
}

impl<'de> Deserialize<'de> for Split {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        enum Type {
            Split,
        }

        #[derive(Deserialize)]
        pub struct SplitHelper {
            #[serde(rename = "type")]
            _type: Type,
            pattern: SplitPattern,
            behavior: SplitDelimiterBehavior,
            invert: bool,
        }

        let helper = SplitHelper::deserialize(deserializer)?;
        Self::new(helper.pattern, helper.behavior, helper.invert).map_err(serde::de::Error::custom)
    }
}

impl Clone for Split {
    fn clone(&self) -> Self {
        Self::new(self.pattern.clone(), self.behavior, self.invert).unwrap()
    }
}

impl PartialEq for Split {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.behavior == other.behavior
            && self.invert == other.invert
    }
}

impl Split {
    pub fn new<I: Into<SplitPattern>>(
        pattern: I,
        behavior: SplitDelimiterBehavior,
        invert: bool,
    ) -> Result<Self> {
        let pattern: SplitPattern = pattern.into();
        let regex = match &pattern {
            SplitPattern::String(s) => SysRegex::new(&regex::escape(s))?,
            SplitPattern::Regex(r) => SysRegex::new(r)?,
        };

        Ok(Self {
            pattern,
            regex,
            behavior,
            invert,
        })
    }
}

/// A GPT-family pre-tokenization regex that `bitcannon` reproduces byte for byte (its
/// parity tests check each grammar against oniguruma). For these patterns the matches
/// come from its SIMD splitter instead of the regex engine; anything else uses `regex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Native {
    Gpt2,
    Cl100k,
    Qwen,
    O200k,
    Tekken,
    Kimi,
}

impl Native {
    fn recognize(pattern: &SplitPattern) -> Option<Self> {
        use bitcannon::regexes::{CL100K, GPT2, KIMI_K2, O200K, TEKKEN};
        static QWEN: OnceLock<String> = OnceLock::new();
        let r = match pattern {
            SplitPattern::Regex(r) => r.as_str(),
            SplitPattern::String(_) => return None,
        };
        // Without SIMD the splitter is a byte-at-a-time reference, slower than the regex.
        if !bitcannon::fast_builder() {
            return None;
        }
        // Qwen2 / Qwen3 ship cl100k with a bare `\p{N}`, the form bitcannon's tests use.
        let qwen = QWEN.get_or_init(|| CL100K.replace(r"\p{N}{1,3}", r"\p{N}"));
        match r {
            _ if r == GPT2 => Some(Self::Gpt2),
            _ if r == CL100K => Some(Self::Cl100k),
            _ if r == qwen.as_str() => Some(Self::Qwen),
            _ if r == O200K => Some(Self::O200k),
            _ if r == TEKKEN => Some(Self::Tekken),
            _ if r == KIMI_K2 => Some(Self::Kimi),
            _ => None,
        }
    }
}

/// Reusable buffers for the splitter: one tag per byte, the bitmaps it works on, and
/// one span per byte plus one.
#[derive(Default)]
struct NativeScratch {
    tags: Vec<u8>,
    starts: Vec<u64>,
    flags: Vec<u64>,
    later: Vec<u64>,
    spans: Vec<bitcannon::Span>,
}

thread_local! {
    static NATIVE_SCRATCH: RefCell<NativeScratch> = RefCell::new(NativeScratch::default());
}

impl Pattern for &Native {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }
        let bytes = inside.as_bytes();
        let n = bytes.len();
        NATIVE_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let NativeScratch {
                tags,
                starts,
                flags,
                later,
                spans,
            } = &mut *scratch;
            let words = n.div_ceil(64) + 1;
            tags.resize(n.max(tags.len()), 0);
            starts.resize(words.max(starts.len()), 0);
            flags.resize(words.max(flags.len()), 0);
            later.resize((2 * words).max(later.len()), 0);
            spans.resize((n + 1).max(spans.len()), bitcannon::Span::default());
            bitcannon::classify::classify(bytes, &mut tags[..n]);
            let (t, s, f, l, o) = (
                &tags[..n],
                &mut starts[..words],
                &mut flags[..words],
                &mut later[..2 * words],
                &mut spans[..n + 1],
            );
            let k = match self {
                Native::Gpt2 => bitcannon::bitcannon_byte_level(bytes, t, s, f, o),
                Native::Cl100k => bitcannon::bitcannon_cl100k(bytes, t, s, f, o),
                Native::Qwen => bitcannon::bitcannon_qwen(bytes, t, s, f, o),
                Native::O200k => bitcannon::bitcannon_o200k(bytes, t, s, f, l, o),
                Native::Tekken => bitcannon::bitcannon_tekken(bytes, t, s, f, l, o),
                Native::Kimi => bitcannon::bitcannon_kimi(bytes, t, s, f, l, o),
            };
            // Same layout as `&SysRegex`: every match, with any gap between them.
            let mut prev = 0;
            let mut splits = Vec::with_capacity(k + 1);
            for span in &spans[..k] {
                let (start, end) = (span.start as usize, span.end as usize);
                if prev != start {
                    splits.push(((prev, start), false));
                }
                splits.push(((start, end), true));
                prev = end;
            }
            if prev != n {
                splits.push(((prev, n), false));
            }
            Ok(splits)
        })
    }
}

impl PreTokenizer for Split {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        if let Some(native) = Native::recognize(&self.pattern) {
            // The splitter's span offsets are u32.
            let fits = |s: &str| s.len() <= u32::MAX as usize;
            return if self.invert {
                pretokenized.split(|_, normalized| {
                    if fits(normalized.get()) {
                        normalized.split(Invert(&native), self.behavior)
                    } else {
                        normalized.split(Invert(&self.regex), self.behavior)
                    }
                })
            } else {
                pretokenized.split(|_, normalized| {
                    if fits(normalized.get()) {
                        normalized.split(&native, self.behavior)
                    } else {
                        normalized.split(&self.regex, self.behavior)
                    }
                })
            };
        }
        if self.invert {
            pretokenized.split(|_, normalized| normalized.split(Invert(&self.regex), self.behavior))
        } else {
            pretokenized.split(|_, normalized| normalized.split(&self.regex, self.behavior))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizer};
    use SplitDelimiterBehavior::*;

    #[test]
    fn basic() {
        let tests = vec![
            (
                Removed,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                Isolated,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    (" ", (3, 4)),
                    ("are", (4, 7)),
                    (" ", (7, 8)),
                    ("you", (8, 11)),
                    (" ", (11, 12)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                MergedWithPrevious,
                "How are you doing?",
                vec![
                    ("How ", (0, 4)),
                    ("are ", (4, 8)),
                    ("you ", (8, 12)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                MergedWithNext,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    (" are", (3, 7)),
                    (" you", (7, 11)),
                    (" doing", (11, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                Contiguous,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    (" ", (3, 4)),
                    ("are", (4, 7)),
                    (" ", (7, 8)),
                    ("you", (8, 11)),
                    (" ", (11, 12)),
                    ("doing?", (12, 18)),
                ],
            ),
        ];

        // use whitespace regex
        let regex = SplitPattern::Regex(r"\w+|[^\w\s]+".into());

        for (behavior, s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            let pretok = Split::new(regex.clone(), behavior, true).unwrap();
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }

    #[test]
    fn regex_string() {
        let mut pretok_str_for_regex = PreTokenizedString::from("Hey, man!");
        let mut pretok_str_for_string = pretok_str_for_regex.clone();

        // pre-tokenizer splits on " " - one from Regex, one from string
        let pretokenizer_regex = Split::new(
            SplitPattern::Regex(r"\s+".into()),
            SplitDelimiterBehavior::Removed,
            false,
        )
        .unwrap();
        let pretokenizer_string = Split::new(" ", SplitDelimiterBehavior::Removed, false).unwrap();

        pretokenizer_regex
            .pre_tokenize(&mut pretok_str_for_regex)
            .unwrap();
        pretokenizer_string
            .pre_tokenize(&mut pretok_str_for_string)
            .unwrap();

        assert_eq!(pretok_str_for_regex, pretok_str_for_string);
    }

    #[test]
    fn invert() {
        let mut pretok_str = PreTokenizedString::from("Hello Hello Hello");
        let mut pretok_str_for_invert = pretok_str.clone();

        // one pre-tokenizer splits on " " - one splits inverted on "Hello"
        let pretokenizer = Split::new(" ", SplitDelimiterBehavior::Removed, false).unwrap();
        let pretokenizer_invert =
            Split::new("Hello", SplitDelimiterBehavior::Removed, true).unwrap();

        pretokenizer.pre_tokenize(&mut pretok_str).unwrap();
        pretokenizer_invert
            .pre_tokenize(&mut pretok_str_for_invert)
            .unwrap();

        assert_eq!(pretok_str, pretok_str_for_invert);
    }

    #[test]
    fn serialization() {
        use SplitDelimiterBehavior::*;

        let split = Split::new("Hello", Removed, true).unwrap();
        let split_s =
            r#"{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true}"#;
        assert_eq!(serde_json::to_string(&split).unwrap(), split_s);
        assert_eq!(serde_json::from_str::<Split>(split_s).unwrap(), split);

        let split = Split::new(SplitPattern::Regex(r"\s+".into()), Isolated, false).unwrap();
        let split_s =
            r#"{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Isolated","invert":false}"#;
        assert_eq!(serde_json::to_string(&split).unwrap(), split_s);
        assert_eq!(serde_json::from_str::<Split>(split_s).unwrap(), split);
    }

    #[test]
    fn native_matches_regex() {
        use bitcannon::regexes::{CL100K, GPT2, KIMI_K2, O200K, TEKKEN};
        let qwen = CL100K.replace(r"\p{N}{1,3}", r"\p{N}");
        let texts = [
            "",
            "The quick brown fox. Don't 12345 numbers, \u{00BD}\u{00B2}\u{00BC} \u{2168}!",
            "café × naïve — Привет, наука! Ελλάδα 中文分词。ひらがな カタカナ 한글 مرحبا",
            "IT'S O'Brien can't 'quoted' l'été rock'n'roll\r\n0 42 999 1000 1234567 v1.2.3",
            "https://host/a/b?c=1&d=2 path/to//file /\r\n/ ///x CamelCase XMLHttpRequest",
            "日本語1234テスト ½3¼ #tag @user $9.99 100% end\n\n\nlines\r\n\r\n  \t  trailing   ",
            "😀👍 hello  world\ttabs\nnewlines   end \u{200d}\u{0301}a\u{0301}b",
        ];
        // Every split of a longer text at each char boundary walks the rules across the
        // splitter's 64-byte blocks.
        let long: String = texts.concat().repeat(3);
        let mut inputs: Vec<&str> = texts.to_vec();
        inputs.extend(long.char_indices().map(|(i, _)| &long[i..]));
        for pattern in [GPT2, CL100K, qwen.as_str(), O200K, TEKKEN, KIMI_K2] {
            let split = Split::new(SplitPattern::Regex(pattern.into()), Isolated, false).unwrap();
            let native = match Native::recognize(&split.pattern) {
                Some(native) => native,
                None if !bitcannon::fast_builder() => return,
                None => panic!("not recognized: {pattern}"),
            };
            for text in &inputs {
                assert_eq!(
                    (&native).find_matches(text).unwrap(),
                    (&split.regex).find_matches(text).unwrap(),
                    "{pattern} on {text:?}"
                );
            }
        }
    }
}
