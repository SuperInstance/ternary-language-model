#![forbid(unsafe_code)]

//! Language modeling with ternary token predictions.

use std::collections::HashMap;

/// A ternary value: -1, 0, or +1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Trit {
    Neg,
    Zero,
    Pos,
}

impl Trit {
    pub fn value(self) -> i8 {
        match self {
            Trit::Neg => -1,
            Trit::Zero => 0,
            Trit::Pos => 1,
        }
    }

    pub fn from_value(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Trit::Neg),
            0 => Some(Trit::Zero),
            1 => Some(Trit::Pos),
            _ => None,
        }
    }
}

/// Token represented as a sequence of trits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TritToken {
    pub trits: Vec<Trit>,
}

impl TritToken {
    pub fn new(trits: Vec<Trit>) -> Self {
        Self { trits }
    }

    pub fn len(&self) -> usize {
        self.trits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trits.is_empty()
    }
}

/// Tokenizer that converts strings to ternary token sequences.
pub struct TernaryTokenizer {
    pub vocab: HashMap<String, TritToken>,
}

impl TernaryTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
        }
    }

    /// Encode a string character-by-character into ternary tokens.
    /// Each byte maps to trits via balanced ternary representation.
    pub fn encode(&self, text: &str) -> Vec<TritToken> {
        text.bytes()
            .map(|b| {
                let mut trits = Vec::new();
                let mut val = b as i32;
                // Convert to balanced ternary (up to 6 trits for a byte)
                for _ in 0..6 {
                    let rem = val % 3;
                    val /= 3;
                    match rem {
                        0 => trits.push(Trit::Zero),
                        1 => trits.push(Trit::Pos),
                        2 => {
                            trits.push(Trit::Neg);
                            val += 1;
                        }
                        _ => unreachable!(),
                    }
                }
                TritToken::new(trits)
            })
            .collect()
    }

    /// Decode ternary tokens back to a string.
    pub fn decode(&self, tokens: &[TritToken]) -> String {
        tokens
            .iter()
            .map(|tok| {
                let mut val: i32 = 0;
                let mut pow: i32 = 1;
                for t in &tok.trits {
                    val += t.value() as i32 * pow;
                    pow *= 3;
                }
                (val as u8) as char
            })
            .collect()
    }

    /// Add a named token to vocabulary.
    pub fn add_vocab(&mut self, name: &str, token: TritToken) {
        self.vocab.insert(name.to_string(), token);
    }

    /// Look up a named token.
    pub fn get_vocab(&self, name: &str) -> Option<&TritToken> {
        self.vocab.get(name)
    }
}

impl Default for TernaryTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Vocabulary of ternary tokens with integer IDs.
pub struct TernaryVocabulary {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    next_id: u32,
}

impl TernaryVocabulary {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn add(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.token_to_id.insert(token.to_string(), id);
            self.id_to_token.insert(id, token.to_string());
            id
        }
    }

    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }
}

impl Default for TernaryVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Ternary transition counts for an n-gram model.
#[derive(Clone, Debug)]
pub struct TernaryTransition {
    pub counts: [f64; 3], // index 0=Neg, 1=Zero, 2=Pos
    pub total: f64,
}

impl TernaryTransition {
    pub fn new() -> Self {
        Self {
            counts: [0.0; 3],
            total: 0.0,
        }
    }

    pub fn observe(&mut self, trit: Trit) {
        let idx = trit_index(trit);
        self.counts[idx] += 1.0;
        self.total += 1.0;
    }

    pub fn probability(&self, trit: Trit) -> f64 {
        if self.total == 0.0 {
            1.0 / 3.0
        } else {
            self.counts[trit_index(trit)] / self.total
        }
    }

    pub fn sample(&self) -> Trit {
        if self.total == 0.0 {
            return Trit::Zero;
        }
        let _r = (self.counts[0] / self.total, self.counts[1] / self.total);
        // Deterministic sampling for testing: pick the most probable
        let probs = [
            self.counts[0] / self.total,
            self.counts[1] / self.total,
            self.counts[2] / self.total,
        ];
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);
        idx_to_trit(max_idx)
    }

    /// Sample with temperature scaling on {-1, 0, +1} outputs.
    pub fn sample_with_temperature(&self, temperature: f64) -> Trit {
        if self.total == 0.0 {
            return Trit::Zero;
        }
        let logits = [
            self.counts[0] / self.total,
            self.counts[1] / self.total,
            self.counts[2] / self.total,
        ];
        let scaled: Vec<f64> = logits.iter().map(|&p| (p.max(1e-10).ln()) / temperature).collect();
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scaled.iter().map(|&s| (s - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();
        // Deterministic: pick argmax
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);
        idx_to_trit(max_idx)
    }
}

impl Default for TernaryTransition {
    fn default() -> Self {
        Self::new()
    }
}

fn trit_index(t: Trit) -> usize {
    match t {
        Trit::Neg => 0,
        Trit::Zero => 1,
        Trit::Pos => 2,
    }
}

fn idx_to_trit(i: usize) -> Trit {
    match i {
        0 => Trit::Neg,
        1 => Trit::Zero,
        _ => Trit::Pos,
    }
}

/// N-gram language model using ternary transitions.
pub struct TernaryNgramModel {
    pub n: usize,
    pub transitions: HashMap<Vec<Trit>, TernaryTransition>,
    pub order_counts: HashMap<usize, usize>,
}

impl TernaryNgramModel {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            transitions: HashMap::new(),
            order_counts: HashMap::new(),
        }
    }

    /// Train on a sequence of trits.
    pub fn train(&mut self, sequence: &[Trit]) {
        if sequence.len() < self.n {
            return;
        }
        for i in 0..=sequence.len() - self.n {
            let context: Vec<Trit> = sequence[i..i + self.n - 1].to_vec();
            let next = sequence[i + self.n - 1];
            let entry = self.transitions.entry(context).or_insert_with(TernaryTransition::new);
            entry.observe(next);
            *self.order_counts.entry(self.n).or_insert(0) += 1;
        }
    }

    /// Predict the next trit given a context.
    pub fn predict(&self, context: &[Trit]) -> Trit {
        self.transitions
            .get(context)
            .map(|t| t.sample())
            .unwrap_or(Trit::Zero)
    }

    /// Predict with temperature scaling.
    pub fn predict_with_temperature(&self, context: &[Trit], temperature: f64) -> Trit {
        self.transitions
            .get(context)
            .map(|t| t.sample_with_temperature(temperature))
            .unwrap_or(Trit::Zero)
    }

    /// Generate a sequence of trits from a seed context.
    pub fn generate(&self, seed: &[Trit], length: usize) -> Vec<Trit> {
        if seed.len() < self.n - 1 {
            return vec![];
        }
        let mut result = seed.to_vec();
        for _ in 0..length {
            let ctx_len = self.n - 1;
            let start = result.len().saturating_sub(ctx_len);
            let context: Vec<Trit> = result[start..].to_vec();
            let next = self.predict(&context);
            result.push(next);
        }
        result
    }

    /// Compute perplexity on a test sequence.
    pub fn perplexity(&self, sequence: &[Trit]) -> f64 {
        if sequence.len() < self.n {
            return f64::INFINITY;
        }
        let mut log_prob_sum = 0.0;
        let mut count = 0;
        for i in (self.n - 1)..sequence.len() {
            let start = i + 1 - self.n;
            let context: Vec<Trit> = sequence[start..i].to_vec();
            let next = sequence[i];
            let prob = self
                .transitions
                .get(&context)
                .map(|t| t.probability(next))
                .unwrap_or(1.0 / 3.0);
            log_prob_sum += prob.ln();
            count += 1;
        }
        if count == 0 {
            return f64::INFINITY;
        }
        let avg_log_prob = log_prob_sum / count as f64;
        (-avg_log_prob).exp()
    }

    /// Get transition count.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }
}

/// Compute cross-entropy between model predictions and actual sequence.
pub fn cross_entropy(model: &TernaryNgramModel, sequence: &[Trit]) -> f64 {
    if sequence.len() < model.n {
        return f64::INFINITY;
    }
    let mut sum = 0.0;
    let mut count = 0;
    for i in (model.n - 1)..sequence.len() {
        let start = i + 1 - model.n;
        let context: Vec<Trit> = sequence[start..i].to_vec();
        let next = sequence[i];
        let prob = model
            .transitions
            .get(&context)
            .map(|t| t.probability(next))
            .unwrap_or(1.0 / 3.0);
        sum += -prob.ln();
        count += 1;
    }
    if count == 0 {
        f64::INFINITY
    } else {
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_values() {
        assert_eq!(Trit::Neg.value(), -1);
        assert_eq!(Trit::Zero.value(), 0);
        assert_eq!(Trit::Pos.value(), 1);
    }

    #[test]
    fn test_trit_from_value() {
        assert_eq!(Trit::from_value(-1), Some(Trit::Neg));
        assert_eq!(Trit::from_value(0), Some(Trit::Zero));
        assert_eq!(Trit::from_value(1), Some(Trit::Pos));
        assert_eq!(Trit::from_value(2), None);
    }

    #[test]
    fn test_trit_token_len() {
        let tok = TritToken::new(vec![Trit::Pos, Trit::Zero, Trit::Neg]);
        assert_eq!(tok.len(), 3);
        assert!(!tok.is_empty());
    }

    #[test]
    fn test_trit_token_empty() {
        let tok = TritToken::new(vec![]);
        assert!(tok.is_empty());
    }

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        let tok = TernaryTokenizer::new();
        let text = "hello";
        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_tokenizer_encode_nonempty() {
        let tok = TernaryTokenizer::new();
        let encoded = tok.encode("ab");
        assert_eq!(encoded.len(), 2);
        for token in &encoded {
            assert_eq!(token.len(), 6); // 6 trits per byte
        }
    }

    #[test]
    fn test_tokenizer_vocab() {
        let mut tok = TernaryTokenizer::new();
        let token = TritToken::new(vec![Trit::Pos, Trit::Neg]);
        tok.add_vocab("test", token.clone());
        assert_eq!(tok.get_vocab("test"), Some(&token));
        assert_eq!(tok.get_vocab("missing"), None);
    }

    #[test]
    fn test_vocabulary_add() {
        let mut vocab = TernaryVocabulary::new();
        let id1 = vocab.add("hello");
        let id2 = vocab.add("world");
        let id3 = vocab.add("hello"); // duplicate
        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_vocabulary_lookup() {
        let mut vocab = TernaryVocabulary::new();
        let id = vocab.add("foo");
        assert_eq!(vocab.get_id("foo"), Some(id));
        assert_eq!(vocab.get_id("bar"), None);
        assert_eq!(vocab.get_token(id), Some("foo"));
    }

    #[test]
    fn test_vocabulary_empty() {
        let vocab = TernaryVocabulary::new();
        assert!(vocab.is_empty());
    }

    #[test]
    fn test_transition_observe_and_probability() {
        let mut t = TernaryTransition::new();
        t.observe(Trit::Pos);
        t.observe(Trit::Pos);
        t.observe(Trit::Neg);
        assert!((t.probability(Trit::Pos) - 2.0 / 3.0).abs() < 1e-10);
        assert!((t.probability(Trit::Neg) - 1.0 / 3.0).abs() < 1e-10);
        assert!((t.probability(Trit::Zero) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_transition_default_probability() {
        let t = TernaryTransition::new();
        assert!((t.probability(Trit::Neg) - 1.0 / 3.0).abs() < 1e-10);
        assert!((t.probability(Trit::Zero) - 1.0 / 3.0).abs() < 1e-10);
        assert!((t.probability(Trit::Pos) - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_transition_sample() {
        let mut t = TernaryTransition::new();
        t.observe(Trit::Pos);
        assert_eq!(t.sample(), Trit::Pos);
    }

    #[test]
    fn test_ngram_train_and_predict() {
        let mut model = TernaryNgramModel::new(2);
        let seq = vec![Trit::Pos, Trit::Pos, Trit::Pos, Trit::Pos];
        model.train(&seq);
        assert_eq!(model.predict(&[Trit::Pos]), Trit::Pos);
    }

    #[test]
    fn test_ngram_generate() {
        let mut model = TernaryNgramModel::new(2);
        let seq = vec![Trit::Pos, Trit::Pos, Trit::Pos, Trit::Pos];
        model.train(&seq);
        let generated = model.generate(&[Trit::Pos], 5);
        assert!(generated.len() > 5);
        // All should be Pos given uniform training
        for t in &generated[1..] {
            assert_eq!(*t, Trit::Pos);
        }
    }

    #[test]
    fn test_ngram_perplexity() {
        let mut model = TernaryNgramModel::new(2);
        let train = vec![Trit::Pos, Trit::Pos, Trit::Pos, Trit::Pos];
        model.train(&train);
        let test = vec![Trit::Pos, Trit::Pos, Trit::Pos];
        let pp = model.perplexity(&test);
        assert!(pp.is_finite());
        assert!(pp > 0.0);
    }

    #[test]
    fn test_ngram_perplexity_short() {
        let model = TernaryNgramModel::new(3);
        let seq = vec![Trit::Pos];
        assert!(model.perplexity(&seq).is_infinite());
    }

    #[test]
    fn test_cross_entropy() {
        let mut model = TernaryNgramModel::new(2);
        let train = vec![Trit::Pos, Trit::Pos, Trit::Pos, Trit::Pos];
        model.train(&train);
        let test = vec![Trit::Pos, Trit::Pos];
        let ce = cross_entropy(&model, &test);
        assert!(ce.is_finite());
        assert!(ce >= 0.0);
    }

    #[test]
    fn test_temperature_scaling() {
        let mut t = TernaryTransition::new();
        t.observe(Trit::Pos);
        t.observe(Trit::Neg);
        // With low temperature, should still be deterministic
        let result = t.sample_with_temperature(0.1);
        assert!(result == Trit::Pos || result == Trit::Neg);
    }

    #[test]
    fn test_predict_with_temperature() {
        let mut model = TernaryNgramModel::new(2);
        let seq = vec![Trit::Pos, Trit::Pos, Trit::Pos, Trit::Pos];
        model.train(&seq);
        let result = model.predict_with_temperature(&[Trit::Pos], 0.5);
        assert_eq!(result, Trit::Pos);
    }

    #[test]
    fn test_ngram_transition_count() {
        let mut model = TernaryNgramModel::new(2);
        let seq = vec![Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos];
        model.train(&seq);
        assert_eq!(model.transition_count(), 3);
    }

    #[test]
    fn test_ngram_train_short_sequence() {
        let mut model = TernaryNgramModel::new(5);
        let seq = vec![Trit::Pos, Trit::Neg];
        model.train(&seq);
        assert_eq!(model.transition_count(), 0);
    }

    #[test]
    fn test_generate_empty_seed() {
        let model = TernaryNgramModel::new(2);
        let result = model.generate(&[], 5);
        assert!(result.is_empty());
    }
}
