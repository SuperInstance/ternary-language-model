# ternary-language-model

N-gram language modeling where predictions and tokens are ternary values (−1, 0, +1).

## Why This Exists

Standard language models operate over large vocabularies of discrete tokens. But there's a class of problem — signal processing, sentiment streams, ternary sensor data — where the "vocabulary" is just three symbols. This crate provides an n-gram model that learns transition probabilities over {-1, 0, +1}, predicts the next symbol given context, generates sequences, and computes perplexity and cross-entropy. It also includes a ternary tokenizer that encodes/decodes strings as balanced ternary byte representations.

## Core Concepts

- **Trit** — A single ternary value: Neg (−1), Zero (0), Pos (+1).
- **TritToken** — A sequence of trits. The tokenizer represents each byte as 6 trits in balanced ternary.
- **Balanced ternary encoding** — Each byte is decomposed into trits with place values 1, 3, 9, 27, 81, 243. The digit 2 in standard ternary is carried as a −1 with a carry to the next place: `2 = 3 − 1`, so digit becomes −1 and increment the next place.
- **TernaryNgramModel** — An n-gram model where contexts are sequences of Trits and transitions are probability distributions over {Neg, Zero, Pos}. Default probability for unseen contexts is uniform (1/3 each).
- **Perplexity** — `exp(−average_log_probability)`. Lower is better. Infinite if the sequence is shorter than n.
- **Cross-entropy** — The average negative log-probability of a test sequence under the model. The per-symbol cost in bits (well, in "trits").
- **Temperature scaling** — Higher temperature flattens the distribution; lower temperature sharpens it. Applied via `log(p) / temperature` followed by softmax normalization. The current implementation uses deterministic argmax sampling.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-language-model = "0.1"
```

```rust
use ternary_language_model::*;

fn main() {
    // Train a bigram model
    let mut model = TernaryNgramModel::new(2);
    let training: Vec<Trit> = vec![
        Trit::Pos, Trit::Pos, Trit::Pos, Trit::Neg,
        Trit::Pos, Trit::Pos, Trit::Pos, Trit::Neg,
    ];
    model.train(&training);

    // Predict
    let next = model.predict(&[Trit::Pos]);
    println!("After Pos, predict: {:?}", next); // Pos (most common)

    // Generate
    let generated = model.generate(&[Trit::Pos], 10);
    println!("Generated: {:?}", generated);

    // Measure perplexity
    let test = vec![Trit::Pos, Trit::Pos, Trit::Neg, Trit::Pos];
    println!("Perplexity: {:.2}", model.perplexity(&test));

    // Tokenizer: encode/decode text
    let tok = TernaryTokenizer::new();
    let encoded = tok.encode("hi");
    let decoded = tok.decode(&encoded);
    println!("Roundtrip: {}", decoded); // "hi"
}
```

## API Overview

| Type | Description |
|------|-------------|
| `Trit` | Core value: Neg/Zero/Pos |
| `TritToken` | Sequence of trits (used for tokenization) |
| `TernaryTokenizer` | Encodes strings to trit sequences and back |
| `TernaryVocabulary` | String ↔ u32 ID mapping for token vocabularies |
| `TernaryTransition` | Counts and probabilities for one context |
| `TernaryNgramModel` | N-gram model with train/predict/generate/perplexity |

| Function | Description |
|----------|-------------|
| `cross_entropy(model, sequence)` | Average negative log-probability of a test sequence |

## How It Works

**Training:** For each position i in the sequence, the context is `sequence[i..i+n-1]` and the target is `sequence[i+n-1]`. The context maps to a `TernaryTransition` that tallies counts per target trit.

**Prediction:** Look up the context in the transition table. Return the trit with the highest count. If the context is unseen, return Zero (the default).

**Generation:** Start from a seed context. At each step, predict the next trit and append it. The context window slides forward by one.

**Perplexity:** For each position with a valid n-gram, compute `log(P(next | context))`. Average these log-probabilities, negate, and exponentiate. A perplexity of 1.0 means perfect prediction; a perplexity of 3.0 means uniform guessing (for ternary).

**Temperature sampling:** Scale log-probabilities by `1/temperature`, apply softmax to get normalized probabilities, then take argmax. Low temperature (e.g., 0.1) concentrates probability; high temperature (e.g., 2.0) flattens it. Note: the current implementation uses deterministic argmax, not random sampling, so temperature affects which symbol wins but doesn't introduce stochasticity.

**Tokenizer encoding:** Each byte is converted to balanced ternary with 6 trits. The algorithm uses the standard balanced ternary decomposition: for each place, compute `rem = (val + 1) mod 3 − 1`. If rem is −1, the trit is Neg; if 0, Zero; if 1, Pos. The carry propagates via `val = (val − rem) / 3`. Decoding reverses this by summing `trit_value × 3^place`.

## Known Limitations

- **Deterministic sampling only.** The `sample()` and `sample_with_temperature()` methods always return the argmax. For stochastic generation (useful for text diversity or Monte Carlo methods), you'd need to add random sampling.
- **No smoothing beyond uniform default.** Unseen contexts get 1/3 probability. There's no Laplace smoothing, Kneser-Ney, or backoff to lower-order n-grams.
- **Fixed context length.** The model uses a single n. No interpolation between n-gram orders.
- **Memory grows with distinct contexts.** Every unique context sequence gets its own `TernaryTransition`. For long sequences with high n, this can get large.

## Use Cases

- **Sentiment stream prediction** — Model the flow of positive/neutral/negative sentiment in a text stream as a ternary time series.
- **Ternary signal compression** — Encode signals as ternary sequences, model them with n-grams, and use the model for prediction-based compression.
- **Game AI** — Model opponent behavior as ternary (aggressive/passive/neutral) and predict next moves.

## Ecosystem Context

Part of the SuperInstance ternary crate family. `ternary-language-model` consumes ternary data from `ternary-database`, can be driven by programs compiled with `ternary-compiler-v2`, and its outputs can be visualized with `ternary-visualization` or diffed with `ternary-diff`.

## License

MIT

## See Also
- **ternary-attention** — related
- **ternary-language** — related
- **ternary-transform** — related
- **ternary-bayesian** — related
- **ternary-entropy** — related

