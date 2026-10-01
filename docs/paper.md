# Generating Music with LSTM Neural Networks: A Modern Reimplementation and Experimental Extension

## Abstract

This project reimplements and extends a 2017 Keras-based approach to symbolic music generation with Long Short-Term Memory networks. MAESTRO MIDI files are parsed with music21, reduced to note and chord tokens, and arranged into 100-event next-token prediction examples. Two models were trained on the same 100-file dataset construction: a baseline using normalized scalar token IDs and a variant using 128-dimensional learned token embeddings. The embedding model improved best validation loss from 5.3470 to 4.72693 and validation accuracy near the best epoch from approximately 2.27% to 7.06%, although the models were not parameter matched. During autoregressive generation, both models collapsed under greedy decoding. Temperature sampling reduced repetition, but the embedding model remained more concentrated at lower temperatures and reached source-token diversity comparable to the scalar model at temperature 1.2. Objective MIDI metrics characterize diversity, repetition, and pitch movement; they do not establish musical quality. Because duration, timing, dynamics, pedal state, voicing, and other expressive information are omitted, the system is best described as a pitch/event sequence generator rather than a complete piano-performance model.

## 1. Introduction

Recurrent neural networks were widely used for sequence generation before Transformer architectures became dominant. In symbolic music generation, a recurrent model can learn from ordered musical events and predict which event should follow a fixed context.

This project revisits the 2017 article *How to Generate Music using a LSTM Neural Network in Keras*. It began as a reproduction of the original note/chord pipeline and developed into an experimental analysis of input representation, training behavior, and decoding strategy using current TensorFlow and Keras workflows.

The repository preserves two implementation tracks. The local `src/` directory remains close to the original workflow, while `src_colab/` implements piece-aware data handling, sparse targets, recoverable training, configurable scalar or embedding inputs, reproducible generation settings, and objective MIDI-level evaluation. The completed study asks whether learned embeddings improve next-token prediction and whether those improvements correspond to different autoregressive generation behavior.

The work is research-style but is not peer reviewed, published research, a novel architecture, or a state-of-the-art claim. Predictive metrics, decoding behavior, event diversity, and musical quality are treated as separate questions.

## 2. Background

### 2.1 Recurrent Neural Networks

Recurrent neural networks process ordered data while maintaining a hidden state. At timestep `t`, the current input and previous state determine the next hidden state:

```text
h_t = f(x_t, h_{t-1})
```

This recurrence makes RNNs suitable for language, time series, and symbolic music, where event order carries information.

### 2.2 LSTM Networks

Standard RNNs can struggle with long-range dependencies because gradients may vanish or grow across many timesteps. Long Short-Term Memory networks add gated cell and hidden states that regulate what information is retained, updated, and exposed. In this project, the LSTM receives 100 prior musical events and predicts the next categorical event.

### 2.3 Symbolic Music Sequence Modeling

The project operates on MIDI rather than audio waveforms. Given a token sequence, the learning task is:

```text
P(x_{t+1} | x_{t-99}, ..., x_t)
```

Each training example is:

```text
X_i = [x_i, x_{i+1}, ..., x_{i+99}]
y_i = x_{i+100}
```

The target `y_i` is one class from the observed note/chord vocabulary. During generation, predicted classes are fed back into subsequent input windows, making the process autoregressive.

### 2.4 The Original 2017 Implementation

The original implementation demonstrated a compact pipeline in which MIDI files were parsed into note and chord tokens, unique tokens were mapped to integers, fixed event windows became inputs, and an LSTM predicted the next token. Scalar integer IDs were divided by the vocabulary size before being passed to the network.

This project retains that formulation as a reproducible reference while separating historical design choices from later experimental changes.

### 2.5 Baseline and Modernized Implementations

The local `src/` pipeline is an original-style educational baseline. It concatenates pieces into a flattened event stream, uses sequence-level `validation_split=0.2`, one-hot targets, categorical cross-entropy, basic checkpointing, and greedy-only generation.

The `src_colab/` pipeline is the main experimental implementation. It preserves piece boundaries, splits complete pieces into training and validation sets, uses sparse integer targets and sparse categorical cross-entropy, supports scalar or learned-embedding inputs, and includes best-model checkpointing, `EarlyStopping`, `ReduceLROnPlateau`, `BackupAndRestore`, `CSVLogger`, temperature sampling, generation metadata, and reproducible MIDI-level evaluation.

## 3. Dataset

### 3.1 MAESTRO

The experiments use symbolic MIDI files from MAESTRO. MAESTRO contains aligned piano MIDI and audio, but this project uses only MIDI. The experimental subset contains the first 100 sorted MIDI paths selected by the preprocessing script.

### 3.2 MIDI Representation

MIDI can encode pitch, note onset and offset, velocity, timing, pedal state, and instrument information. The present tokenization uses only individual note pitches and chord pitch-class groups. It therefore discards much of the expressive information available in the source performances.

### 3.3 Dataset Scope

The shared 100-file setup contains:

| Quantity | Value |
| --- | ---: |
| MIDI pieces | 100 |
| Musical events | 328,489 |
| Training pieces | 80 |
| Validation pieces | 20 |
| Vocabulary size | 1,962 |
| Training sequences | 249,276 |
| Validation sequences | 69,213 |
| Sequence length | 100 |
| Batch size | 64 |
| Maximum epochs | 50 |
| Split/seed value | 42 |

The study has no independent held-out test set. The vocabulary is built from all selected pieces, including validation pieces. Random seed 42 controls the piece split and reported generation sampling, but TensorFlow/Keras initialization and training randomness are not fully controlled.

## 4. Data Preprocessing

### 4.1 MIDI Parsing

Each MIDI file is parsed with `music21.converter.parse()`. The extractor attempts `instrument.partitionByInstrument()` and traverses the first part when available; otherwise it uses the flattened note stream. Training and seed generation reuse the same extraction function.

### 4.2 Note Representation

Individual `music21.note.Note` objects become pitch strings:

```text
C4
F#5
A3
```

### 4.3 Chord Representation

`music21.chord.Chord` objects become dot-separated normal-order pitch classes:

```text
0.4.7
2.5.9
```

This captures pitch-class membership but not octave placement, register, voicing, inversion, duration, or rhythmic position. A single numeric token such as `"4"` is treated as a pitch-class chord token during reconstruction rather than as a note name.

### 4.4 Vocabulary Construction

The vocabulary is the sorted set of all note and chord tokens in the 100 selected pieces:

```text
V = {v_1, v_2, ..., v_1962}
```

Sorting makes the token-to-integer mapping deterministic for a fixed selected dataset. The integers are categorical identifiers, not musical measurements.

### 4.5 Sequence Construction

For each piece, overlapping 100-event windows are paired with the following event. Windows are constructed independently within each composition and never cross piece boundaries. Entire pieces are split 80/20 before their windows are generated.

### 4.6 Scalar Input Representation

The scalar experiment follows the original-style encoding:

```text
integer token ID
-> float32
-> divide by |V|
-> shape (samples, 100, 1)
```

For a token ID `x`, the input value is `x / |V|`. This scales values but retains an artificial ordering: IDs 400 and 401 are numerically close even though their tokens need not be musically similar.

### 4.7 Embedding Input Representation

The embedding experiment keeps token IDs as `int32` arrays with shape `(samples, 100)` and passes them through a learned table:

```text
token ID -> trainable 128-dimensional vector
```

The embedding output for one sample has shape `(100, 128)`. Unlike scalar normalization, embeddings can learn task-relevant relationships among categories from training data. The learned dimensions are not assumed to correspond to interpretable musical concepts.

### 4.8 Sparse Target Encoding

Targets remain `int32` class IDs for both experiments. The final softmax has 1,962 classes, and training uses sparse categorical cross-entropy. Sparse targets avoid allocating a 1,962-element one-hot vector for every example.

### 4.9 Representation Limitations

Neither input representation restores information removed during tokenization. The model does not preserve duration, precise inter-event timing, velocity, sustain pedal, rests, articulation, expressive timing, full chord register, voicing, or inversion. Scalar and embedding experiments therefore operate on the same reduced pitch/event sequence.

## 5. Model Architectures

### 5.1 Scalar-Input LSTM

```text
Input (100, 1), normalized float32 IDs
-> LSTM 512, return_sequences=True
-> Dropout 0.3
-> LSTM 512
-> Dropout 0.3
-> Dense 256, ReLU
-> Dense 1962, Softmax
```

The scalar model contains 3,787,434 parameters.

### 5.2 Embedding-Input LSTM

```text
Input (100), int32 IDs
-> Embedding(1962, 128)
-> LSTM 512, return_sequences=True
-> Dropout 0.3
-> LSTM 512
-> Dropout 0.3
-> Dense 256, ReLU
-> Dense 1962, Softmax
```

The embedding table contains `1962 x 128 = 251,136` parameters. The complete embedding model contains 4,298,666 parameters. It has greater capacity than the scalar model because it adds the embedding table and widens the first LSTM input from one feature to 128. The comparison is therefore representation-focused but not parameter matched.

Both models use Adam, sparse categorical cross-entropy, and accuracy.

## 6. Experiment 1: Scalar Baseline

### 6.1 Training Setup

The scalar run used the shared 100-file dataset, 100-event contexts, batch size 64, and a maximum of 50 epochs. Model selection monitored validation loss. Training used best-model checkpointing, early stopping with best-weight restoration, learning-rate reduction, CSV logging, and backup/recovery.

### 6.2 Training Results

| Metric | Result |
| --- | ---: |
| Best epoch | 8 |
| Best validation loss | 5.3470 |
| Training accuracy at epoch 8 | ~2.79% |
| Validation accuracy at epoch 8 | ~2.27% |
| Training stopped | Epoch 15 |

Training loss continued decreasing after epoch 8 while validation loss stopped improving, indicating overfitting under this configuration. Learning-rate reductions occurred at epochs 11 and 14. Exact next-token accuracy across 1,962 classes does not directly measure musical quality.

### 6.3 Decoding Results

Greedy decoding generated one unique source token across 500 events. Temperature sampling increased source-token diversity to 126, 182, 233, and 269 at temperatures 0.5, 0.8, 1.0, and 1.2 respectively.

## 7. Experiment 2: Learned Embeddings

### 7.1 Motivation

Scalar token IDs encode arbitrary categorical labels as ordered numeric values. Learned embeddings replace each ID with a trainable dense vector, allowing similarity useful to next-event prediction to be learned rather than imposed by integer ordering.

### 7.2 Architecture Change

The only intended representation change was the addition of a 128-dimensional embedding and the corresponding rank-2 integer input. Dataset selection, vocabulary, sequence length, LSTM widths, dropout, dense layers, optimizer, loss, callbacks, and decoding implementation remained shared. Parameter count increased from 3,787,434 to 4,298,666, so the ablation was not capacity matched.

### 7.3 Training Results

| Metric | Result |
| --- | ---: |
| Embedding dimension | 128 |
| Best epoch | 7 |
| Best validation loss | 4.72693 |
| Validation accuracy at epoch 7 | ~7.06% |
| Training stopped | Epoch 14 |

By epoch 14, training accuracy reached approximately 17.15% and validation accuracy approximately 7.97%, but validation loss had already worsened relative to epoch 7. Epoch 14 validation accuracy is therefore not presented as the best-checkpoint metric. The divergence is consistent with overfitting after the best validation-loss epoch.

### 7.4 Decoding Results

Greedy decoding again produced one unique source token in 500 events. Temperature sampling produced 25, 65, 177, and 272 unique source tokens at temperatures 0.5, 0.8, 1.0, and 1.2. The embedding output was substantially more concentrated at lower temperatures than the scalar output under this seed condition.

## 8. Comparative Analysis

### 8.1 Predictive Performance

| Metric | Scalar | Embedding-128 |
| --- | ---: | ---: |
| Best validation loss | 5.3470 | 4.72693 |
| Validation accuracy near best epoch | ~2.27% | ~7.06% |
| Best epoch | 8 | 7 |
| Total parameters | 3,787,434 | 4,298,666 |
| Early stopping epoch | 15 | 14 |

The embedding model reduced best validation loss by `5.3470 - 4.72693 = 0.62007` and increased validation accuracy near the selected checkpoint from approximately 2.27% to 7.06%. Learned embeddings therefore substantially improved next-token predictive performance under this setup. Because parameter count also increased and only one primary run per representation is reported, this is not a controlled parameter-matched architecture comparison.

### 8.2 Source-Token Diversity

Generation used the same seed MIDI, selected seed position 912, random seed 42, 500 generated events, model vocabulary, and generation logic.

Generation is autoregressive. The model receives a 100-event seed, predicts a distribution over the vocabulary, appends the selected token, removes the oldest event, and repeats with the updated context. Each prediction therefore influences later inputs, so errors or repetitive states can compound over time.

Greedy decoding selects `argmax` from the predicted distribution. Temperature sampling instead transforms class probabilities `p_i` using:

```text
q_i = exp(log(p_i) / T) / sum_j exp(log(p_j) / T)
```

Temperatures below 1 sharpen the distribution, temperature 1 samples from the original distribution, and temperatures above 1 flatten it. Higher temperature changes the concentration/diversity tradeoff; it is not inherently better.

| Decoding | Scalar | Embedding-128 |
| --- | ---: | ---: |
| Greedy | 1 | 1 |
| T=0.5 | 126 | 25 |
| T=0.8 | 182 | 65 |
| T=1.0 | 233 | 177 |
| T=1.2 | 269 | 272 |

Both models collapsed under greedy decoding. Embedding sampling was more concentrated at lower temperatures, but the gap narrowed as temperature increased. At `T=1.2`, source-token diversity was nearly equal.

### 8.3 MIDI-Level Metrics

`src_colab/evaluate_generation.py` parses generated MIDI and canonicalizes notes as integer MIDI pitches and chords as sorted tuples of MIDI pitches. It reports total events, unique patterns and ratio, adjacent repeats and rate, longest identical run, and mean absolute pitch jump. Chord pitch jumps use the arithmetic mean of each chord's MIDI pitches.

Embedding results:

| Decoding | Unique patterns | Adjacent repeats | Longest run | Mean pitch jump |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 1 | 499 | 500 | 0.00 |
| T=0.5 | 25 | 319 | 44 | 0.89 |
| T=0.8 | 64 | 187 | 18 | 1.75 |
| T=1.0 | 169 | 21 | 3 | 6.41 |
| T=1.2 | 264 | 9 | 4 | 6.32 |

Scalar results:

| Decoding | Unique patterns | Adjacent repeats | Longest run | Mean pitch jump |
| --- | ---: | ---: | ---: | ---: |
| T=0.5 | 119 | 52 | 6 | ~3.67 |
| T=0.8 | 172 | 16 | 2 | ~8.07 |
| T=1.0 | 221 | 3 | 2 | ~8.65 |
| T=1.2 | 258 | 1 | 2 | ~6.45 |

No scalar greedy MIDI-level value is added because it was not part of the earlier documented MIDI table.

These measurements characterize event diversity, repetition, and pitch movement. They do not directly evaluate harmony, melody, long-range form, stylistic quality, listener preference, or musicality.

### 8.4 Interpretation

Learned embeddings improved predictive performance but did not eliminate autoregressive decoding failure. Greedy decoding collapsed for both models. At lower temperatures, the embedding model's sampled output was more repetitive and less diverse, consistent with a sharper or more concentrated effective output distribution. This is an interpretation of observed behavior rather than a directly proven causal mechanism. Increasing temperature reduced the concentration, and at temperature 1.2 the embedding model reached diversity comparable to the scalar model.

Predictive accuracy and generative diversity are distinct properties. Input representation and decoding strategy should be evaluated as separate components of an autoregressive system, and neither metric alone establishes musical quality.

## 9. Limitations

- The dataset is the first 100 sorted MIDI paths, not a representative random sample or the official MAESTRO split.
- The study uses an 80/20 training-validation split with no independent held-out test set.
- The vocabulary is built from all selected pieces, including validation pieces.
- Scalar and embedding models are not parameter matched.
- One primary training run per representation is reported, without statistical testing across repeated runs.
- TensorFlow/Keras initialization and training randomness are not fully controlled by the current seed handling.
- The generation comparison uses one main seed condition at selected position 912.
- No formal human listening evaluation was conducted.
- Diversity, repetition, and pitch-jump metrics do not measure musical quality.
- The representation omits duration, rhythm, velocity, pedal state, expressive timing, rests, chord voicing, inversion, and full register.
- Chords are reconstructed around a fixed register, and all generated events receive fixed duration.
- Training scale is limited to 100 files.
- No Transformer, attention-based, or other model-family comparison was completed.

## 10. Future Work

Possible extensions include using official MAESTRO train/validation/test splits, running multiple independent training trials, building vocabulary from training pieces with unknown-token handling, adding duration/rhythm/velocity/pedal events, testing richer note and chord encodings, performing a parameter-matched embedding comparison, evaluating multiple generation seeds, conducting human listening studies, and comparing attention or Transformer models. These are optional continuations after the completed study, not claims about implemented work.

## 11. Conclusion

The project evolved from a reproduction of a 2017 LSTM/Keras tutorial into an experimental analysis of input representation, training behavior, and decoding strategy. Learned embeddings substantially improved next-event prediction under the tested setup, reducing best validation loss by 0.62007 and increasing validation accuracy near the selected checkpoint. However, decoding remained a separate bottleneck: greedy generation collapsed for both models, while temperature sampling controlled the observed diversity/repetition tradeoff.

The embedding model was more concentrated and repetitive at lower temperatures but reached source-token and MIDI-pattern diversity comparable to the scalar model at temperature 1.2. Better predictive metrics therefore did not imply better generation under every decoding strategy. The strongest conclusion is that representation and decoding must be assessed separately, and neither objective sequence metric justifies the claim of objectively better music.