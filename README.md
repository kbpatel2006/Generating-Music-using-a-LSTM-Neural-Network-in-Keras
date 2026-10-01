# Generating Music with LSTM Neural Networks: A Modern Reimplementation and Experimental Extension

This repository reproduces the symbolic next-event LSTM pipeline from the 2017 article *How to Generate Music using a LSTM Neural Network in Keras* and extends it into a modern experimental study. Using MAESTRO MIDI data, the project modernizes preprocessing, training, autoregressive generation, and objective MIDI-level evaluation while retaining a local original-style baseline for comparison.

The completed study compares normalized scalar token IDs with 128-dimensional learned token embeddings and compares greedy decoding with temperature sampling. Learned embeddings improved next-event predictive metrics under the tested setup, but both models collapsed under greedy generation, and their diversity/repetition behavior depended strongly on temperature. These findings concern prediction and observable sequence behavior, not objective musical quality. Full methodology and analysis are in [docs/paper.md](docs/paper.md).

```text
2017 tutorial
    |
baseline reproduction
    |
modernized scalar experiment
    |
decoding analysis
    |
embedding ablation
    |
scalar vs. embedding comparison
```

## Final Findings

- Learned embeddings reduced best validation loss from `5.3470` to `4.72693` and increased validation accuracy near the best epoch from approximately `2.27%` to `7.06%`.
- The comparison is representation-focused but not parameter matched: the scalar model has 3,787,434 parameters and the embedding model has 4,298,666.
- Greedy decoding collapsed to one repeated source token for both models.
- At lower temperatures, embedding outputs were more concentrated and repetitive than scalar outputs under the reported seed condition.
- At `T=1.2`, source-token diversity was nearly equal: 269 unique scalar tokens and 272 unique embedding tokens out of 500 events.
- Better next-token prediction did not automatically produce less repetitive autoregressive generation.

## Baseline vs. Modernized Implementation

| Area | `src/` baseline | `src_colab/` modernized |
| --- | --- | --- |
| Purpose | Original-style local reproduction | Main experimental pipeline |
| Execution | Local / VS Code | Google Colab + GPU |
| Dataset handling | Flattened event sequence | Piece boundaries preserved |
| Validation | Sequence-level Keras split | Piece-level train/validation split |
| Targets | One-hot vectors | Sparse integer classes |
| Loss | `categorical_crossentropy` | `sparse_categorical_crossentropy` |
| Input | Normalized scalar IDs | Scalar IDs or learned embeddings |
| Checkpointing | Basic `ModelCheckpoint` | Best checkpoint + recovery |
| Training controls | Fixed local loop | Early stopping + LR reduction |
| Generation | Greedy only | Greedy + temperature sampling |
| Metadata | Minimal training artifacts | Training config + generation sidecars |
| Evaluation | Manual inspection | Reproducible MIDI metric script |

## Musical Representation

`music21` parses MIDI into individual notes represented by pitch strings:

```text
C4
F#5
A3
```

Chords are represented by dot-separated normal-order pitch classes:

```text
0.4.7
2.5.9
```

The representation omits duration, precise rhythm and timing, velocity, sustain pedal, chord voicing, inversion, full register, and expressive timing. Reconstruction assigns a fixed duration and places pitch-class chords around a fixed register. The system is therefore a **pitch/event sequence generator**, not a full expressive piano-performance model.

## Experimental Setup

Both 100-file experiments used the same dataset construction and training setup:

| Setting | Value |
| --- | ---: |
| MIDI files | 100 |
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

There is no independent held-out test set. The vocabulary was built from all 100 selected pieces, including validation pieces.

## Scalar Experiment

```text
integer token ID
    -> float32
    -> divide by vocabulary size
    -> shape (100, 1)
    -> LSTM 512
    -> Dropout 0.3
    -> LSTM 512
    -> Dropout 0.3
    -> Dense 256, ReLU
    -> Softmax |V|
```

| Metric | Scalar result |
| --- | ---: |
| Parameters | 3,787,434 |
| Best epoch | 8 |
| Best validation loss | 5.3470 |
| Train accuracy at best epoch | ~2.79% |
| Validation accuracy at best epoch | ~2.27% |
| Training stopped | Epoch 15 |

## Embedding Experiment

```text
integer token ID
    -> Embedding(|V|, 128)
    -> shape (100, 128)
    -> LSTM 512
    -> Dropout 0.3
    -> LSTM 512
    -> Dropout 0.3
    -> Dense 256, ReLU
    -> Softmax |V|
```

| Metric | Embedding-128 result |
| --- | ---: |
| Embedding dimension | 128 |
| Embedding parameters | 251,136 |
| Total parameters | 4,298,666 |
| Best epoch | 7 |
| Best validation loss | 4.72693 |
| Validation accuracy at best epoch | ~7.06% |
| Training stopped | Epoch 14 |

The embedding model has greater capacity because it adds an embedding table and increases the input width of the first LSTM. This is not a perfectly parameter-matched ablation.

## Predictive Comparison

| Metric | Scalar | Embedding-128 |
| --- | ---: | ---: |
| Best validation loss | 5.3470 | 4.7269 |
| Validation accuracy near best epoch | ~2.27% | ~7.06% |
| Best epoch | 8 | 7 |
| Total parameters | 3,787,434 | 4,298,666 |
| Early stopping epoch | 15 | 14 |

The learned representation substantially improved next-event predictive performance in this setup. That result does not, by itself, demonstrate better generated music.

## Generation Comparison

The controlled comparison used the same seed MIDI, selected seed position 912, random seed 42, 500 generated events, model vocabulary, and generation logic.

| Decoding | Scalar | Embedding-128 |
| --- | ---: | ---: |
| Greedy | 1 | 1 |
| Temperature 0.5 | 126 | 25 |
| Temperature 0.8 | 182 | 65 |
| Temperature 1.0 | 233 | 177 |
| Temperature 1.2 | 269 | 272 |

Both models collapsed under greedy decoding. At lower temperatures, embedding output was more concentrated and repetitive. Differences narrowed as temperature increased, and embedding diversity was comparable to scalar diversity at `T=1.2`. This is an interpretation of the observed behavior, not proof of a general causal mechanism or musical superiority.

## MIDI-Level Evaluation

[src_colab/evaluate_generation.py](src_colab/evaluate_generation.py) parses generated MIDI and reports total events, unique MIDI patterns and ratio, adjacent repeats and rate, longest identical run, and mean absolute pitch jump in semitones. Notes use their MIDI pitch; chords use sorted MIDI-pitch tuples, with mean chord pitch used for pitch-jump calculations.

### Embedding-128 Outputs

| Decoding | Unique MIDI patterns | Adjacent repeats | Longest run | Mean pitch jump |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 1 | 499 | 500 | 0.00 |
| T=0.5 | 25 | 319 | 44 | 0.89 |
| T=0.8 | 64 | 187 | 18 | 1.75 |
| T=1.0 | 169 | 21 | 3 | 6.41 |
| T=1.2 | 264 | 9 | 4 | 6.32 |

### Scalar Outputs

| Decoding | Unique MIDI patterns | Adjacent repeats | Longest run | Mean pitch jump |
| --- | ---: | ---: | ---: | ---: |
| T=0.5 | 119 | 52 | 6 | ~3.67 |
| T=0.8 | 172 | 16 | 2 | ~8.07 |
| T=1.0 | 221 | 3 | 2 | ~8.65 |
| T=1.2 | 258 | 1 | 2 | ~6.45 |

These metrics characterize event diversity, repetition, and pitch movement. They do not directly measure harmony, melody, long-range form, stylistic quality, listener preference, or musicality.

## Repository Structure

```text
.
|-- src/                         # local original-style baseline
|-- src_colab/                   # main experimental pipeline
|   |-- preprocess.py
|   |-- dataset.py
|   |-- model.py
|   |-- train.py
|   |-- generate.py
|   `-- evaluate_generation.py  # objective MIDI metrics
|-- docs/
|   `-- paper.md                 # detailed research-style writeup
|-- data/
|   `-- midi/
|-- checkpoints/
|-- output/
`-- requirements.txt
```

## Running the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Local Baseline

```bash
python src/train.py \
  --data-dir data/midi \
  --midi-limit 10 \
  --sequence-length 100 \
  --epochs 5 \
  --batch-size 64 \
  --output-dir checkpoints/local

python src/generate.py \
  --model checkpoints/local/best_model.keras \
  --vocabulary checkpoints/local/vocabulary.json \
  --config checkpoints/local/training_config.json \
  --seed-midi /path/to/seed.midi \
  --output output/local/generated_music.mid
```

### Scalar Colab Experiment

```bash
python src_colab/train.py \
  --data-dir /path/to/maestro-midi \
  --output-dir /path/to/scalar-run \
  --midi-limit 100 \
  --sequence-length 100 \
  --epochs 50 \
  --batch-size 64 \
  --validation-fraction 0.2 \
  --seed 42 \
  --input-representation scalar
```

### Embedding-128 Experiment

```bash
python src_colab/train.py \
  --data-dir /path/to/maestro-midi \
  --output-dir /path/to/embedding-128-run \
  --midi-limit 100 \
  --sequence-length 100 \
  --epochs 50 \
  --batch-size 64 \
  --validation-fraction 0.2 \
  --seed 42 \
  --input-representation embedding \
  --embedding-dim 128
```

### Generation

The input representation is inferred from `training_config.json`.

```bash
python src_colab/generate.py \
  --model /path/to/run/checkpoints/best_model.keras \
  --vocabulary /path/to/run/vocabulary.json \
  --config /path/to/run/training_config.json \
  --seed-midi /path/to/seed.midi \
  --output-dir /path/to/generated-output \
  --num-events 500 \
  --strategy temperature \
  --temperature 1.0 \
  --random-seed 42
```

### Evaluation

```bash
python src_colab/evaluate_generation.py \
  generated/file1.mid \
  generated/file2.mid \
  --output-json results.json \
  --output-csv results.csv
```

## Limitations

- The experiments use the first 100 sorted MAESTRO MIDI paths rather than a representative random sample.
- The 80/20 split provides training and validation sets only; there is no independent test set.
- The vocabulary is built from all selected pieces, including validation pieces.
- The scalar and embedding models are not parameter matched.
- The reported generation comparison uses one primary seed at position 912.
- No formal listening study or statistical test across repeated training runs was conducted.
- Diversity and repetition metrics do not measure musical quality.
- Duration, rhythm, velocity, pedal state, voicing, inversion, and full register are not modeled.
- MIDI reconstruction uses fixed durations and simplified chord placement.
- TensorFlow/Keras initialization and training randomness are not fully controlled by the current seed handling.
- No Transformer, attention-based, or other model-family comparison was completed.

## Optional Future Work

Possible extensions include official MAESTRO train/validation/test splits, multiple independent training runs, a train-only vocabulary with unknown-token handling, richer duration/rhythm/velocity/pedal events, parameter-matched representation comparisons, multiple generation seeds, human listening evaluation, and attention or Transformer baselines.

## Scope

This repository is a completed research-style reproduction and experimental extension of a prior implementation. It is not presented as peer-reviewed work, published research, a novel architecture, or a state-of-the-art music generator. Its contribution is a documented analysis of input representation, training behavior, and decoding strategy within a shared symbolic LSTM pipeline.
