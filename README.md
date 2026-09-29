# Generating Music with LSTM Neural Networks

This repository is a research-style reimplementation and extension of the 2017 article *How to Generate Music using a LSTM Neural Network in Keras*. It reproduces the original symbolic next-event prediction approach with current TensorFlow/Keras, then develops it into a controlled experimental pipeline using MAESTRO MIDI data, reproducible preprocessing and training, and multiple autoregressive decoding strategies.

The repository preserves an original-style local baseline while maintaining a separate modernized Colab implementation used for the main experiments. The work examines preprocessing decisions, training behavior and overfitting, greedy decoding collapse, temperature-based sampling, and the limits of representing music as pitch and chord tokens. The detailed methodology and analysis are in [docs/paper.md](docs/paper.md).

```text
2017 tutorial
    -> baseline reproduction
    -> modernized implementation
    -> controlled generation experiments
    -> limitations and future research
```

## Research Motivation

The original LSTM approach remains useful for studying symbolic sequence generation, but several implementation and experimental choices are dated by current standards. This project investigates questions including:

- Can the original next-event LSTM pipeline be reproduced with current Keras and TensorFlow?
- What changes make the experiment more reproducible and scalable?
- How does piece-level splitting change the experimental setup?
- What are the practical effects of replacing one-hot targets with sparse integer targets?
- How strongly does decoding strategy affect generated sequences from a fixed model?
- What is lost when music is represented only as note and chord pitch tokens?

These questions motivate the experiments; they have not all been fully resolved.

## Original vs. Modernized Implementation

| Area | `src/` baseline | `src_colab/` modernized |
| --- | --- | --- |
| Purpose | Reproduce the original-style workflow | Main experimental implementation |
| Execution | Local / VS Code | Google Colab + GPU |
| Dataset handling | Flattened event sequence | Piece boundaries preserved |
| Validation | Sequence-level Keras split | Piece-level train/validation split |
| Targets | One-hot categorical vectors | Sparse integer classes |
| Loss | `categorical_crossentropy` | `sparse_categorical_crossentropy` |
| Checkpointing | Basic `ModelCheckpoint` | Best-model checkpoint + recovery |
| Early stopping | No | Yes |
| LR scheduling | No | `ReduceLROnPlateau` |
| Generation | Greedy only | Greedy + temperature sampling |
| Experiment metadata | Vocabulary and training config | Training config + generation JSON sidecars |

The baseline intentionally keeps flattened sequences, one-hot targets, sequence-level validation, and greedy generation. The modernized implementation is not a replacement copy; it is the experiment-oriented comparison path.

## Current Model

The modernized model predicts the next token from the previous 100 symbolic musical events:

```text
Input: (100, 1)
    |
LSTM 512, return_sequences=True
    |
Dropout 0.3
    |
LSTM 512
    |
Dropout 0.3
    |
Dense 256, ReLU
    |
Dense |V|, Softmax
```

For the completed 100-file experiment, `|V| = 1,962` and the model contains 3,787,434 parameters. Inputs remain normalized scalar token IDs, preserving the original categorical pipeline for comparison.

## Musical Representation

MIDI files are parsed with `music21`. Individual notes are stored as pitch strings:

```text
C4
F#5
A3
```

Chords are stored as dot-separated normal-order pitch classes:

```text
0.4.7
2.5.9
```

This representation does **not** preserve duration, precise inter-event timing, velocity, sustain pedal, full chord register, voicing, or inversion. Generated events are reconstructed with a fixed duration. The system therefore models pitch/event sequences rather than complete expressive piano performances.

## Data Pipeline

The main modernized pipeline is:

```text
MAESTRO MIDI
    |
music21 parsing
    |
note / chord tokens
    |
piece-level split
    |
integer vocabulary
    |
100-event sliding windows
    |
normalized scalar token IDs
    |
LSTM next-event prediction
    |
autoregressive generation
    |
MIDI reconstruction
```

In `src_colab/`, sequence windows remain within composition boundaries. Random seed 42 is used for reproducible piece splitting, seed-window selection, and controlled generation experiments.

## 100-File Training Experiment

| Metric | Value |
| --- | ---: |
| MIDI pieces | 100 |
| Musical events | 328,489 |
| Training pieces | 80 |
| Validation pieces | 20 |
| Vocabulary size | 1,962 |
| Training sequences | 249,276 |
| Validation sequences | 69,213 |
| Sequence length | 100 |
| Parameters | 3,787,434 |
| Max epochs | 50 |
| Training stopped | Epoch 15 |
| Best epoch | 8 |
| Best validation loss | 5.3470 |
| Epoch-8 train accuracy | ~2.79% |
| Epoch-8 validation accuracy | ~2.27% |

Training loss continued to decrease after epoch 8 while validation loss stopped improving, indicating overfitting under this configuration. `EarlyStopping` restored the best weights, and `ReduceLROnPlateau` reduced the learning rate at epochs 11 and 14.

Exact next-token accuracy is a 1,962-class classification measurement. It does not directly measure musical coherence or listening quality.

## Generation Study

Generation is autoregressive: each predicted token becomes part of the next input window.

```text
100-event seed
      |
predict distribution over vocabulary
      |
select next token
      |
append prediction
      |
drop oldest event
      |
repeat
```

### Greedy Decoding

Greedy decoding always selects:

```text
argmax(P(next token))
```

The observed greedy run produced 500 events containing one unique source token, a severe repetitive collapse. This is a decoding failure mode rather than proof that the model learned nothing: repeated predictions are fed back into later contexts and can reinforce the same high-probability state.

### Temperature Sampling

The modernized generator can instead sample from a temperature-adjusted distribution:

```text
q_i proportional to exp(log(p_i) / T)
```

- `T < 1` sharpens the distribution.
- `T = 1` samples from the original model distribution.
- `T > 1` flattens the distribution.

| Decoding | Unique source tokens / 500 |
| --- | ---: |
| Greedy | 1 |
| Temperature 0.5 | 126 |
| Temperature 0.8 | 182 |
| Temperature 1.0 | 233 |
| Temperature 1.2 | 269 |

Under this controlled setup, higher temperatures increased source-token diversity and generally reduced direct repetition. Increased diversity does not establish increased musical quality; no formal listening study has been conducted.

## What Was Modernized

The experiment-oriented implementation adds:

- Current TensorFlow and Keras APIs
- MAESTRO MIDI data
- Piece-level train/validation separation
- Sequence construction that preserves composition boundaries
- Sparse integer targets and sparse categorical cross-entropy
- Fixed random seeds for reproducibility
- Google Colab GPU training
- Persistent Google Drive artifacts
- Best-model checkpointing
- Early stopping and learning-rate reduction
- Training backup and recovery
- CSV metric logging
- Configurable greedy and temperature generation
- Safer reconstruction of note, chord, and single-numeric pitch-class tokens
- Saved training configuration and per-generation metadata
- Collision protection for generated experiment outputs

These additions turn the reproduction into a controlled study of the original workflow. They should not be interpreted as evidence that every modernization improves musical quality.

## Repository Structure

```text
.
|-- src/             # local, original-style baseline
|-- src_colab/       # modernized experimental implementation
|-- docs/
|   |-- paper.md     # research-style project paper
|   `-- references.md
|-- notebooks/
|-- data/
|   `-- midi/
|-- checkpoints/
|-- output/
`-- requirements.txt
```

`src/` is suitable for small local runs and comparison with the original approach. `src_colab/` contains the reproducible training and generation workflow used for the documented experiments.

## Running the Project

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

### Local Baseline

Train a small one-hot baseline. Artifacts default to `checkpoints/local/`:

```bash
python src/train.py \
  --data-dir data/midi \
  --midi-limit 10 \
  --sequence-length 100 \
  --epochs 5 \
  --batch-size 64 \
  --output-dir checkpoints/local
```

Generate with greedy decoding:

```bash
python src/generate.py \
  --model checkpoints/local/best_model.keras \
  --vocabulary checkpoints/local/vocabulary.json \
  --config checkpoints/local/training_config.json \
  --seed-midi /path/to/seed.midi \
  --output output/local/generated_music.mid \
  --num-events 500
```

### Colab / Modernized Experiment

Train the modernized model:

```bash
python src_colab/train.py \
  --data-dir /path/to/maestro-midi \
  --output-dir /path/to/training-output \
  --midi-limit 100 \
  --sequence-length 100 \
  --epochs 50 \
  --batch-size 64 \
  --validation-fraction 0.2 \
  --seed 42
```

Generate with temperature sampling:

```bash
python src_colab/generate.py \
  --model /path/to/training-output/checkpoints/best_model.keras \
  --vocabulary /path/to/training-output/vocabulary.json \
  --config /path/to/training-output/training_config.json \
  --seed-midi /path/to/seed.midi \
  --output-dir /path/to/generated-output \
  --num-events 500 \
  --strategy temperature \
  --temperature 0.8 \
  --random-seed 42
```

Without an explicit `--output`, the modernized generator creates a descriptive filename such as `temp_0.8_500.mid` plus a matching JSON metadata file. It refuses to replace existing output artifacts unless `--overwrite` is provided.

## Research Paper and Documentation

See [docs/paper.md](docs/paper.md) for the detailed research-style writeup, including background, preprocessing methodology, architecture, the 100-file training experiment, generation experiments, limitations, and future work.

The References section remains intentionally incomplete pending a separate citation-verification pass; this README does not invent bibliographic details.

## Current Limitations

- Token IDs are arbitrary categories represented as normalized scalar values.
- Rhythm, duration, expressive timing, velocity, and pedal information are discarded.
- Chords lose voicing, inversion, octave placement, and full register.
- MIDI reconstruction assigns fixed event durations.
- The main evidence comes from one 100-file model experiment.
- Generation evaluation covers limited seeds and diversity/repetition measurements.
- No formal human listening study has been performed.
- Embeddings, richer event representations, Transformers, and other architectures have not yet been compared experimentally.

## Next Research Directions

Possible next experiments include learned token embeddings, explicit duration and rhythm events, velocity and pedal modeling, richer chord/register representations, alternative sampling controls, multiple-seed generation evaluation, larger dataset runs, and architecture comparisons. These are proposed directions, not completed features.

## Technical Stack

- Python
- TensorFlow / Keras
- NumPy
- music21
- MAESTRO MIDI dataset
- Google Colab
- NVIDIA L4 GPU for the documented 100-file experiment

## Scope

This repository is a research-style reproduction and extension of a prior implementation. It is not presented as peer-reviewed work, a published paper, or a state-of-the-art music-generation system. Its current contribution is the systematic modernization and analysis of the original workflow, including reproducible preprocessing and training changes and controlled decoding experiments.
