# Generating Music with LSTM Neural Networks: A Modern Reimplementation

## Abstract

This project is a modern reimplementation of a 2017 Keras-based approach to symbolic music generation with Long Short-Term Memory networks. MIDI files from the MAESTRO dataset are parsed with music21, reduced to note and chord tokens, converted into integer identifiers, and arranged into fixed-length next-event prediction examples. A trained LSTM then generates new event sequences autoregressively from a seed MIDI file. Controlled decoding experiments show that greedy selection collapses into repetition, while temperature sampling increases generated-token diversity and reduces direct repetition. These observations describe decoding behavior, not musical quality, which has not yet been formally evaluated. The current system is best understood as a pitch/event sequence generator because its representation omits timing, dynamics, articulation, and other expressive performance information.

## 1. Introduction

Recurrent neural networks were a common choice for sequence generation tasks before Transformer architectures became dominant. One application was symbolic music generation, where a model learns patterns from a sequence of musical events and predicts the event that should follow.

This project revisits the 2017 *Medium* article *How to Generate Music using a LSTM Neural Network in Keras*. The purpose is not to claim a new state-of-the-art result, but to reproduce the original pipeline clearly enough to understand its behavior, then use that understanding as a foundation for future improvements.

The project now includes preprocessing, LSTM training, and MIDI generation. The generation study compares greedy decoding with temperature-based probabilistic sampling under controlled conditions. Its reported metrics concern token variety, MIDI onset-pattern variety, adjacent repetition, identical-run length, and pitch movement. These measurements do not establish whether any output is musically coherent, expressive, or preferable to another.

## 2. Background

### 2.1 Recurrent Neural Networks

Recurrent neural networks process ordered data by maintaining a hidden state across timesteps. At each timestep, the network receives the current input and an internal state derived from previous inputs.

For a sequence `(x_1, x_2, ..., x_t)`, the recurrence can be summarized as:

```text
h_t = f(x_t, h_{t-1})
```

where `h_t` is the hidden state at timestep `t`. This structure makes RNNs suitable for data where order matters, including language, time series, and symbolic music.

### 2.2 LSTM Networks

Standard RNNs can struggle with long-range dependencies because gradients may vanish or grow during training across many timesteps. Long Short-Term Memory networks address this problem with gates that regulate how information is stored, updated, and exposed.

An LSTM maintains a cell state as well as a hidden state. Its input, forget, and output gates allow the network to retain useful information across longer contexts than a simple RNN usually can. In this project, the intended LSTM task is to receive a sequence of previous musical events and predict the next event.

### 2.3 Sequence Modeling for Music

This project treats music as symbolic sequence data rather than raw audio. MIDI files do not contain waveform samples like WAV or MP3 files. They contain structured musical events such as pitches, timings, instruments, and note activity.

The current preprocessing stage simplifies MIDI into a single sequence of note and chord tokens. Given the previous 100 events, the learning task is next-event prediction:

```text
P(x_{t+1} | x_{t-99}, ..., x_t)
```

Equivalently, for each training example:

```text
X_i = [x_i, x_{i+1}, ..., x_{i+99}]
y_i = x_{i+100}
```

The target `y_i` is not a continuous value. It is a categorical class selected from the vocabulary of observed note and chord tokens.

### 2.4 The Original 2017 Implementation

The original 2017 implementation demonstrated a compact symbolic music generation pipeline using Keras and music21. MIDI files were parsed into notes and chords, unique events were mapped to integers, fixed-length event windows were used as model inputs, and an LSTM was trained to predict the next event.

This project initially keeps the same general framing so the reproduction remains understandable. Later stages can then evaluate which parts should change, including the event representation, the input encoding, the model architecture, the training procedure, and the generation strategy.

## 3. Dataset

### 3.1 MAESTRO

This project uses MAESTRO MIDI data as the source material. MAESTRO contains piano performances with aligned audio and MIDI data. The current implementation uses the symbolic MIDI files only; it does not use the audio recordings.

MAESTRO is larger than needed for early pipeline testing. During initial debugging, the training script was limited to small file subsets so that parsing and sequence generation remained manageable. The main completed experiment described in Section 6 uses 100 MIDI files.

### 3.2 MIDI Representation

MIDI is a symbolic representation of music. It stores performance events rather than acoustic waveforms. A MIDI file can represent note onsets, note offsets, pitch values, velocity, timing, and instrument-related information.

The current pipeline does not use every feature available in MIDI. It extracts individual note pitches and chord pitch-class groups, then stores them as string tokens. This gives the model a simple event stream to learn from, while leaving rhythm, duration, velocity, rests, and more detailed performance information for later work.

### 3.3 Scope of the Current Experiment

The current preprocessing script searches recursively under `data/midi/` for `.midi` files:

```python
midi_files = glob.glob(
    "data/midi/**/*.midi",
    recursive=True
)
```

The training command retains a configurable MIDI limit and defaults to ten files for quick development runs. The preprocessing module applies the requested limit after sorting the discovered paths:

```python
midi_files = midi_files[:limit]
```

The ten-file default is an early debugging convenience, not the scale of the main completed training experiment. The 100-file run overrides this limit.

## 4. Data Preprocessing

The preprocessing pipeline converts MIDI files into numerical arrays suitable for an LSTM-style next-event prediction model.

The current transformation is:

```text
MIDI file
→ music21 objects
→ note/chord tokens
→ integer IDs
→ 100-event sliding windows
→ 3D LSTM tensors
→ normalized inputs
→ sparse integer targets
```

The implemented Colab pipeline is split across `src_colab/preprocess.py` and `src_colab/dataset.py`. The first file handles MIDI discovery, parsing, and token extraction while keeping pieces separate. The second file builds the vocabulary, creates training windows within piece boundaries, reshapes model inputs, normalizes input values, and stores targets as sparse integer class IDs.

### 4.1 MIDI Parsing

The first step is to parse each MIDI file into music21 objects:

```python
midi = converter.parse(file)
```

music21 converts the MIDI file into Python objects that can be traversed and inspected. This matters because the downstream code needs symbolic objects such as `note.Note` and `chord.Chord`, not raw MIDI bytes.

After parsing, the script attempts to partition the file by instrument:

```python
parts = instrument.partitionByInstrument(midi)

if parts:
    notes_to_parse = parts.parts[0].recurse()
else:
    notes_to_parse = midi.flatten().notes
```

If instrument parts are available, the current implementation processes the first part recursively. If music21 does not return instrument parts, the script flattens the MIDI structure and extracts notes from the flattened stream. Since MAESTRO is piano-oriented, this simplified handling is acceptable for the current baseline, but it is still a modeling choice.

At this stage, the data changes from files on disk into iterable symbolic music objects:

```text
.midi file → music21 stream → note/chord elements
```

### 4.2 Note and Chord Representation

The current representation keeps two event types: individual notes and chords.

Individual notes are detected as `music21.note.Note` objects and stored by pitch name:

```python
if isinstance(element, note.Note):
    notes.append(str(element.pitch))
```

Example note tokens include:

```text
C4
F#5
A3
```

Chords are detected as `music21.chord.Chord` objects. The code stores each chord as a dot-separated sequence of normal-order pitch classes:

```python
elif isinstance(element, chord.Chord):
    notes.append(
        ".".join(str(n) for n in element.normalOrder)
    )
```

For example, a C major triad may be represented as:

```text
0.4.7
```

The pitch-class numbers range from `0` to `11`, where `0` corresponds to C, `1` to C#/Db, and so on up to `11` for B. This chord representation captures pitch-class content, but it does not preserve octave placement, voicing, duration, or rhythmic position.

After this step, the musical data is a one-dimensional sequence of categorical string tokens:

```python
[
    "C4",
    "E4",
    "G4",
    "0.4.7",
    "F#5",
]
```

### 4.3 Vocabulary Construction

The vocabulary is the set of unique note and chord tokens observed in the parsed data:

```text
V = {v_1, v_2, ..., v_n}
```

In code, the vocabulary is built with `sorted(set(notes))`:

```python
def build_vocabulary(notes):
    pitchnames = sorted(set(notes))

    note_to_int = {
        note_name: number
        for number, note_name in enumerate(pitchnames)
    }

    return pitchnames, note_to_int
```

Sorting makes the mapping deterministic for a given token set. The resulting vocabulary size is:

```text
|V| = number of unique note/chord tokens
```

An early debugging run produced approximately 741 unique tokens. This historical value depends on the selected subset and is not the vocabulary size of the main 100-file experiment, which is reported in Section 6.1.

### 4.4 Integer Encoding

Neural network inputs cannot be passed as arbitrary strings, so each vocabulary token is mapped to an integer ID:

```python
note_to_int = {
    note_name: number
    for number, note_name in enumerate(pitchnames)
}
```

This transforms a token sequence such as:

```text
["C4", "E4", "G4", "0.4.7"]
```

into a sequence of integer identifiers:

```text
[13, 42, 57, 3]
```

The specific numbers are categorical identifiers. They do not mean that token `57` is musically greater than token `42`, or that the distance between IDs is musically meaningful. This distinction matters because the current implementation later normalizes these IDs as scalar values.

### 4.5 Sequence Generation

The model task is constructed as next-event prediction. With a sequence length of 100, each input contains 100 consecutive integer-encoded events, and the target is the event immediately after that window:

```text
X_i = [x_i, x_{i+1}, ..., x_{i+99}]
y_i = x_{i+100}
```

The implementation creates these windows in `create_sequences`:

```python
for i in range(len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append([note_to_int[note] for note in sequence_in])
    network_output.append(note_to_int[sequence_out])
```

For example:

```text
notes[0:100] → notes[100]
notes[1:101] → notes[101]
notes[2:102] → notes[102]
```

This creates overlapping training examples. Overlap is expected in sequence modeling because each adjacent window provides a slightly shifted context.

The Colab implementation performs this operation independently for each piece. It therefore does not create windows that join the end of one composition to the beginning of another, and it appends each input-target pair once.

### 4.6 Input Reshaping

Keras recurrent layers expect inputs in the form:

```text
(samples, timesteps, features)
```

For this project:

```text
timesteps = 100
features = 1
```

The preprocessing code reshapes the list of encoded windows into a three-dimensional array:

```python
n_patterns = len(network_input)
network_input = np.reshape(
    network_input,
    (n_patterns, len(network_input[0]), 1)
)
```

The representation changes from a two-dimensional list:

```text
(samples, 100)
```

to an LSTM-compatible tensor:

```text
(samples, 100, 1)
```

An early debugging run produced an input shape of approximately:

```text
(42446, 100, 1)
```

That historical value reflects a smaller development subset. It is not a result from the main 100-file experiment and is included only to document early pipeline verification.

### 4.7 Input Normalization

The current implementation normalizes input IDs by dividing by the vocabulary size:

```python
network_input = network_input / float(n_vocab)
```

For an integer token ID `x`, the normalized value is:

```text
x' = x / |V|
```

This scales input values into a smaller numeric range, roughly between 0 and 1. The step follows the style of the original pipeline and produces floating-point inputs for the LSTM.

However, this normalization does not make the token IDs semantically continuous. The IDs are category labels, not measurements. Dividing them by `|V|` preserves an artificial ordering and distance between tokens. For example, two IDs that are numerically close are not necessarily musically similar. This is a limitation of the current representation and one motivation for considering embeddings in a later version.

### 4.8 Target Encoding

The target for each training example is the next event after the 100-event input window. Since the target is one of the vocabulary tokens, this is a multiclass classification problem over `|V|` possible classes.

The target remains an integer class ID:

```python
network_output = np.asarray(network_output, dtype=np.int32)
```

The model uses sparse categorical cross-entropy, so allocating a full one-hot vector for every target is unnecessary. The softmax output still represents a probability distribution over all `|V|` classes, but each training target is stored as one integer rather than a length-`|V|` vector.

### 4.9 Current Preprocessing Pipeline

The completed preprocessing stage performs the following operations:

1. Recursively search `data/midi/` for `.midi` files.
2. Apply the configurable file limit, which defaults to ten files during development.
3. Parse each MIDI file with music21 and retain each piece as a separate event sequence.
4. Attempt to partition each parsed file by instrument and traverse the first part, or use flattened notes when no parts are returned.
5. Store individual notes as pitch strings such as `C4`, `F#5`, and `A3`.
6. Store chords as dot-separated normal-order pitch classes such as `0.4.7`.
7. Build a deterministic vocabulary from all selected pieces.
8. Split complete pieces into training and validation sets using a fixed seed.
9. Generate 100-event next-event windows without crossing piece boundaries.
10. Reshape inputs into `(samples, 100, 1)` and divide token IDs by the vocabulary size.
11. Retain target events as sparse integer class IDs.

At the end of this stage, the project has numerical inputs and categorical targets prepared for LSTM training while preserving composition boundaries in both the split and window construction.

### 4.10 Current Limitations and Modernization Opportunities

The current preprocessing pipeline is intentionally close to the original 2017 approach. That makes it useful as a reproduction baseline, but several limitations should be addressed before treating it as a modern representation of symbolic music.

First, the integer token IDs are arbitrary categorical labels. The sorted vocabulary provides deterministic IDs, but it does not provide musical distances. Normalizing those IDs with `x' = x / |V|` creates scalar values that imply an artificial numeric relationship between categories. A future implementation could represent events with learned embeddings, allowing the model to learn a dense representation instead of receiving normalized category IDs as if they were measurements.

Second, the event representation omits several musically important features. Durations, note offsets, velocity, rests, rhythm, tempo context, articulation, and expressive timing are not currently modeled. This reduces the task to predicting pitch and chord-event tokens, which is simpler than generating complete musical performances.

Third, the chord representation is simplified. Normal-order pitch classes such as `0.4.7` capture the pitch classes in a chord, but not voicing, inversion, octave placement, register, duration, or onset context. Two musically different chord events can therefore collapse into the same token.

Finally, the current instrument handling uses only the first partitioned part when `instrument.partitionByInstrument` succeeds. That reproduces a simple extraction strategy, but future work should decide explicitly how to handle multiple parts, piano hands, tracks, and any non-piano material.

## 5. Model Architecture

The implemented Keras model receives inputs with shape `(100, 1)`. It contains two LSTM layers with 512 units each. The first returns a sequence to the second, and each LSTM is followed by dropout with a rate of 0.3. A 256-unit dense layer with ReLU activation precedes the final `|V|`-unit softmax layer. The softmax output is a probability distribution over the vocabulary.

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as a training metric. This configuration matches the sparse integer targets described in Section 4.8.

## 6. Training

The training pipeline splits complete pieces into training and validation sets before constructing windows. It saves the vocabulary and training configuration alongside the model artifacts, allowing generation to reproduce the same token mapping and sequence length. Checkpointing selects `best_model.keras` according to validation loss. Early stopping, learning-rate reduction, CSV logging, and training-state backup are also implemented.

### 6.1 Completed 100-File Experiment

The main training experiment used 100 MIDI pieces containing 328,489 extracted musical events. The piece-level split assigned 80 pieces to training and 20 to validation. The resulting vocabulary contained 1,962 note and chord classes.

| Quantity | Training | Validation |
| --- | ---: | ---: |
| Pieces | 80 | 20 |
| Sequences | 249,276 | 69,213 |
| Input shape | `(249276, 100, 1)` | `(69213, 100, 1)` |
| Target shape | `(249276,)` | `(69213,)` |
| Input dtype | `float32` | `float32` |
| Target dtype | `int32` | `int32` |

The model contained 3,787,434 parameters. Training requested a maximum of 50 epochs and stopped at epoch 15 through early stopping. The best checkpoint occurred at epoch 8 with a validation loss of 5.3470. At that epoch, training accuracy was approximately 2.79% and validation accuracy was approximately 2.27%. `ReduceLROnPlateau` reduced the learning rate at epochs 11 and 14.

Training loss continued to decrease after epoch 8 while validation loss stopped improving. Under this experiment, that divergence is evidence of overfitting. The low exact next-token classification accuracy does not, by itself, establish that generation failed: each prediction selects among 1,962 classes, and generated musical behavior is evaluated separately from exact next-token accuracy.

The run saved `checkpoints/best_model.keras`, `final_model.keras`, `vocabulary.json`, `training_config.json`, and `training_log.csv`. In the completed Colab experiment these artifacts were persisted under `/content/drive/MyDrive/lstm-music-generator/training-100/`. That Google Drive location records the experiment environment; it is not a required project path.

The generation experiment uses the saved best model as a fixed component so that decoding strategies can be compared without retraining. Training performance, decoding behavior, output diversity, and musical quality remain distinct evaluation questions.

## 7. Music Generation

### 7.1 Generation Pipeline

The generation pipeline in `src_colab/generate.py` loads `best_model.keras`, `vocabulary.json`, and `training_config.json`. The configuration supplies the training sequence length, while the vocabulary reconstructs both token-to-integer and integer-to-token mappings. Before generation, the script verifies that the model input length matches the configuration, the model output width matches the vocabulary size, and the vocabulary is non-empty.

A seed MIDI file is parsed through the same `extract_notes_from_midi()` function used by training preprocessing. Training and generation therefore share instrument partitioning, first-part selection, flattened-stream fallback, note pitch strings, and normal-order chord tokens rather than maintaining separate extraction logic. The generator searches the resulting events for contiguous 100-event windows whose tokens all occur in the training vocabulary. A valid window is selected with random seed 42 and converted to integer IDs. Before prediction, the current context is normalized by the vocabulary size and reshaped to `(1, 100, 1)`.

At each generation step, the LSTM returns a probability distribution over the vocabulary. The decoding strategy selects one token from this distribution. Its integer ID is appended to the context, the oldest ID is removed, and the resulting 100-event window becomes the input to the next prediction. This cycle repeats for the requested number of events.

This process is autoregressive: every prediction becomes part of the context used to make later predictions. Consequently, an error or repetitive pattern can influence subsequent output and compound over time because the model conditions on its own generated history rather than on ground-truth events.

### 7.2 Greedy Decoding

Greedy decoding selects the highest-probability class at every step:

```text
x_(t+1) = argmax_k P(x_(t+1) = k | x_(t-99), ..., x_t)
```

In the initial experiment, greedy decoding generated 500 events but selected only one unique generated token. This is a decoding failure mode, not evidence by itself that the trained model learned nothing. Once greedy decoding enters a high-probability repetitive state, it always chooses the same most likely continuation, and that prediction is fed back into the next context window. The repeated output can therefore reinforce itself.

### 7.3 Temperature Sampling

Temperature sampling selects probabilistically after transforming the model distribution. For class probabilities `p_i` and temperature `T > 0`, the implementation computes:

```text
q_i = exp(log(p_i) / T) / sum_j exp(log(p_j) / T)
```

The next class is sampled from `q`. Temperatures below 1.0 sharpen the distribution and favor higher-probability tokens. A temperature of 1.0 samples directly from the model distribution. Temperatures above 1.0 flatten the distribution and increase the probability of lower-ranked tokens. Higher temperature is not inherently better; it changes the balance between concentration and variety and may also increase unlikely transitions.

### 7.4 MIDI Reconstruction

Generated tokens are converted back into music21 objects and written to a MIDI file. Pitch-name tokens such as `C4` and `F#5` become `Note` objects. Numeric pitch-class tokens such as `0.4.7` become `Chord` objects whose pitch classes are placed relative to MIDI pitch 60. Generated objects receive a fixed duration of 0.5 quarter notes.

An edge case discovered during generation was a token containing one numeric pitch class, such as `"4"`. Although it contains no dot, it belongs to the chord/pitch-class token family rather than the note-name family. The conversion logic now validates numeric components in the range 0 through 11 and reconstructs these tokens as pitch-class-based `Chord` objects instead of passing them to `music21.note.Note`.

### 7.5 Generation Artifacts

When no output filename is supplied, the generator derives one from the decoding condition and requested event count, such as `greedy_500.mid` or `temp_0.8_500.mid`. An optional output directory keeps storage location under caller control. Existing MIDI or metadata artifacts are not overwritten unless the caller explicitly enables overwriting.

Each MIDI output has a JSON sidecar with the same base filename. It records the generation strategy, applicable temperature, requested event count, random seed, sequence length, vocabulary size, seed MIDI path, selected seed start index, number of valid seed windows, unique generated source-token count, input artifact paths, and output MIDI path. These records capture generation provenance without introducing unobserved training metrics.

## 8. Modernization of the Original Approach

The reimplementation uses current Keras and TensorFlow APIs while preserving the original categorical next-event framing and trained-model architecture. Sparse integer targets and `sparse_categorical_crossentropy` replace one-hot target matrices. Piece-level train/validation splitting and within-piece sequence creation prevent windows from crossing composition boundaries, and random seed 42 makes data splitting and seed selection reproducible.

The main training run used a Google Colab GPU and persisted its artifacts to Google Drive. Persistent best-model checkpoints, `EarlyStopping`, `ReduceLROnPlateau`, `BackupAndRestore`, and `CSVLogger` improve recoverability and experiment traceability. The generation workflow adds explicit greedy and temperature-sampling strategies, shared training/generation token extraction, compatibility validation, descriptive collision-safe output handling, and generation metadata sidecars.

These changes make the original approach easier to reproduce and inspect, but they do not remove the underlying limits of scalar token encoding or the reduced musical representation. Learned embeddings, richer rhythmic representations, and different model families remain possible future work rather than implemented modernization.

## 9. Results

### 9.1 Controlled Decoding Experiment

All decoding runs used the same trained model, seed MIDI file, selected seed position, random seed 42, 500 generated events, and generation logic. Only the decoding strategy or sampling temperature changed.

| Decoding condition | Unique source tokens | Generated events |
| --- | ---: | ---: |
| Greedy | 1 | 500 |
| Temperature 0.5 | 126 | 500 |
| Temperature 0.8 | 182 | 500 |
| Temperature 1.0 | 233 | 500 |
| Temperature 1.2 | 269 | 500 |

The sampled runs were also inspected after MIDI reconstruction:

| Temperature | Unique MIDI onset patterns | Adjacent repeated events | Longest identical run | Mean pitch jump (semitones) |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 119 | 52 | 6 | approximately 3.67 |
| 0.8 | 172 | 16 | 2 | approximately 8.07 |
| 1.0 | 221 | 3 | 2 | approximately 8.65 |
| 1.2 | 258 | 1 | 2 | approximately 6.45 |

Increasing temperature in this experiment increased source-token diversity and generally reduced direct adjacent repetition. The measurements do not show that higher-temperature output has better musical quality. No formal listening study, expert assessment, structural music analysis, or other musical-quality evaluation has yet been performed.

Source-token diversity and MIDI-level onset-pattern diversity differ because reconstruction is lossy. Multiple source tokens or distinctions from the parsed data can map to MIDI events that are equivalent under the reconstruction and analysis procedure, especially when chord register, voicing, duration, and timing have been discarded.

## 10. Limitations

The event representation stores note pitches such as `C4` and `F#5` and chord normal-order pitch classes such as `0.4.7`. It does not preserve original note duration, timing between events, velocity, pedal information, full chord octave or register, chord voicing, or inversion. Fixed-duration reconstruction further prevents the generated MIDI from reproducing the timing and articulation of the source performances.

Chord tokens are reconstructed from pitch classes around a fixed octave, so their output register and voicing are synthetic rather than recovered from the original MAESTRO performance. The present system should therefore be described as a pitch/event sequence generator, not a full expressive piano-performance generator.

The experiment evaluates observable diversity and repetition, not musical quality. More unique events can indicate less direct collapse, but it does not by itself imply stronger harmony, melody, phrasing, long-range form, stylistic consistency, or listener preference. The experiment also uses one model and one controlled seed condition, limiting the generality of its conclusions.

## 11. Future Improvements

The next stage should evaluate where improvements have the greatest effect: generation strategy, musical representation, or model architecture and training. Generation work could compare additional sampling controls and repetition-aware methods. Representation work could preserve duration, inter-event timing, velocity, pedal state, register, voicing, and rests, or replace scalar IDs with learned embeddings. Modeling work could then test architectural and training changes against clearly defined validation and musical-evaluation criteria.

These are proposed directions rather than implemented improvements. Future experiments should separate objective distributional measurements from claims about musical coherence or quality and should include multiple models, seeds, pieces, and human or musically grounded evaluation where appropriate.

## 12. Conclusion

The completed generation experiment demonstrates that decoding strategy can substantially change output from the same trained LSTM. Greedy decoding collapsed to one repeatedly generated token, whereas temperature sampling increased token and MIDI-event diversity and reduced immediate repetition under the tested conditions. This supports the narrow conclusion that decoding is a major determinant of observable generation behavior.

It does not establish that the more diverse outputs are musically superior. The current representation discards much of the information that makes a piano performance expressive, and musical coherence has not yet been formally evaluated. The next investigation should determine whether the most useful gains come from decoding, richer event representation, or changes to model architecture and training.

## References
