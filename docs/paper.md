# Generating Music with LSTM Neural Networks: A Modern Reimplementation

## Abstract

This project is a modern reimplementation of a 2017 Keras-based approach to symbolic music generation with Long Short-Term Memory networks. The current stage focuses on reproducing and documenting the preprocessing pipeline rather than training a model or evaluating generated music. MIDI files from the MAESTRO dataset are parsed with music21, reduced to note and chord tokens, converted into integer identifiers, and arranged into fixed-length next-event prediction examples. The resulting arrays are reshaped into the three-dimensional format expected by Keras recurrent layers, normalized as scalar inputs, and paired with one-hot encoded target classes. This document describes the completed preprocessing stage, the assumptions inherited from the original implementation, and the representational limitations that motivate later modernization work.

## 1. Introduction

Recurrent neural networks were a common choice for sequence generation tasks before Transformer architectures became dominant. One application was symbolic music generation, where a model learns patterns from a sequence of musical events and predicts the event that should follow.

This project revisits the 2017 *Medium* article *How to Generate Music using a LSTM Neural Network in Keras*. The purpose is not to claim a new state-of-the-art result, but to reproduce the original pipeline clearly enough to understand its behavior, then use that understanding as a foundation for future improvements.

The project is currently at the preprocessing stage. The code parses MIDI files, extracts note and chord events, builds a vocabulary, creates 100-event sliding windows, reshapes the data for an LSTM input interface, and encodes the target event as a categorical label. No LSTM model has been trained yet, and no generated music or evaluation results are reported here.

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

MAESTRO is larger than needed for early pipeline testing. During development, the preprocessing script limits execution to a small subset of files so that parsing and sequence generation remain fast.

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

For development, the script processes only the first ten discovered files:

```python
midi_files = midi_files[:10]
```

This subset is a development constraint, not a property of the full method. It allows the preprocessing pipeline to be tested before running over a larger portion of MAESTRO.

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
→ one-hot targets
```

The implemented pipeline is split across `src/preprocess.py` and `src/dataset.py`. The first file handles MIDI discovery, parsing, and token extraction. The second file builds the vocabulary, creates training windows, reshapes model inputs, normalizes input values, and one-hot encodes targets.

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

During one development run, the observed vocabulary size was approximately 741 unique tokens. That value depends on which MIDI files are included and should be treated as a development-run observation, not a universal constant.

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

The current implementation also contains a duplicated append inside the same loop:

```python
encoded_sequence = [note_to_int[note] for note in sequence_in]

network_input.append(encoded_sequence)
network_output.append(note_to_int[sequence_out])
```

As written, each window is added twice. This doubles the number of generated samples without adding new musical contexts. It does not change the representation itself, but it affects sample counts and should be corrected or intentionally justified before training.

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

In one development run, the prepared input shape was approximately:

```text
(42446, 100, 1)
```

That value reflects the subset, vocabulary, and current sequence-generation behavior in that run. It should not be interpreted as a fixed property of the full dataset.

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

The current implementation converts integer target IDs into one-hot vectors:

```python
network_output = to_categorical(network_output, num_classes=n_vocab)
```

If the vocabulary has `|V| = 741` tokens, each target vector has 741 positions. A single position is `1`, corresponding to the correct next token, and all other positions are `0`.

In one development run, the prepared output shape was approximately:

```text
(42446, 741)
```

As with the input shape, this is a development-run value rather than a guaranteed size.

### 4.9 Current Preprocessing Pipeline

The completed preprocessing stage performs the following operations:

1. Recursively search `data/midi/` for `.midi` files.
2. Limit the development run to the first ten files.
3. Parse each MIDI file with music21.
4. Attempt to partition the parsed file by instrument.
5. If instrument parts exist, traverse the first part recursively.
6. Otherwise, flatten the MIDI stream and extract notes.
7. Store individual notes as pitch strings such as `C4`, `F#5`, and `A3`.
8. Store chords as dot-separated normal-order pitch classes such as `0.4.7`.
9. Build a deterministic vocabulary with `sorted(set(notes))`.
10. Map each unique token to an integer ID.
11. Generate 100-event sliding windows for next-event prediction.
12. Reshape inputs into `(samples, 100, 1)`.
13. Normalize input IDs by dividing by the vocabulary size.
14. One-hot encode targets with Keras `to_categorical`.

At the end of this stage, the project has numerical inputs and categorical targets prepared for an LSTM-style model. Training has not yet been performed.

### 4.10 Current Limitations and Modernization Opportunities

The current preprocessing pipeline is intentionally close to the original 2017 approach. That makes it useful as a reproduction baseline, but several limitations should be addressed before treating it as a modern representation of symbolic music.

First, the integer token IDs are arbitrary categorical labels. The sorted vocabulary provides deterministic IDs, but it does not provide musical distances. Normalizing those IDs with `x' = x / |V|` creates scalar values that imply an artificial numeric relationship between categories. A future implementation could represent events with learned embeddings, allowing the model to learn a dense representation instead of receiving normalized category IDs as if they were measurements.

Second, the event representation omits several musically important features. Durations, note offsets, velocity, rests, rhythm, tempo context, articulation, and expressive timing are not currently modeled. This reduces the task to predicting pitch and chord-event tokens, which is simpler than generating complete musical performances.

Third, the chord representation is simplified. Normal-order pitch classes such as `0.4.7` capture the pitch classes in a chord, but not voicing, inversion, octave placement, register, duration, or onset context. Two musically different chord events can therefore collapse into the same token.

Fourth, the preprocessing script appends events from all processed MIDI files into one long `notes` list before sequence generation. This can create training windows that cross from the end of one piece into the beginning of another. That behavior is acceptable for a preliminary baseline, but a cleaner preprocessing design would create windows within each piece independently, then combine the resulting samples.

Fifth, the current `create_sequences` implementation appends each generated window twice. This duplicates training examples and affects reported sample counts. Before model training, this should be corrected unless duplicate weighting is intended.

Finally, the current instrument handling uses only the first partitioned part when `instrument.partitionByInstrument` succeeds. That reproduces a simple extraction strategy, but future work should decide explicitly how to handle multiple parts, piano hands, tracks, and any non-piano material.

## 5. Model Architecture

## 6. Training

## 7. Music Generation

## 8. Modernization of the Original Approach

## 9. Results

## 10. Limitations

## 11. Future Improvements

## 12. Conclusion

## References
