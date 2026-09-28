# Generating Music with LSTM Neural Networks: A Modern Reimplementation

## Abstract

This project revisits a 2017 LSTM-based music generation approach and rebuilds it using a more modern approach through a newer Python machine learning stack and a more reproducible dataset. The system treats symbolic music generation as a next-event prediction problem. First, MIDI files are parsed into sequences of note and chord tokens. Then they are converted into numerical representations in matricies. After that, they are divided into fixed-length training windows and prepared as tensors for an LSTM network. The current implementation uses the MAESTRO dataset, music21 for MIDI parsing, NumPy for numerical preprocessing, and modern Keras utilities for categorical targets. The first phase of the project focuses on reproducing the original preprocessing pipeline before introducing updated modeling and generation techniques that build from the original approach.

## 1. Introduction

Recurrent neural networks, or RNN, were widely used for sequential generation tasks before Transformer architectures became dominant in the late 2010s and early 2020s. One common application was symbolic music generation, where a model learns patterns from sequences of notes and predicts what musical event should occur next.

This project reproduces and updates the approach described in the 2017 *Medium* article *How to Generate Music using a LSTM Neural Network in Keras*. The original implementation of this demonstrated how an LSTM, or a Long Short-Term Memory network, could learn from MIDI files and generate new musical sequences. The goal of this project is not only to reproduce that pipeline, but also to examine how the implementation can be updated using current datasets, software libraries, and modeling practices nearly 10 years later.

The first stage of this implementation focuses on converting MIDI performances into training data suitable for a recurrent neural network. This process includes parsing MIDI files, representing notes and chords as discrete tokens, constructing a vocabulary from them, encoding those tokens numerically, generating fixed-length sequences, and formatting the resulting data for neural network training.

## 2. Background
### 2.1 Recurrent Neural Networks

Recurrent neural networks are designed for sequential data. Unlike a standard feedforward neural network, an RNN maintains a hidden state that carries information from previous timesteps. At each timestep, the network processes both the current input and information from the previous hidden state.

For a sequence $(x_1, x_2, \dots, x_t)$, the hidden state can be expressed conceptually as:

$$
h_t = f(x_t, h_{t-1})
$$

where $h_t$ represents the network's internal state at timestep $t$.

This makes RNNs suitable for data where ordering matters, including text, time series, speech, and symbolic music.

### 2.2 LSTM Networks

Standard RNNs can struggle to learn long-range relationships because gradients may shrink or grow as they propagate through many timesteps. Long Short-Term Memory networks were developed to address this problem by introducing gated mechanisms that control how information is stored, updated, and discarded.

An LSTM maintains an internal cell state along with a hidden state. Input, forget, and output gates regulate the flow of information through the network. This allows the model to preserve useful information across longer sequences than a simple RNN typically can.

In this project, the LSTM receives a sequence of previous musical events and learns to predict the event that follows.

### 2.3 Sequence Modeling for Music

Symbolic music can be modeled as a sequence of discrete events. Instead of processing raw audio waveforms, the system operates on MIDI data, which explicitly represents musical information such as note pitch, timing, and instrument events.
The current project simplifies the problem by representing the music as a sequence of note and chord tokens. Given a fixed number of previous tokens, the model is trained to predict the next token in the sequence.

The learning task can therefore be written as:

$$
P(x_{t+1} \mid x_{t-L+1}, \dots, x_t)
$$

where $L$ is the sequence length used as input. In the current implementation, $L = 100$.

### 2.4 The Original 2017 Implementation

The original 2017 implementation demonstrated a straightforward pipeline for symbolic music generation using Keras and music21. MIDI files were parsed into notes and chords, each unique musical event was assigned a numerical representation, and fixed-length sequences were used to train a multi-layer LSTM network.

That implementation provided a useful demonstration of sequence generation, but several parts of the software stack and experimental setup are now outdated. This project initially preserves the core modeling idea so that a comparable baseline can be established. Later stages will update selected components such as data representation, model training practices, and generation strategy.

## 3. Dataset
### 3.1 MAESTRO

This project uses the MAESTRO dataset as the source of MIDI training data. MAESTRO contains professionally performed piano recordings paired with MIDI representations. Because the current project operates on symbolic musical information rather than raw audio, only the MIDI files are required.

The dataset is organized into multiple performance years and contains substantially more material than is necessary during early development. To reduce iteration time while validating the preprocessing pipeline, the current experiment uses a subset of ten MIDI files.

### 3.2 MIDI Representation

MIDI is a symbolic music format rather than an audio format. A MIDI file does not store a waveform in the same way as a WAV or MP3 file. Instead, it stores structured performance information including note events, pitches, timing, and instrument data.

This distinction is useful for the current task because the model does not need to infer musical notes from audio. The note and chord information can be extracted directly from the MIDI representation.

### 3.3 Scope of the Current Experiment

The current preprocessing experiment uses ten MIDI files from the MAESTRO dataset. These files are processed recursively from the dataset directory and combined into a single sequence of musical events.

This limited subset is used only during development. The purpose at this stage is to verify the complete preprocessing pipeline before scaling the experiment to a larger training dataset.

## 4. Data Preprocessing

The purpose of the preprocessing pipeline is to convert raw MIDI files into numerical training data that can be used by an LSTM network.

The current pipeline performs the following transformations:

MIDI files
    ↓
music21 parsing
    ↓
note and chord extraction
    ↓
categorical token sequence
    ↓
vocabulary construction
    ↓
integer encoding
    ↓
fixed-length sequence generation
    ↓
LSTM-compatible tensor reshaping
    ↓
input normalization
    ↓
one-hot encoded targets

### 4.1 MIDI Parsing

The first preprocessing step is converting the raw MIDI files into Python objects that can be inspected and processed.

This project uses the `music21` library to parse MIDI files:

```python
midi = converter.parse(file)
```

MIDI files contain structured musical information rather than raw audio waveforms. After parsing, music21 represents the contents of the file as Python objects such as notes, chords, instruments, measures, and other musical events.

The dataset is searched recursively so that MIDI files can remain inside the original directory structure:

```python
midi_files = glob.glob(
    "data/midi/**/*.midi",
    recursive=True
)
```

During early development, only a small subset of the available MIDI files is processed. This makes the preprocessing pipeline faster to test and easier to debug before scaling to the full dataset.

After loading each MIDI file, the program attempts to separate the performance by instrument:

```python
parts = instrument.partitionByInstrument(midi)
```

If instrument parts are available, the first part is traversed recursively. If instrument partitions are not available, the complete MIDI structure is flattened and the note events are extracted.

Conceptually, this stage performs the following transformation:

MIDI file
    ↓
music21 object representation
    ↓
iterable musical events

At this stage, the musical data has been loaded into Python, but it has not yet been converted into the numerical format required by the neural network.

### 4.2 Note and Chord Representation

After parsing the MIDI files, the next step is extracting the musical events that will be used as model inputs.

The current implementation keeps individual notes and chords.

Single notes are identified using:

```python
if isinstance(element, note.Note):
```

The note pitch is then converted into a string. Example note representations include:

C4
F#5
A3

The letter represents the pitch class, while the number represents the octave. Chord objects are handled separately. 

Pitch classes are represented numerically from 0 to 11:

C  = 0
C# = 1
D  = 2
D# = 3
E  = 4
F  = 5
F# = 6
G  = 7
G# = 8
A  = 9
A# = 10
B  = 11

After extraction, the music becomes a sequential list of categorical tokens:

```python
[
    "C4",
    "E4",
    "G4",
    "0.4.7",
    "F#5",
    ...
]
```

This step reduces the original MIDI representation into a simpler sequence of musical events that can later be encoded numerically.

The current representation does not explicitly preserve all available MIDI information. Features such as note velocity, note duration, rests, articulation, and detailed rhythmic structure are not yet modeled.

### 4.3 Vocabulary Construction
### 4.4 Integer Encoding
### 4.5 Sequence Generation
### 4.6 Input Reshaping
### 4.7 Normalization
### 4.8 Target Encoding

## 5. Model Architecture
### 5.1 LSTM Layers
### 5.2 Dropout
### 5.3 Dense Output Layer
### 5.4 Softmax
### 5.5 Loss Function

## 6. Training
### 6.1 Training Configuration
### 6.2 Checkpointing
### 6.3 Hardware
### 6.4 Training Behavior

## 7. Music Generation
### 7.1 Seed Sequence
### 7.2 Autoregressive Prediction
### 7.3 Decoding Predictions
### 7.4 MIDI Reconstruction
### 7.5 Temperature Sampling

## 8. Modernization of the Original Approach

## 9. Results

## 10. Limitations

## 11. Future Improvements

## 12. Conclusion

## References