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
### 4.1 MIDI Parsing
### 4.2 Note and Chord Representation
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