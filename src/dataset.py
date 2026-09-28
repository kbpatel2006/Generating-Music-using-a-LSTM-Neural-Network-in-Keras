import numpy as np
from keras.utils import to_categorical

def build_vocabulary(notes):
    pitchnames = sorted(set(notes))

    note_to_int = {
        note_name: number
        for number, note_name in enumerate(pitchnames)
    }

    return pitchnames, note_to_int

def create_sequences(notes, note_to_int, sequence_length=100):
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])

        encoded_sequence = [note_to_int[note] for note in sequence_in]

        network_input.append(encoded_sequence)
        network_output.append(note_to_int[sequence_out])

    return network_input, network_output

def prepare_sequences(network_input, network_output, n_vocab):
    n_patterns = len(network_input)
    network_input = np.reshape(
        network_input,
        (n_patterns, len(network_input[0]), 1)
    )

    network_input = network_input / float(n_vocab)
    
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return network_input, network_output
