import numpy as np
from keras.utils import to_categorical


def build_vocabulary(notes):
    pitchnames = sorted(set(notes))

    note_to_int = {
        note_name: number
        for number, note_name in enumerate(pitchnames)
    }

    return pitchnames, note_to_int


def create_sequences(
    notes,
    note_to_int,
    sequence_length=100
):
    network_input = []
    network_output = []

    for i in range(
        len(notes) - sequence_length
    ):
        sequence_in = notes[
            i:i + sequence_length
        ]

        sequence_out = notes[
            i + sequence_length
        ]

        encoded_sequence = [
            note_to_int[note_name]
            for note_name in sequence_in
        ]

        network_input.append(
            encoded_sequence
        )

        network_output.append(
            note_to_int[sequence_out]
        )

    return (
        network_input,
        network_output
    )


def prepare_sequences(
    network_input,
    network_output,
    n_vocab
):
    network_input = np.asarray(
        network_input,
        dtype=np.float32
    )

    network_output = np.asarray(
        network_output,
        dtype=np.int32
    )

    n_patterns = len(network_input)

    network_input = np.reshape(
        network_input,
        (
            n_patterns,
            network_input.shape[1],
            1
        )
    )

    network_input /= float(n_vocab)

    network_output = to_categorical(
        network_output,
        num_classes=n_vocab
    )

    return (
        network_input,
        network_output
    )