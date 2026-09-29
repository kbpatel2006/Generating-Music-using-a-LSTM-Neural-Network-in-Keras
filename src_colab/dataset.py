import numpy as np

from keras.utils import to_categorical


def build_vocabulary(pieces):
    all_notes = [
        note_name
        for piece in pieces
        for note_name in piece
    ]

    pitchnames = sorted(
        set(all_notes)
    )

    note_to_int = {
        note_name: number
        for number, note_name
        in enumerate(pitchnames)
    }

    return (
        pitchnames,
        note_to_int
    )


def create_sequences(
    pieces,
    note_to_int,
    sequence_length=100
):
    network_input = []
    network_output = []

    skipped_pieces = 0

    for piece in pieces:

        if len(piece) <= sequence_length:
            skipped_pieces += 1
            continue

        for i in range(
            len(piece) - sequence_length
        ):
            sequence_in = piece[
                i:i + sequence_length
            ]

            sequence_out = piece[
                i + sequence_length
            ]

            encoded_sequence = [
                note_to_int[note_name]
                for note_name
                in sequence_in
            ]

            network_input.append(
                encoded_sequence
            )

            network_output.append(
                note_to_int[
                    sequence_out
                ]
            )

    if skipped_pieces:
        print(
            f"Skipped {skipped_pieces} "
            f"pieces shorter than "
            f"{sequence_length + 1} events"
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

    n_patterns = len(
        network_input
    )

    network_input = np.reshape(
        network_input,
        (
            n_patterns,
            network_input.shape[1],
            1
        )
    )

    network_input /= float(
        n_vocab
    )

    network_output = to_categorical(
        network_output,
        num_classes=n_vocab
    ).astype(
        np.float32
    )

    return (
        network_input,
        network_output
    )