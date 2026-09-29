import random

import numpy as np


def split_pieces(
    pieces,
    validation_fraction=0.2,
    seed=42
):
    """
    Split entire musical pieces into training and validation sets.

    A fixed seed makes the experiment reproducible.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    if len(pieces) < 2:
        raise ValueError(
            "At least two pieces are required "
            "for a train/validation split"
        )

    shuffled_pieces = list(pieces)

    rng = random.Random(seed)
    rng.shuffle(shuffled_pieces)

    split_index = int(
        len(shuffled_pieces)
        * (1 - validation_fraction)
    )

    train_pieces = shuffled_pieces[
        :split_index
    ]

    validation_pieces = shuffled_pieces[
        split_index:
    ]

    return (
        train_pieces,
        validation_pieces
    )


def build_vocabulary(pieces):
    """
    Build a vocabulary from every unique note/chord token
    in the selected dataset.
    """
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
    """
    Create next-event prediction samples without crossing
    musical piece boundaries.
    """
    network_input = []
    network_output = []

    skipped_pieces = 0

    for piece in pieces:
        if len(piece) <= sequence_length:
            skipped_pieces += 1
            continue

        encoded_piece = [
            note_to_int[note_name]
            for note_name in piece
        ]

        for i in range(
            len(encoded_piece)
            - sequence_length
        ):
            sequence_in = encoded_piece[
                i:i + sequence_length
            ]

            sequence_out = encoded_piece[
                i + sequence_length
            ]

            network_input.append(
                sequence_in
            )

            network_output.append(
                sequence_out
            )

    if skipped_pieces:
        print(
            f"Skipped {skipped_pieces} pieces "
            f"with <= {sequence_length} events"
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
    """
    Convert input sequences to float32 LSTM tensors.

    Targets remain integer class IDs so Keras can use
    sparse categorical cross-entropy instead of allocating
    a large one-hot target matrix.
    """
    network_input = np.asarray(
        network_input,
        dtype=np.float32
    )

    network_output = np.asarray(
        network_output,
        dtype=np.int32
    )

    if len(network_input) == 0:
        raise ValueError(
            "No sequences were generated"
        )

    network_input = np.expand_dims(
        network_input,
        axis=-1
    )

    network_input /= float(
        n_vocab
    )

    return (
        network_input,
        network_output
    )