import argparse
import json
import os

import tensorflow as tf

from keras.callbacks import (
    BackupAndRestore,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from preprocess import get_pieces

from dataset import (
    build_vocabulary,
    create_sequences,
    prepare_sequences
)

from model import create_model


DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64


def configure_gpu():
    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    if not gpus:
        print(
            "WARNING: No GPU detected"
        )
        return

    print(
        f"GPU detected: {gpus[0]}"
    )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )
        except RuntimeError:
            pass


def save_vocabulary(
    pitchnames,
    output_dir
):
    path = os.path.join(
        output_dir,
        "vocabulary.json"
    )

    with open(
        path,
        "w"
    ) as file:
        json.dump(
            pitchnames,
            file,
            indent=2
        )

    print(
        f"Saved vocabulary to {path}"
    )


def save_training_config(
    args,
    n_vocab,
    number_of_pieces,
    number_of_sequences
):
    config = {
        "sequence_length":
            args.sequence_length,

        "vocabulary_size":
            n_vocab,

        "midi_limit":
            args.midi_limit,

        "number_of_pieces":
            number_of_pieces,

        "number_of_sequences":
            number_of_sequences,

        "batch_size":
            args.batch_size,

        "epochs_requested":
            args.epochs,

        "piece_boundaries_preserved":
            True
    }

    path = os.path.join(
        args.output_dir,
        "training_config.json"
    )

    with open(
        path,
        "w"
    ) as file:
        json.dump(
            config,
            file,
            indent=2
        )

    print(
        f"Saved training config "
        f"to {path}"
    )


def train(args):
    configure_gpu()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    checkpoint_dir = os.path.join(
        args.output_dir,
        "checkpoints"
    )

    backup_dir = os.path.join(
        args.output_dir,
        "training_backup"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    print(
        "\nLoading MIDI data..."
    )

    pieces = get_pieces(
        data_dir=args.data_dir,
        limit=args.midi_limit
    )

    print(
        "\nBuilding vocabulary..."
    )

    pitchnames, note_to_int = (
        build_vocabulary(
            pieces
        )
    )

    n_vocab = len(
        pitchnames
    )

    print(
        f"Vocabulary size: "
        f"{n_vocab}"
    )

    save_vocabulary(
        pitchnames,
        args.output_dir
    )

    print(
        "\nCreating sequences..."
    )

    network_input, network_output = (
        create_sequences(
            pieces,
            note_to_int,
            sequence_length=
                args.sequence_length
        )
    )

    n_sequences = len(
        network_input
    )

    print(
        f"Sequences: {n_sequences}"
    )

    if n_sequences == 0:
        raise ValueError(
            "No training sequences "
            "were generated."
        )

    save_training_config(
        args,
        n_vocab,
        len(pieces),
        n_sequences
    )

    print(
        "\nPreparing tensors..."
    )

    network_input, network_output = (
        prepare_sequences(
            network_input,
            network_output,
            n_vocab
        )
    )

    print(
        "Input shape:",
        network_input.shape
    )

    print(
        "Output shape:",
        network_output.shape
    )

    print(
        "Input dtype:",
        network_input.dtype
    )

    print(
        "Output dtype:",
        network_output.dtype
    )

    print(
        "\nBuilding model..."
    )

    model = create_model(
        sequence_length=
            args.sequence_length,
        n_vocab=n_vocab
    )

    model.summary()

    best_model_path = os.path.join(
        checkpoint_dir,
        "best_model.keras"
    )

    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=7,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    csv_logger = CSVLogger(
        os.path.join(
            args.output_dir,
            "training_log.csv"
        ),
        append=True
    )

    backup = BackupAndRestore(
        backup_dir=backup_dir
    )

    print(
        "\nStarting training..."
    )

    history = model.fit(
        network_input,
        network_output,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr,
            csv_logger,
            backup
        ],
        verbose=1
    )

    final_model_path = os.path.join(
        args.output_dir,
        "final_model.keras"
    )

    model.save(
        final_model_path
    )

    print(
        "\nTraining complete"
    )

    print(
        f"Best model: "
        f"{best_model_path}"
    )

    print(
        f"Final model: "
        f"{final_model_path}"
    )

    return history


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/midi"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_output"
    )

    parser.add_argument(
        "--midi-limit",
        type=int,
        default=10
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(args)