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
    prepare_sequences,
    split_pieces
)

from model import create_model


DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_RANDOM_SEED = 42
DEFAULT_INPUT_REPRESENTATION = "scalar"
DEFAULT_EMBEDDING_DIM = 128


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
    total_pieces,
    training_pieces,
    validation_pieces,
    training_sequences,
    validation_sequences
):
    config = {
        "sequence_length":
            args.sequence_length,

        "vocabulary_size":
            n_vocab,

        "midi_limit":
            args.midi_limit,

        "total_pieces":
            total_pieces,

        "training_pieces":
            training_pieces,

        "validation_pieces":
            validation_pieces,

        "training_sequences":
            training_sequences,

        "validation_sequences":
            validation_sequences,

        "validation_fraction":
            args.validation_fraction,

        "random_seed":
            args.seed,

        "batch_size":
            args.batch_size,

        "epochs_requested":
            args.epochs,

        "piece_boundaries_preserved":
            True,

        "split_by_piece":
            True,

        "target_encoding":
            "sparse_integer",

        "input_representation":
            args.input_representation,

        "embedding_dim": (
            args.embedding_dim
            if args.input_representation == "embedding"
            else None
        ),

        "input_encoding": (
            "normalized_scalar_token_ids"
            if args.input_representation == "scalar"
            else "integer_token_ids_with_learned_embedding"
        ),

        "loss":
            "sparse_categorical_crossentropy"
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
        f"Saved training config to {path}"
    )


def train(args):
    if args.embedding_dim <= 0:
        raise ValueError(
            "Embedding dimension must be greater than 0."
        )

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
        "\nSplitting pieces..."
    )

    (
        train_pieces,
        validation_pieces
    ) = split_pieces(
        pieces,
        validation_fraction=
            args.validation_fraction,
        seed=args.seed
    )

    print(
        f"Training pieces: "
        f"{len(train_pieces)}"
    )

    print(
        f"Validation pieces: "
        f"{len(validation_pieces)}"
    )

    print(
        "\nBuilding vocabulary..."
    )

    # Baseline design:
    # vocabulary is built from all selected pieces.
    #
    # This avoids unknown validation tokens while the
    # project reproduces the original categorical setup.
    pitchnames, note_to_int = (
        build_vocabulary(
            pieces
        )
    )

    n_vocab = len(
        pitchnames
    )

    print(
        f"Vocabulary size: {n_vocab}"
    )

    save_vocabulary(
        pitchnames,
        args.output_dir
    )

    print(
        "\nCreating training sequences..."
    )

    (
        train_input,
        train_output
    ) = create_sequences(
        train_pieces,
        note_to_int,
        sequence_length=
            args.sequence_length
    )

    print(
        f"Training sequences: "
        f"{len(train_input)}"
    )

    print(
        "\nCreating validation sequences..."
    )

    (
        validation_input,
        validation_output
    ) = create_sequences(
        validation_pieces,
        note_to_int,
        sequence_length=
            args.sequence_length
    )

    print(
        f"Validation sequences: "
        f"{len(validation_input)}"
    )

    if not train_input:
        raise ValueError(
            "No training sequences generated"
        )

    if not validation_input:
        raise ValueError(
            "No validation sequences generated"
        )

    save_training_config(
        args=args,
        n_vocab=n_vocab,
        total_pieces=len(pieces),
        training_pieces=len(
            train_pieces
        ),
        validation_pieces=len(
            validation_pieces
        ),
        training_sequences=len(
            train_input
        ),
        validation_sequences=len(
            validation_input
        )
    )

    print(
        "\nPreparing training tensors..."
    )

    (
        train_input,
        train_output
    ) = prepare_sequences(
        train_input,
        train_output,
        n_vocab,
        input_representation=
            args.input_representation,
    )

    print(
        "\nPreparing validation tensors..."
    )

    (
        validation_input,
        validation_output
    ) = prepare_sequences(
        validation_input,
        validation_output,
        n_vocab,
        input_representation=
            args.input_representation,
    )

    print(
        "\nTraining input shape:",
        train_input.shape
    )

    print(
        "Training output shape:",
        train_output.shape
    )

    print(
        "Validation input shape:",
        validation_input.shape
    )

    print(
        "Validation output shape:",
        validation_output.shape
    )

    print(
        "\nTraining input dtype:",
        train_input.dtype
    )

    print(
        "Training output dtype:",
        train_output.dtype
    )

    print(
        "Validation input dtype:",
        validation_input.dtype
    )

    print(
        "Validation output dtype:",
        validation_output.dtype
    )

    print(
        "\nBuilding model..."
    )

    model = create_model(
        sequence_length=
            args.sequence_length,
        n_vocab=n_vocab,
        input_representation=
            args.input_representation,
        embedding_dim=args.embedding_dim,
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
        train_input,
        train_output,

        validation_data=(
            validation_input,
            validation_output
        ),

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
    parser = argparse.ArgumentParser(
        description=(
            "Train the Colab LSTM "
            "music-generation model."
        )
    )

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

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=
            DEFAULT_VALIDATION_FRACTION
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED
    )

    parser.add_argument(
        "--input-representation",
        choices=[
            "scalar",
            "embedding",
        ],
        default=DEFAULT_INPUT_REPRESENTATION
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(args)
