import argparse
import json
from pathlib import Path

from keras.callbacks import ModelCheckpoint

from preprocess import get_notes
from dataset import (
    build_vocabulary,
    create_sequences,
    prepare_sequences,
)
from model import create_model


DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_MIDI_LIMIT = 10
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_OUTPUT_DIR = "checkpoints/local"


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def train(args):
    if args.sequence_length <= 0:
        raise ValueError("Sequence length must be greater than 0.")

    if args.epochs <= 0:
        raise ValueError("Epoch count must be greater than 0.")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    notes = get_notes(
        data_dir=args.data_dir,
        limit=args.midi_limit
    )

    if len(notes) <= args.sequence_length:
        raise ValueError(
            "Not enough MIDI events to create baseline sequences. "
            f"Found {len(notes)} events for a sequence length of "
            f"{args.sequence_length}."
        )

    pitchnames, note_to_int = build_vocabulary(notes)
    n_vocab = len(pitchnames)

    print(f"\nVocabulary size: {n_vocab}")

    save_json(
        pitchnames,
        output_dir / "vocabulary.json"
    )

    training_config = {
        "sequence_length": args.sequence_length,
        "vocabulary_size": n_vocab,
        "midi_limit": args.midi_limit,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "target_encoding": "one-hot",
        "loss": "categorical_crossentropy",
        "validation_method": (
            "sequence-level Keras validation_split=0.2"
        ),
    }

    save_json(
        training_config,
        output_dir / "training_config.json"
    )

    network_input, network_output = create_sequences(
        notes,
        note_to_int,
        sequence_length=args.sequence_length
    )

    network_input, network_output = prepare_sequences(
        network_input,
        network_output,
        n_vocab
    )

    print("Input shape:", network_input.shape)
    print("Output shape:", network_output.shape)

    model = create_model(
        sequence_length=args.sequence_length,
        n_vocab=n_vocab
    )

    model.summary()

    checkpoint = ModelCheckpoint(
        filepath=str(output_dir / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    history = model.fit(
        network_input,
        network_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[checkpoint]
    )

    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the local baseline LSTM model."
    )
    parser.add_argument(
        "--data-dir",
        default="data/midi"
    )
    parser.add_argument(
        "--midi-limit",
        type=int,
        default=DEFAULT_MIDI_LIMIT
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH
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
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
