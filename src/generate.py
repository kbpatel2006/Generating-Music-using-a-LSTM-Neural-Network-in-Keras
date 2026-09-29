import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from music21 import chord, note, stream

from preprocess import extract_notes_from_midi


DEFAULT_SEQUENCE_LENGTH = 100


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def load_vocabulary(path):
    vocabulary = load_json(path)

    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError(
            "Baseline vocabulary.json must be a non-empty list."
        )

    token_to_int = {
        token: index
        for index, token in enumerate(vocabulary)
    }

    return vocabulary, token_to_int


def get_sequence_length(config_path):
    if config_path is None:
        return DEFAULT_SEQUENCE_LENGTH

    config_path = Path(config_path)

    if not config_path.exists():
        print(
            f"Training config not found at {config_path}; "
            f"using sequence length {DEFAULT_SEQUENCE_LENGTH}."
        )
        return DEFAULT_SEQUENCE_LENGTH

    config = load_json(config_path)

    return int(
        config.get(
            "sequence_length",
            DEFAULT_SEQUENCE_LENGTH
        )
    )


def find_seed(events, token_to_int, sequence_length):
    for start in range(
        len(events) - sequence_length + 1
    ):
        seed = events[start:start + sequence_length]

        if all(token in token_to_int for token in seed):
            return seed

    raise ValueError(
        "Seed MIDI does not contain a valid contiguous "
        f"sequence of {sequence_length} vocabulary events."
    )


def prepare_model_input(pattern, n_vocab):
    model_input = np.asarray(
        pattern,
        dtype=np.float32
    )
    model_input = model_input.reshape(
        1,
        len(pattern),
        1
    )

    return model_input / float(n_vocab)


def generate_tokens(
    model,
    seed_pattern,
    vocabulary,
    num_events,
):
    pattern = list(seed_pattern)
    n_vocab = len(vocabulary)
    generated_tokens = []

    for _ in range(num_events):
        probabilities = model.predict(
            prepare_model_input(pattern, n_vocab),
            verbose=0
        )[0]

        prediction_index = int(
            np.argmax(probabilities)
        )
        generated_tokens.append(
            vocabulary[prediction_index]
        )

        pattern.append(prediction_index)
        pattern = pattern[1:]

    return generated_tokens


def token_to_music21(token, duration=0.5):
    parts = token.split(".")
    pitch_classes = []

    for part in parts:
        try:
            value = int(part)
        except ValueError:
            pitch_classes = []
            break

        if value < 0 or value > 11:
            pitch_classes = []
            break

        pitch_classes.append(value)

    if pitch_classes:
        musical_object = chord.Chord([
            60 + pitch_class
            for pitch_class in pitch_classes
        ])
    else:
        musical_object = note.Note(token)

    musical_object.quarterLength = duration

    return musical_object


def write_midi(generated_tokens, output_path):
    output_stream = stream.Stream()

    for token in generated_tokens:
        output_stream.append(
            token_to_music21(token)
        )

    output_stream.write(
        "midi",
        fp=str(output_path)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MIDI with the local baseline model."
    )
    parser.add_argument(
        "--model",
        default="checkpoints/local/best_model.keras"
    )
    parser.add_argument(
        "--vocabulary",
        default="checkpoints/local/vocabulary.json"
    )
    parser.add_argument(
        "--config",
        default="checkpoints/local/training_config.json"
    )
    parser.add_argument(
        "--seed-midi",
        required=True
    )
    parser.add_argument(
        "--output",
        default="output/local/generated_music.mid"
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=500
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_events <= 0:
        raise ValueError(
            "Number of generated events must be greater than 0."
        )

    vocabulary, token_to_int = load_vocabulary(
        args.vocabulary
    )
    sequence_length = get_sequence_length(
        args.config
    )

    seed_events = extract_notes_from_midi(
        args.seed_midi
    )
    seed = find_seed(
        seed_events,
        token_to_int,
        sequence_length
    )
    seed_pattern = [
        token_to_int[token]
        for token in seed
    ]

    model = tf.keras.models.load_model(
        args.model
    )
    generated_tokens = generate_tokens(
        model=model,
        seed_pattern=seed_pattern,
        vocabulary=vocabulary,
        num_events=args.num_events
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    write_midi(
        generated_tokens,
        output_path
    )

    print(f"Generated MIDI saved to {output_path}")


if __name__ == "__main__":
    main()
