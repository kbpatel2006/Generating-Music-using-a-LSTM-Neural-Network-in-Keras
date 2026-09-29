import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from music21 import chord, converter, note, stream


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def load_vocabulary(vocabulary_path):
    vocab_data = load_json(vocabulary_path)

    if isinstance(vocab_data, list):
        token_to_int = {
            token: index
            for index, token in enumerate(vocab_data)
        }

    elif isinstance(vocab_data, dict):
        if "token_to_int" in vocab_data:
            token_to_int = {
                token: int(index)
                for token, index in vocab_data["token_to_int"].items()
            }

        elif "note_to_int" in vocab_data:
            token_to_int = {
                token: int(index)
                for token, index in vocab_data["note_to_int"].items()
            }

        else:
            token_to_int = {
                token: int(index)
                for token, index in vocab_data.items()
            }

    else:
        raise ValueError("Unsupported vocabulary.json format.")

    int_to_token = {
        index: token
        for token, index in token_to_int.items()
    }

    return token_to_int, int_to_token


def extract_events_from_midi(midi_path):
    midi = converter.parse(midi_path)

    events = []

    for element in midi.flatten().notes:
        if isinstance(element, note.Note):
            events.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            chord_token = ".".join(
                str(pitch_class)
                for pitch_class in element.normalOrder
            )
            events.append(chord_token)

    return events


def find_valid_seed(
    events,
    token_to_int,
    sequence_length,
    random_seed=42,
):
    valid_start_positions = []

    max_start = len(events) - sequence_length

    for start in range(max_start + 1):
        window = events[start:start + sequence_length]

        if all(token in token_to_int for token in window):
            valid_start_positions.append(start)

    if not valid_start_positions:
        raise ValueError(
            "Could not find a valid seed sequence."
        )

    rng = random.Random(random_seed)

    start = rng.choice(valid_start_positions)

    seed_events = events[
        start:start + sequence_length
    ]

    print(
        f"Found {len(valid_start_positions)} valid seed windows."
    )

    print(
        f"Selected seed starting at event {start}."
    )

    return seed_events


def events_to_pattern(seed_events, token_to_int):
    return [
        token_to_int[event]
        for event in seed_events
    ]


def prepare_model_input(pattern, n_vocab):
    model_input = np.array(
        pattern,
        dtype=np.float32,
    )

    model_input = model_input.reshape(
        1,
        len(pattern),
        1,
    )

    model_input = model_input / float(n_vocab)

    return model_input


def sample_with_temperature(
    probabilities,
    temperature,
    rng,
):
    """
    Sample from the model's probability distribution.

    Lower temperature:
        more conservative / repetitive

    Higher temperature:
        more random / diverse

    temperature = 1.0:
        original model distribution
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    # Avoid log(0)
    probabilities = np.clip(
        probabilities,
        1e-10,
        1.0,
    )

    logits = np.log(probabilities)

    logits = logits / temperature

    logits = logits - np.max(logits)

    adjusted_probabilities = np.exp(logits)

    adjusted_probabilities /= np.sum(
        adjusted_probabilities
    )

    return rng.choice(
        len(adjusted_probabilities),
        p=adjusted_probabilities,
    )


def choose_next_token(
    probabilities,
    strategy,
    temperature,
    rng,
):
    if strategy == "greedy":
        return int(
            np.argmax(probabilities)
        )

    if strategy == "temperature":
        return int(
            sample_with_temperature(
                probabilities,
                temperature,
                rng,
            )
        )

    raise ValueError(
        f"Unknown generation strategy: {strategy}"
    )


def generate_tokens(
    model,
    seed_pattern,
    int_to_token,
    n_vocab,
    num_generate=500,
    strategy="greedy",
    temperature=1.0,
    random_seed=42,
):
    pattern = list(seed_pattern)

    generated_tokens = []

    rng = np.random.default_rng(
        random_seed
    )

    for step in range(num_generate):
        model_input = prepare_model_input(
            pattern,
            n_vocab,
        )

        probabilities = model.predict(
            model_input,
            verbose=0,
        )[0]

        prediction_index = choose_next_token(
            probabilities=probabilities,
            strategy=strategy,
            temperature=temperature,
            rng=rng,
        )

        predicted_token = int_to_token[
            prediction_index
        ]

        generated_tokens.append(
            predicted_token
        )

        pattern.append(
            prediction_index
        )

        pattern = pattern[1:]

        if (step + 1) % 50 == 0:
            unique_count = len(
                set(generated_tokens)
            )

            print(
                f"Generated {step + 1}/{num_generate} events "
                f"| unique tokens: {unique_count}"
            )

    return generated_tokens


def token_to_music21(token, duration=0.5):
    """
    Convert one generated token back into a music21 object.

    Note tokens look like:
        C4
        F#5
        B-3

    Chord/pitch-class tokens look like:
        0.4.7
        2.5.9
        4

    Numeric tokens are interpreted as pitch-class-based chord tokens.
    """

    parts = token.split(".")

    is_pitch_class_token = True

    pitch_classes = []

    for part in parts:
        try:
            value = int(part)

            if value < 0 or value > 11:
                is_pitch_class_token = False
                break

            pitch_classes.append(value)

        except ValueError:
            is_pitch_class_token = False
            break

    if is_pitch_class_token:
        midi_pitches = [
            60 + pitch_class
            for pitch_class in pitch_classes
        ]

        musical_object = chord.Chord(
            midi_pitches
        )

    else:
        musical_object = note.Note(
            token
        )

    musical_object.quarterLength = duration

    return musical_object


def write_midi(
    generated_tokens,
    output_path,
    duration=0.5,
):
    output_stream = stream.Stream()

    for index, token in enumerate(generated_tokens):
        try:
            musical_object = token_to_music21(
                token,
                duration=duration,
            )

            output_stream.append(
                musical_object
            )

        except Exception as error:
            print(
                f"Failed to convert token at position {index}: "
                f"{token!r}"
            )
            raise error

    output_stream.write(
        "midi",
        fp=str(output_path),
    )


def get_sequence_length(training_config):
    possible_keys = [
        "sequence_length",
        "sequence_len",
        "seq_length",
    ]

    for key in possible_keys:
        if key in training_config:
            return int(training_config[key])

    print(
        "Warning: sequence length not found in config. "
        "Using 100."
    )

    return 100


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIDI music using a trained LSTM."
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--vocabulary",
        required=True,
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--seed-midi",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="generated_music.mid",
    )

    parser.add_argument(
        "--num-events",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "greedy",
            "temperature",
        ],
        default="greedy",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if (
        args.strategy == "temperature"
        and args.temperature <= 0
    ):
        raise ValueError(
            "Temperature must be greater than 0."
        )

    model_path = Path(args.model)
    vocabulary_path = Path(args.vocabulary)
    config_path = Path(args.config)
    seed_midi_path = Path(args.seed_midi)
    output_path = Path(args.output)

    print("Loading training configuration...")

    training_config = load_json(
        config_path
    )

    sequence_length = get_sequence_length(
        training_config
    )

    print(
        f"Sequence length: {sequence_length}"
    )

    print("Loading vocabulary...")

    token_to_int, int_to_token = load_vocabulary(
        vocabulary_path
    )

    n_vocab = len(token_to_int)

    print(
        f"Vocabulary size: {n_vocab}"
    )

    print("Loading trained model...")

    model = tf.keras.models.load_model(
        model_path
    )

    print(
        f"Model input shape: {model.input_shape}"
    )

    print(
        f"Model output shape: {model.output_shape}"
    )

    print("Parsing seed MIDI...")

    seed_file_events = extract_events_from_midi(
        seed_midi_path
    )

    print(
        f"Seed MIDI contains "
        f"{len(seed_file_events)} musical events."
    )

    seed_events = find_valid_seed(
        events=seed_file_events,
        token_to_int=token_to_int,
        sequence_length=sequence_length,
        random_seed=args.random_seed,
    )

    seed_pattern = events_to_pattern(
        seed_events,
        token_to_int,
    )

    print(
        f"Generation strategy: {args.strategy}"
    )

    if args.strategy == "temperature":
        print(
            f"Temperature: {args.temperature}"
        )

    print("Generating music...")

    generated_tokens = generate_tokens(
        model=model,
        seed_pattern=seed_pattern,
        int_to_token=int_to_token,
        n_vocab=n_vocab,
        num_generate=args.num_events,
        strategy=args.strategy,
        temperature=args.temperature,
        random_seed=args.random_seed,
    )

    unique_tokens = len(
        set(generated_tokens)
    )

    print(
        f"Unique generated tokens: "
        f"{unique_tokens}/{len(generated_tokens)}"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Writing MIDI file...")

    write_midi(
        generated_tokens=generated_tokens,
        output_path=output_path,
    )

    print(
        f"Generated MIDI saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()