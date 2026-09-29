import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from music21 import chord, converter, note, stream


def load_json(path):
    """Load a JSON file from disk."""
    with open(path, "r") as file:
        return json.load(file)


def load_vocabulary(vocabulary_path):
    """
    Load the vocabulary saved during training.

    Returns:
        token_to_int: maps musical token -> integer ID
        int_to_token: maps integer ID -> musical token
    """
    vocab_data = load_json(vocabulary_path)

    # Case 1:
    # vocabulary.json is simply a list:
    #
    # ["C4", "D4", "0.4.7", ...]
    if isinstance(vocab_data, list):
        token_to_int = {
            token: index
            for index, token in enumerate(vocab_data)
        }

    # Case 2:
    # vocabulary.json contains a saved mapping.
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
            # Assume the dictionary itself is:
            #
            # {
            #     "C4": 0,
            #     "D4": 1,
            #     ...
            # }
            token_to_int = {
                token: int(index)
                for token, index in vocab_data.items()
            }

    else:
        raise ValueError(
            "Unsupported vocabulary.json format."
        )

    int_to_token = {
        index: token
        for token, index in token_to_int.items()
    }

    return token_to_int, int_to_token


def extract_events_from_midi(midi_path):
    """
    Parse a MIDI file using the same event representation used during training.

    Notes:
        C4
        F#5

    Chords:
        0.4.7
        2.5.9
    """
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
    """
    Find a contiguous sequence of sequence_length events where every
    token exists in the training vocabulary.
    """
    valid_start_positions = []

    max_start = len(events) - sequence_length

    for start in range(max_start + 1):
        window = events[start:start + sequence_length]

        if all(token in token_to_int for token in window):
            valid_start_positions.append(start)

    if not valid_start_positions:
        raise ValueError(
            "Could not find a valid seed sequence in this MIDI file. "
            "The seed must contain at least "
            f"{sequence_length} consecutive events from the vocabulary."
        )

    rng = random.Random(random_seed)

    start = rng.choice(valid_start_positions)

    seed_events = events[start:start + sequence_length]

    print(f"Found {len(valid_start_positions)} valid seed windows.")
    print(f"Selected seed starting at event {start}.")

    return seed_events


def events_to_pattern(seed_events, token_to_int):
    """
    Convert musical tokens into integer IDs.
    """
    return [
        token_to_int[event]
        for event in seed_events
    ]


def prepare_model_input(pattern, n_vocab):
    """
    Convert integer token IDs into the normalized format expected
    by the LSTM.

    Shape:
        (100,)
          ->
        (1, 100, 1)
    """
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


def generate_tokens(
    model,
    seed_pattern,
    int_to_token,
    n_vocab,
    num_generate=500,
):
    """
    Generate new musical events autoregressively.

    For each prediction:

    1. normalize current 100-event pattern
    2. call model.predict()
    3. select highest-probability token
    4. append prediction
    5. remove oldest event
    6. repeat
    """
    pattern = list(seed_pattern)

    generated_tokens = []

    for step in range(num_generate):

        model_input = prepare_model_input(
            pattern,
            n_vocab,
        )

        prediction = model.predict(
            model_input,
            verbose=0,
        )[0]

        prediction_index = int(
            np.argmax(prediction)
        )

        predicted_token = int_to_token[
            prediction_index
        ]

        generated_tokens.append(
            predicted_token
        )

        # Slide the sequence window forward.
        pattern.append(prediction_index)
        pattern = pattern[1:]

        if (step + 1) % 50 == 0:
            print(
                f"Generated {step + 1}/{num_generate} events"
            )

    return generated_tokens


def token_to_music21(token, duration=0.5):
    """
    Convert one predicted token back into a music21 object.

    Notes:
        "C4" -> music21.note.Note

    Chords:
        "0.4.7" -> music21.chord.Chord

    The chord representation only contains pitch classes, so octave
    information must be reconstructed. Here, pitch classes are placed
    around MIDI octave 4 using MIDI pitches 60-71.
    """

    # Chord token
    if "." in token:
        pitch_classes = [
            int(value)
            for value in token.split(".")
        ]

        # Place pitch classes beginning at middle C (MIDI 60).
        midi_pitches = [
            60 + pitch_class
            for pitch_class in pitch_classes
        ]

        musical_object = chord.Chord(
            midi_pitches
        )

    # Note token
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
    """
    Convert generated tokens into a MIDI file.

    Because the training representation does not currently store
    duration information, each generated event receives the same
    duration.
    """
    output_stream = stream.Stream()

    for token in generated_tokens:
        musical_object = token_to_music21(
            token,
            duration=duration,
        )

        output_stream.append(
            musical_object
        )

    output_stream.write(
        "midi",
        fp=str(output_path),
    )


def get_sequence_length(training_config):
    """
    Retrieve sequence length from training_config.json.

    Falls back to 100 because that is the current experiment setting.
    """
    possible_keys = [
        "sequence_length",
        "sequence_len",
        "seq_length",
    ]

    for key in possible_keys:
        if key in training_config:
            return int(training_config[key])

    print(
        "Warning: sequence length not found in training_config.json. "
        "Using 100."
    )

    return 100


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIDI music using the trained LSTM model."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to best_model.keras",
    )

    parser.add_argument(
        "--vocabulary",
        required=True,
        help="Path to vocabulary.json",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to training_config.json",
    )

    parser.add_argument(
        "--seed-midi",
        required=True,
        help="MIDI file used to obtain the initial 100-event seed.",
    )

    parser.add_argument(
        "--output",
        default="generated_music.mid",
        help="Output MIDI path.",
    )

    parser.add_argument(
        "--num-events",
        type=int,
        default=500,
        help="Number of musical events to generate.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when choosing the seed window.",
    )

    args = parser.parse_args()

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
        f"Seed MIDI contains {len(seed_file_events)} musical events."
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

    print("Generating music...")

    generated_tokens = generate_tokens(
        model=model,
        seed_pattern=seed_pattern,
        int_to_token=int_to_token,
        n_vocab=n_vocab,
        num_generate=args.num_events,
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
        f"Generated MIDI saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()