import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from music21 import chord, converter, note


RESULT_FIELDS = [
    "file",
    "total_events",
    "unique_patterns",
    "unique_pattern_ratio",
    "adjacent_repeats",
    "adjacent_repeat_rate",
    "longest_identical_run",
    "mean_pitch_jump_semitones",
]


def canonicalize_event(element):
    """
    Return a comparable event pattern and representative MIDI pitch.

    Notes use one integer MIDI pitch. Chords use a sorted tuple of MIDI
    pitches, while their representative pitch is the arithmetic mean.
    """
    if isinstance(element, note.Note):
        midi_pitch = int(element.pitch.midi)
        return midi_pitch, float(midi_pitch)

    if isinstance(element, chord.Chord):
        midi_pitches = tuple(
            sorted(
                int(pitch.midi)
                for pitch in element.pitches
            )
        )

        if not midi_pitches:
            raise ValueError("Encountered a chord with no pitches.")

        representative_pitch = float(
            np.mean(midi_pitches)
        )

        return midi_pitches, representative_pitch

    raise TypeError(
        f"Unsupported music21 event type: {type(element).__name__}"
    )


def longest_identical_run(patterns):
    longest_run = 1
    current_run = 1

    for previous, current in zip(
        patterns,
        patterns[1:]
    ):
        if current == previous:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run
            )
        else:
            current_run = 1

    return longest_run


def evaluate_midi(midi_path):
    """
    Evaluate event diversity, repetition, and consecutive pitch movement.

    Pattern and repeat rates use total parsed events as their denominator.
    A one-event file has no pitch transitions, so its mean jump is 0.0.
    """
    midi_path = Path(midi_path)

    if not midi_path.is_file():
        raise FileNotFoundError(
            f"MIDI file does not exist: {midi_path}"
        )

    try:
        midi = converter.parse(str(midi_path))
    except Exception as error:
        raise ValueError(
            f"Could not parse MIDI file {midi_path}: {error}"
        ) from error

    parsed_events = list(
        midi.flatten().notes
    )

    if not parsed_events:
        raise ValueError(
            f"MIDI file contains no note or chord events: {midi_path}"
        )

    canonical_events = [
        canonicalize_event(element)
        for element in parsed_events
    ]

    patterns = [
        pattern
        for pattern, _ in canonical_events
    ]
    representative_pitches = [
        representative_pitch
        for _, representative_pitch in canonical_events
    ]

    total_events = len(patterns)
    unique_patterns = len(set(patterns))
    adjacent_repeats = sum(
        current == previous
        for previous, current in zip(
            patterns,
            patterns[1:]
        )
    )

    pitch_jumps = [
        abs(current - previous)
        for previous, current in zip(
            representative_pitches,
            representative_pitches[1:]
        )
    ]

    mean_pitch_jump = (
        float(np.mean(pitch_jumps))
        if pitch_jumps
        else 0.0
    )

    return {
        "file": str(midi_path),
        "total_events": total_events,
        "unique_patterns": unique_patterns,
        "unique_pattern_ratio": round(
            unique_patterns / total_events,
            6
        ),
        "adjacent_repeats": adjacent_repeats,
        "adjacent_repeat_rate": round(
            adjacent_repeats / total_events,
            6
        ),
        "longest_identical_run": longest_identical_run(
            patterns
        ),
        "mean_pitch_jump_semitones": round(
            mean_pitch_jump,
            6
        ),
    }


def print_results(results):
    headers = [
        "File",
        "Events",
        "Unique",
        "Unique rate",
        "Adj. repeats",
        "Repeat rate",
        "Longest run",
        "Mean jump",
    ]

    rows = [
        [
            result["file"],
            str(result["total_events"]),
            str(result["unique_patterns"]),
            f'{result["unique_pattern_ratio"]:.3f}',
            str(result["adjacent_repeats"]),
            f'{result["adjacent_repeat_rate"]:.3f}',
            str(result["longest_identical_run"]),
            f'{result["mean_pitch_jump_semitones"]:.3f}',
        ]
        for result in results
    ]

    widths = [
        max(
            len(headers[index]),
            *(len(row[index]) for row in rows)
        )
        for index in range(len(headers))
    ]

    def format_row(row):
        return "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        )

    print(format_row(headers))
    print(format_row([
        "-" * width
        for width in widths
    ]))

    for row in rows:
        print(format_row(row))

    print(
        "\nMetrics describe event diversity, repetition, and pitch "
        "movement; they do not measure musical quality."
    )


def save_json(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as file:
        json.dump(results, file, indent=2)


def save_csv(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS
        )
        writer.writeheader()
        writer.writerows(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate objective event-level metrics for generated MIDI."
        )
    )
    parser.add_argument(
        "midi_files",
        nargs="+",
        help="One or more generated MIDI files to evaluate."
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for JSON results."
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path for CSV results."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    errors = []

    for midi_path in args.midi_files:
        try:
            results.append(
                evaluate_midi(midi_path)
            )
        except (FileNotFoundError, TypeError, ValueError) as error:
            errors.append(str(error))

    if results:
        print_results(results)

        if args.output_json:
            save_json(results, args.output_json)
            print(f"JSON results saved to {args.output_json}")

        if args.output_csv:
            save_csv(results, args.output_csv)
            print(f"CSV results saved to {args.output_csv}")

    for error in errors:
        print(f"Error: {error}", file=sys.stderr)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
