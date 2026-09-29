import argparse
import glob
import os

from music21 import converter, instrument, note, chord


def extract_notes_from_midi(file_path):
    notes = []

    midi = converter.parse(file_path)

    parts = instrument.partitionByInstrument(midi)

    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flatten().notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append(
                ".".join(
                    str(n)
                    for n in element.normalOrder
                )
            )

    return notes


def get_notes(data_dir="data/midi", limit=10):
    notes = []

    search_pattern = os.path.join(
        data_dir,
        "**",
        "*.midi"
    )

    midi_files = sorted(
        glob.glob(
            search_pattern,
            recursive=True
        )
    )

    if limit is not None:
        midi_files = midi_files[:limit]

    print(f"Found {len(midi_files)} MIDI files")

    for file in midi_files:
        print(f"Parsing {file}")

        notes.extend(
            extract_notes_from_midi(file)
        )

    return notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse MIDI files into baseline note/chord tokens."
    )
    parser.add_argument(
        "--data-dir",
        default="data/midi"
    )
    parser.add_argument(
        "--midi-limit",
        type=int,
        default=10
    )
    args = parser.parse_args()

    notes = get_notes(
        data_dir=args.data_dir,
        limit=args.midi_limit
    )

    print("\nFirst 20 notes/chords:")
    print(notes[:20])

    print(f"\nTotal notes/chords: {len(notes)}")
