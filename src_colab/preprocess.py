import glob
import os

from music21 import converter, instrument, note, chord


def get_notes(data_dir="data/midi", limit=None):
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

    if not midi_files:
        raise FileNotFoundError(
            f"No MIDI files found in: {data_dir}"
        )

    print(f"Using {len(midi_files)} MIDI files")

    for index, file in enumerate(midi_files, start=1):
        print(
            f"[{index}/{len(midi_files)}] "
            f"{os.path.basename(file)}"
        )

        midi = converter.parse(file)

        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flatten().notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(
                    str(element.pitch)
                )

            elif isinstance(element, chord.Chord):
                notes.append(
                    ".".join(
                        str(n)
                        for n in element.normalOrder
                    )
                )

    print(
        f"Extracted {len(notes)} musical events"
    )

    return notes


if __name__ == "__main__":
    notes = get_notes(
        data_dir="data/midi",
        limit=10
    )

    print(notes[:20])
    print(f"Total events: {len(notes)}")