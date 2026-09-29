import glob
import os

from music21 import converter, instrument, note, chord


def extract_notes_from_midi(file_path):
    """
    Parse one MIDI file and return its sequence of note/chord tokens.
    """
    notes = []

    midi = converter.parse(file_path)

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

    return notes


def get_pieces(
    data_dir="data/midi",
    limit=None
):
    """
    Find MIDI files and return each piece as its own token sequence.

    Keeping pieces separate prevents training windows from crossing
    from the end of one composition into the beginning of another.
    """
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

    print(
        f"Using {len(midi_files)} MIDI files"
    )

    pieces = []

    for index, file_path in enumerate(
        midi_files,
        start=1
    ):
        print(
            f"[{index}/{len(midi_files)}] "
            f"{os.path.basename(file_path)}"
        )

        notes = extract_notes_from_midi(
            file_path
        )

        if notes:
            pieces.append(notes)

    total_events = sum(
        len(piece)
        for piece in pieces
    )

    print(
        f"Extracted {total_events} musical events "
        f"across {len(pieces)} pieces"
    )

    return pieces


if __name__ == "__main__":
    pieces = get_pieces(
        data_dir="data/midi",
        limit=10
    )

    print(
        f"\nPieces loaded: {len(pieces)}"
    )

    if pieces:
        print(
            "\nFirst 20 events from first piece:"
        )

        print(
            pieces[0][:20]
        )