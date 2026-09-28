import glob

from music21 import converter, instrument, note, chord

notes = []

midi_files = glob.glob(
    "data/midi/**/*.midi",
    recursive=True
)

midi_files = midi_files[:10]

print(f"Found {len(midi_files)} MIDI files")

for file in midi_files:
    print(f"Parsing {file}")

    midi = converter.parse(file)

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
                ".".join(str(n) for n in element.normalOrder)
            )

print("\nFirst 20 notes/chords:")
print(notes[:20])

print(f"\nTotal notes/chords: {len(notes)}")

from dataset import build_vocabulary


pitchnames, note_to_int = build_vocabulary(notes)

print(f"Unique notes/chords: {len(pitchnames)}")

print("\nFirst 20 vocabulary items:")
print(pitchnames[:20])

print("\nExample mappings:")
for note_name in pitchnames[:10]:
    print(f"{note_name} -> {note_to_int[note_name]}")