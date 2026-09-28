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

from dataset import build_vocabulary, create_sequences, prepare_sequences


pitchnames, note_to_int = build_vocabulary(notes)
n_vocab = len(pitchnames)

print(f"Unique notes/chords: {len(pitchnames)}")

network_input, network_output = create_sequences(notes, note_to_int, sequence_length=100)
network_input, network_output = prepare_sequences(network_input, network_output, n_vocab)

print("\nPrepared input shape:")
print(network_input.shape)

print("\nPrepared output shape:")
print(network_output.shape)

print(f"\nTotal sequences: {len(network_input)}")
print("\nFirst input sequence:")
print(network_input[0][:20])
print("\nFirst output:")
print(network_output[0])