from keras.callbacks import ModelCheckpoint

from preprocess import get_notes
from dataset import (
    build_vocabulary,
    create_sequences,
    prepare_sequences,
)
from model import create_model


SEQUENCE_LENGTH = 100
MIDI_LIMIT = 10
EPOCHS = 5
BATCH_SIZE = 64


def train():
    # 1. Load and parse MIDI data
    notes = get_notes(limit=MIDI_LIMIT)

    # 2. Build vocabulary
    pitchnames, note_to_int = build_vocabulary(notes)
    n_vocab = len(pitchnames)

    print(f"\nVocabulary size: {n_vocab}")

    # 3. Build input/target sequences
    network_input, network_output = create_sequences(notes, note_to_int, sequence_length=SEQUENCE_LENGTH)

    # 4. Convert into tensors for the LSTM
    network_input, network_output = prepare_sequences(network_input, network_output, n_vocab)

    print("Input shape:", network_input.shape)
    print("Output shape:", network_output.shape)

    # 5. Build model
    model = create_model(sequence_length=SEQUENCE_LENGTH, n_vocab=n_vocab)

    model.summary()

    # 6. Save the best model based on validation loss
    checkpoint = ModelCheckpoint(
        filepath="checkpoints/model-{epoch:02d}-{val_loss:.4f}.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    # 7. Train
    history = model.fit(
        network_input,
        network_output,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[checkpoint]
    )

    return history


if __name__ == "__main__":
    train()