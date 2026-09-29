from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense


def create_model(sequence_length, n_vocab):
    model = Sequential()

    model.add(LSTM(512, input_shape=(sequence_length, 1), return_sequences=True))

    model.add(Dropout(0.3))

    model.add(LSTM(512, return_sequences=False))

    model.add(Dropout(0.3))

    model.add(Dense(256, activation="relu"))

    model.add(Dense(n_vocab, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model

