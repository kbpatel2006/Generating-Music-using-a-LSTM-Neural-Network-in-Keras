from keras.models import Sequential

from keras.layers import (
    Input,
    LSTM,
    Dropout,
    Dense
)


def create_model(
    sequence_length,
    n_vocab,
    lstm_units=512,
    dropout_rate=0.3
):
    model = Sequential([
        Input(
            shape=(sequence_length, 1)
        ),

        LSTM(
            lstm_units,
            return_sequences=True
        ),

        Dropout(
            dropout_rate
        ),

        LSTM(
            lstm_units,
            return_sequences=False
        ),

        Dropout(
            dropout_rate
        ),

        Dense(
            256,
            activation="relu"
        ),

        Dense(
            n_vocab,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = create_model(
        sequence_length=100,
        n_vocab=741
    )

    model.summary()