from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Input,
    LSTM
)

from keras.models import Sequential


def create_model(
    sequence_length,
    n_vocab,
    lstm_units=512,
    dropout_rate=0.3,
    input_representation="scalar",
    embedding_dim=128,
):
    if input_representation not in {
        "scalar",
        "embedding",
    }:
        raise ValueError(
            "Unsupported input representation: "
            f"{input_representation!r}. Expected "
            "'scalar' or 'embedding'."
        )

    if embedding_dim <= 0:
        raise ValueError(
            "Embedding dimension must be greater than 0."
        )

    if input_representation == "scalar":
        input_layers = [
            Input(
                shape=(
                    sequence_length,
                    1
                )
            )
        ]
    else:
        input_layers = [
            Input(
                shape=(sequence_length,),
                dtype="int32"
            ),
            Embedding(
                input_dim=n_vocab,
                output_dim=embedding_dim
            )
        ]

    model = Sequential(input_layers + [
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

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ]
    )

    return model


if __name__ == "__main__":
    model = create_model(
        sequence_length=100,
        n_vocab=1102
    )

    model.summary()
