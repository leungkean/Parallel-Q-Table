import keras
from keras import layers


class MLPClassifier(keras.Model):
    def __init__(
        self,
        data_shape,
        num_classes,
        hidden_units,
        activation="relu",
        dropout=None,
    ):
        if len(data_shape) == 1:
            mask_shape = data_shape
        else:
            mask_shape = (*data_shape[:-1], 1)

        x = keras.Input(shape=data_shape, name="x")
        b = keras.Input(shape=mask_shape, name="b")

        x_o = layers.Multiply()([x, b])

        x_o_flat = layers.Flatten()(x_o)
        b_flat = layers.Flatten()(b)

        h = layers.Concatenate()([x_o_flat, b_flat])

        for n in hidden_units:
            h = layers.Dense(n, activation=activation)(h)
            if dropout is not None:
                h = layers.Dropout(dropout)(h)

        outputs = layers.Dense(num_classes)(h)
        inputs = {"x": x, "b": b}

        super().__init__(inputs, outputs)
