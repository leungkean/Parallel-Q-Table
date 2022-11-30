import keras
from keras import layers


class CNNClassifier(keras.Model):
    def __init__(
        self,
        image_shape,
        num_classes,
        conv_layers,
        hidden_units,
        activation="leaky_relu",
        dropout=None,
    ):
        x = keras.Input(shape=image_shape, name="x")
        b = keras.Input(shape=(*image_shape[:-1], 1), name="b")

        x_o = layers.Multiply()([x, b])
        h = layers.Concatenate()([x_o, b])

        for filters, kernel, stride in conv_layers:
            h = layers.Conv2D(
                filters, kernel, stride, padding="SAME", activation=activation
            )(h)
            if dropout is not None:
                h = layers.Dropout(dropout)(h)

        h = layers.Flatten()(h)

        for n in hidden_units:
            h = layers.Dense(n, activation=activation)(h)

        outputs = layers.Dense(num_classes)(h)
        inputs = {"x": x, "b": b}

        super().__init__(inputs, outputs)
