import tensorflow.keras as keras

FILTERS = 32
KERNEL_SIZE = 3


# Adapted from
# https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(
        filters=FILTERS, kernel_size=KERNEL_SIZE, padding="same"
    )(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(
        filters=FILTERS, kernel_size=KERNEL_SIZE, padding="same"
    )(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    gap = keras.layers.GlobalAveragePooling1D()(conv2)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)
