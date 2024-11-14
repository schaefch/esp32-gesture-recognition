import tensorflow as tf
import keras

# Convert model to tflite

model = keras.saving.load_model("./best_model.keras")

tf.saved_model.save(model, "saved_model")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
