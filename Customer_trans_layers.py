import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class PositionalIndex(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        bs = tf.shape(x)[0]  # Extract batch size
        number_of_vectors = tf.shape(x)[1]  # Count number of vectors
        indices = tf.range(number_of_vectors)  # Generate indices
        indices = tf.expand_dims(indices, 0)  # Reshape for batch processing
        return tf.tile(indices, [bs, 1])  # Repeat for each batch

    def get_config(self):
        """Ensure the layer can be serialized."""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Allow reloading from config."""
        return cls(**config)


@register_keras_serializable()
class ClassTokenIndex(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        bs = tf.shape(x)[0]  # Extract batch size
        number_of_vectors = 1  # Only generate 1 vector for the class token
        indices = tf.range(number_of_vectors)  # Generate index
        indices = tf.expand_dims(indices, 0)  # Reshape for batch processing
        return tf.tile(indices, [bs, 1])  # Repeat for each batch

    def get_config(self):
        """Ensure the layer can be serialized."""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Allow reloading from config."""
        return cls(**config)
