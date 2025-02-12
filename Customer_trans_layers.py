import tensorflow as tf


class PositionalIndex(tf.keras.layers.Layer):
    def call(self, x):
        bs = tf.shape(x)[0]  # extract batch size
        number_of_vectors = tf.shape(x)[1]  # count number of vectors
        indices = tf.range(number_of_vectors)  # generate indices
        indices = tf.expand_dims(indices, 0)  # reshape for batch processing
        return tf.tile(indices, [bs, 1])  # repeat for each batch

    def get_config(self):
        return super().get_config()  # required for serialization


class ClassTokenIndex(tf.keras.layers.Layer):
    def call(self, x):
        bs = tf.shape(x)[0]  # extract batch size
        number_of_vectors = 1  # only generate 1 vector for the class token
        indices = tf.range(number_of_vectors)  # generate index
        indices = tf.expand_dims(indices, 0)  # reshape for batch processing
        return tf.tile(indices, [bs, 1])  # repeat for each batch

    def get_config(self):
        return super().get_config()  # required for serialization
