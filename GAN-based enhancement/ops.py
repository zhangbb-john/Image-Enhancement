import math
import numpy as np
import tensorflow as tf

class linear(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(linear, self).__init__(*args, **kwargs)

    def build(self, input_,output_size,stddev=0.02,bias_start=0.0):
        shape=input_.get_shape.as_list()
        self.matrix = self.add_weight(
            shape=[shape[1],output_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(stddev=stddev)
            )
        self.bias=self.add_weight(
            shape=[output_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(bias_start)
            )

    @tf.function
    def call(self, input_,with_w):
        if with_w:
            return tf.matmul(input_, self.matrix) + self.bias, self.matrix, self.bias
        else:
            return tf.matmul(input_, self.matrix) + self.bias
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)