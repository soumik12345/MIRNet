import tensorflow as tf


def charbonnier_loss(y_true, y_pred):
    epsilon = 1e-3
    error = y_true - y_pred
    p = tf.sqrt(tf.square(error) + tf.square(epsilon))
    return tf.reduce_mean(p)
