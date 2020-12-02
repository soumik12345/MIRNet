import tensorflow as tf


def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(
        tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3))
    )
