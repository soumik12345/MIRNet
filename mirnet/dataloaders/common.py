import tensorflow as tf


def read_images(image_files):
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.map(
        lambda x: tf.image.decode_png(x, channels=3),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset


def random_crop(low_image, enhanced_image, low_crop_size, enhanced_crop_size):
    low_image_shape = tf.shape(low_image)[:2]
    low_w = tf.random.uniform(
        shape=(), maxval=low_image_shape[1] - low_crop_size + 1, dtype=tf.int32)
    low_h = tf.random.uniform(
        shape=(), maxval=low_image_shape[0] - low_crop_size + 1, dtype=tf.int32)
    enhanced_w = low_w
    enhanced_h = low_h
    low_image_cropped = low_image[
                        low_h:low_h + low_crop_size,
                        low_w:low_w + low_crop_size
                        ]
    enhanced_image_cropped = enhanced_image[
                             enhanced_h:enhanced_h + enhanced_crop_size,
                             enhanced_w:enhanced_w + enhanced_crop_size
                             ]
    return low_image_cropped, enhanced_image_cropped


def random_flip(low_image, enhanced_image):
    return tf.cond(
        tf.random.uniform(shape=(), maxval=1) < 0.5,
        lambda: (low_image, enhanced_image),
        lambda: (
            tf.image.flip_left_right(low_image),
            tf.image.flip_left_right(enhanced_image)
        )
    )


def random_rotate(low_image, enhanced_image):
    condition = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(low_image, condition), tf.image.rot90(enhanced_image, condition)


def apply_scaling(low_image, enhanced_image):
    low_image = tf.cast(low_image, tf.float32)
    enhanced_image = tf.cast(enhanced_image, tf.float32)
    low_image = low_image / 255.0
    enhanced_image = enhanced_image / 255.0
    return low_image, enhanced_image
