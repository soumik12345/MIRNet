from typing import List
from .common import *


class LOLDataLoader:

    def __init__(self, images_lowlight: List[str], images_highlight: List[str]):
        self.images_lowlight = images_lowlight
        self.images_highlight = images_highlight

    def __len__(self):
        assert len(self.images_lowlight) == len(self.images_enhanced)
        return len(self.images_lowlight)

    def build_dataset(self, image_crop_size: int, batch_size: int, apply_transforms: bool):
        low_light_dataset = read_images(self.images_lowlight)
        high_light_dataset = read_images(self.images_highlight)
        dataset = tf.data.Dataset.zip((low_light_dataset, high_light_dataset))
        dataset = dataset.map(apply_scaling, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if apply_transforms:
            dataset = dataset.map(
                lambda low, high: random_crop(low, high, image_crop_size, image_crop_size),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            dataset = dataset.map(random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
