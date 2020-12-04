import gdown
import numpy as np
from PIL import Image
import tensorflow as tf
from .model import mirnet_model


class Inferer:

    def __init__(self):
        self.model = None

    @staticmethod
    def download_weights(file_id: str):
        gdown.download(
            'https://drive.google.com/uc?id={}'.format(file_id),
            'low_light_weights_best.h5', quiet=False
        )

    def build_model(
            self, train_crop_size:int, num_rrg: int,
            num_mrb: int, channels: int, weights_path: str):
        self.model = mirnet_model(
            image_size=train_crop_size, num_rrg=num_rrg,
            num_mrb=num_mrb, channels=channels
        )
        self.model.load_weights(weights_path)

    def infer(self, image_path, image_resize_factor=1):
        original_image = Image.open(image_path)
        width, height = original_image.size
        original_image = original_image.resize(
            (
                width // image_resize_factor,
                height // image_resize_factor
            ),
            Image.ANTIALIAS)
        image = tf.keras.preprocessing.image.img_to_array(original_image)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        output = self.model.predict(image)
        output_image = output[0] * 255.0
        output_image = output_image.clip(0, 255)
        output_image = output_image.reshape(
            (np.shape(output_image)[0], np.shape(output_image)[1], 3)
        )
        output_image = Image.fromarray(np.uint8(output_image))
        original_image = Image.fromarray(np.uint8(original_image))
        return original_image, output_image
