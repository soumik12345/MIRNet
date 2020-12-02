import os
import tensorflow as tf
from typing import List
from .utils import psnr
from .model import mirnet_model
from .losses import charbonnier_loss
from wandb.keras import WandbCallback
from .dataloaders import LOLDataLoader


class LowLightTrainer:

    def __init__(self):
        self.model = None
        self.crop_size = None
        self.train_dataset = None
        self.valid_dataset = None
        # self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        # if len(tf.config.list_physical_devices('GPU')) > 1:
        #     self.strategy = tf.distribute.MirroredStrategy()

    def build_dataset(
            self, train_low_light_images: List[str], train_high_light_images: List[str],
            valid_low_light_images: List[str], valid_high_light_images: List[str],
            crop_size: int, batch_size: int):
        self.crop_size = crop_size
        self.train_dataset = LOLDataLoader(
            images_lowlight=train_low_light_images,
            images_highlight=train_high_light_images
        ).build_dataset(
            image_crop_size=crop_size, batch_size=batch_size, is_dataset_train=True)
        self.valid_dataset = LOLDataLoader(
            images_lowlight=valid_low_light_images,
            images_highlight=valid_high_light_images
        ).build_dataset(
            image_crop_size=crop_size, batch_size=batch_size, is_dataset_train=False)

    def compile(self, num_rrg=3, num_mrb=2, channels=64, learning_rate=1e-4, use_mae_loss=True):
        self.model = mirnet_model(self.crop_size, num_rrg, num_mrb, channels)
        loss_function = tf.keras.losses.MeanAbsoluteError() if use_mae_loss else charbonnier_loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=[psnr])

    def train(self, epochs: int, checkpoint_dir: str):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_psnr",
                patience=10, mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_psnr', factor=0.5,
                patience=5, verbose=1, min_delta=1e-7, mode='max'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(checkpoint_dir, 'low_light_weights_best.h5'),
                monitor="val_psnr", save_weights_only=True,
                mode="max", save_best_only=True, save_freq=1
            ), WandbCallback()
        ]
        history = self.model.fit(
            self.train_dataset, validation_data=self.valid_dataset,
            epochs=epochs, callbacks=callbacks, verbose=1
        )
        return history
