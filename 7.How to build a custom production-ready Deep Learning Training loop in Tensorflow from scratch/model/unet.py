# -*- coding: utf-8 -*-
"""Unet model"""

# standard library

# external
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from dataloader.dataloader import DataLoader
from utils.logger import get_logger
from executor.unet_trainer import UnetTrainer

# internal
from .base_model import BaseModel

LOG = get_logger('unet')


class UNet(BaseModel):
    """Unet Model Class"""

    def __init__(self, config):
        super().__init__(config)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.config.model.input, include_top=False)
        self.model = None
        self.output_channels = self.config.model.output

        self.dataset = None
        self.info = None
        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.epoches = self.config.train.epoches
        self.val_subsplits = self.config.train.val_subsplits
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0

        self.image_size = self.config.data.image_size
        self.train_dataset = []
        self.test_dataset = []

    def load_data(self):
        """Loads and Preprocess data """
        LOG.info(f'Loading {self.config.data.path} dataset...')
        self.dataset, self.info = DataLoader().load_data(self.config.data)
        self.train_dataset, self.test_dataset = DataLoader.preprocess_data(self.dataset, self.batch_size,
                                                                           self.buffer_size, self.image_size)
        self._set_training_parameters()

    def _set_training_parameters(self):
        """Sets training parameters"""
        self.train_length = self.info.splits['train'].num_examples
        self.steps_per_epoch = self.train_length // self.batch_size
        self.validation_steps = self.info.splits['test'].num_examples // self.batch_size // self.val_subsplits

    def build(self):
        """ Builds the Keras model based """
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        layers = [self.base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=layers)

        down_stack.trainable = False

        up_stack = [
            pix2pix.upsample(self.config.model.up_stack.layer_1, self.config.model.up_stack.kernels),  # 4x4 -> 8x8
            pix2pix.upsample(self.config.model.up_stack.layer_2, self.config.model.up_stack.kernels),  # 8x8 -> 16x16
            pix2pix.upsample(self.config.model.up_stack.layer_3, self.config.model.up_stack.kernels),  # 16x16 -> 32x32
            pix2pix.upsample(self.config.model.up_stack.layer_4, self.config.model.up_stack.kernels),  # 32x32 -> 64x64
        ]

        inputs = tf.keras.layers.Input(shape=self.config.model.input)
        x = inputs

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            self.output_channels, self.config.model.up_stack.kernels, strides=2,
            padding='same')  # 64x64 -> 128x128

        x = last(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)

        LOG.info('Keras Model was built successfully')

    def train(self):
        """Compiles and trains the model"""
        LOG.info('Training started')
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()

        trainer = UnetTrainer(self.model, self.train_dataset, loss, optimizer, metrics, self.epoches)
        trainer.train()

    def evaluate(self):
        """Predicts resuts for the test dataset"""

        predictions = []
        LOG.info(f'Predicting segmentation map for test dataset')

        for im in self.test_dataset.as_numpy_iterator():
            DataLoader().validate_schema(im[0])
            break

        for image, mask in self.test_dataset:
            tf.print(image)
            # LOG.info(f'Predicting segmentation map {image}')
            predictions.append(self.model.predict(image))
        return predictions
