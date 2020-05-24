# -*- coding: utf-8 -*-
"""Unet model"""

# standard library

# internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader

# external
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


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
        self.dataset, self.info = DataLoader().load_data(self.config.data)
        self._preprocess_data()

    def _preprocess_data(self):
        """ Splits into training and test and set training parameters"""
        train = self.dataset['train'].map(self._load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = self.dataset['test'].map(self._load_image_test)

        self.train_dataset = train.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
        self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.test_dataset = test.batch(self.batch_size)

        self._set_training_parameters()

    def _set_training_parameters(self):
        """Sets training parameters"""
        self.train_length = self.info.splits['train'].num_examples
        self.steps_per_epoch = self.train_length // self.batch_size
        self.validation_steps = self.info.splits['test'].num_examples // self.batch_size // self.val_subsplits

    def _normalize(self, input_image, input_mask):
        """ Normalise input image
        Args:
            input_image (tf.image): The input image
            input_mask (int): The image mask

        Returns:
            input_image (tf.image): The normalized input image
            input_mask (int): The new image mask
        """
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    @tf.function
    def _load_image_train(self, datapoint):
        """ Loads and preprocess  a single training image """
        input_image = tf.image.resize(datapoint['image'], (self.image_size, self.image_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.image_size, self.image_size))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = self._normalize(input_image, input_mask)

        return input_image, input_mask

    def _load_image_test(self, datapoint):
        """ Loads and preprocess a single test images"""

        input_image = tf.image.resize(datapoint['image'], (self.image_size, self.image_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.image_size, self.image_size))

        input_image, input_mask = self._normalize(input_image, input_mask)

        return input_image, input_mask

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

    def train(self):
        """Compiles and trains the model"""
        self.model.compile(optimizer=self.config.train.optimizer.type,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=self.config.train.metrics)

        model_history = self.model.fit(self.train_dataset, epochs=self.epoches,
                                       steps_per_epoch=self.steps_per_epoch,
                                       validation_steps=self.validation_steps,
                                       validation_data=self.test_dataset)

        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        predictions = []
        for image, mask in self.dataset.take(1):
            predictions.append(self.model.predict(image))

        return predictions
