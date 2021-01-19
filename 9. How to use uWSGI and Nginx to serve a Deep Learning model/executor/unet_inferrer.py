import tensorflow as tf
import numpy as np

from utils.plot_image import display

from utils.config import Config

from configs.config import CFG


class UnetInferrer:
    def __init__(self):
        self.config = Config.from_json(CFG)
        self.image_size = self.config.data.image_size

        self.saved_path = '/home/aisummer/src/soft_eng_for_dl/saved_models/unet'
        self.model = tf.saved_model.load(self.saved_path)

        # print(list(    self.model.signatures.keys()))

        self.predict = self.model.signatures["serving_default"]
        # print(self.predict.structured_outputs)

    def preprocess(self, image):
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return tf.cast(image, tf.float32) / 255.0

    def infer(self, image=None):
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = self.preprocess(tensor_image)
        shape= tensor_image.shape
        tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]])
        print(tensor_image.shape)
        pred = self.predict(tensor_image)['conv2d_transpose_4']
        display([tensor_image[0], pred[0]])
        pred = pred.numpy().tolist()
        return {'segmentation_output':pred}
