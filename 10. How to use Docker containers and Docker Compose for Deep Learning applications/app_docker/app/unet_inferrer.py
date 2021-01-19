import tensorflow as tf


class UnetInferrer:
    def __init__(self):
        self.image_size = 128

        self.saved_path = 'unet'
        self.model = tf.saved_model.load(self.saved_path)

        self.predict = self.model.signatures["serving_default"]

    def preprocess(self, image):
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return tf.cast(image, tf.float32) / 255.0

    def infer(self, image=None):
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = self.preprocess(tensor_image)
        shape= tensor_image.shape
        tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]])
        pred = self.predict(tensor_image)['conv2d_transpose_4']
        pred = pred.numpy().tolist()
        return {'segmentation_output':pred}
