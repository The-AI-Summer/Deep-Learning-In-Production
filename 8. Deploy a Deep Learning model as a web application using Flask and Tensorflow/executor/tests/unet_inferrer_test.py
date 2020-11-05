import unittest

from PIL import Image
import numpy as np

from executor.unet_inferrer import UnetInferrer


class MyTestCase(unittest.TestCase):
    def test_infer(self):
        image = np.asarray(Image.open('../../../Deep-Learning-In-Production/8. Deploy a Deep Learning model as a web application using Flask and Tensorflow/app/resources/yorkshire_terrier.jpg')).astype(np.float32)
        inferrer = UnetInferrer()
        inferrer.infer(image)


if __name__ == '__main__':
    unittest.main()
