import requests
from PIL import Image
import numpy as np

ENDPOINT_URL = "http://0.0.0.0:80/infer"

def infer():
    image = np.asarray(Image.open('resources/yorkshire_terrier.jpg')).astype(np.float32)
    data ={'image': image.tolist()}
    response = requests.post(ENDPOINT_URL, json = data)
    response.raise_for_status()
    print(response.json())


if __name__ =="__main__":
    infer()
