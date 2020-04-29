import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import requests
import json
# import matplotlib.pyplot as plt

image = io.imread('./test.png')
image = rgb2gray(image)
image = resize(image, (28, 28))
image = image * 255
image = image.astype(np.uint8)
# io.imshow(image)
# plt.show()

headers = {'Content-Type': 'application/json'}
address = "http://127.0.0.1:2431/inference/pytorch"
data = {'images': image.tolist()}
result = requests.post(address, data=json.dumps(data), headers=headers)
print(str(result.content, encoding='utf-8'))

address = "http://127.0.0.1:2431/inference/albumentations"
data = {'images': image.tolist()}
result = requests.post(address, data=json.dumps(data), headers=headers)
print(str(result.content, encoding='utf-8'))