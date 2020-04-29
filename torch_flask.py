import numpy as np
import torch
import albumentations
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from flask import Flask, request

from mnist_runner import Net

model = Net()
model.load_state_dict(torch.load('mnist_cnn.pt'), strict=False)
model.eval()

app = Flask(__name__)

# Pytorch transformer
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
])


# albumentations transformer
normalize2 = albumentations.Compose([
    albumentations.Normalize(mean=(0.1307,), std=(0.3081,)),
    albumentations.pytorch.ToTensorV2(),
])


# pytorch transformer
@app.route('/inference/pytorch', methods=['POST'])
def inference_pytorch():
    # data is uint8 ndarray
    data = request.json
    data = np.array(data['images'])
    data = data / 255.0
    data = normalize(data)
    print(data)
    result = model.forward(
        data.unsqueeze(0).float()
    )
    return str(result.argmax().item())


# albumentations transformer
@app.route('/inference/albumentations', methods=['POST'])
def inference_albumentations():
    # data is uint8 ndarray
    data = request.json
    data = np.array(data['images'])
    data = np.expand_dims(data, 2)
    data = normalize2(image=data)['image']
    print(data)
    result = model.forward(
        data.unsqueeze(0).float()
    )
    return str(result.argmax().item())


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2431, threaded=False)