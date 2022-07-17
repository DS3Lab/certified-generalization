import torch
import os

from constants import *
from model_factory import get_architecture

weights_path = 'results/data/imagenet/efficientnet_b7/'
model = get_architecture(EFFICENTNETB7, 'imagenet', load_weights=True, use_cuda=False)
state_dict = model.state_dict()
# state_dict = {'1' + str(k)[1:]: v for k, v in state_dict.items()}
torch.save({'arch': EFFICENTNETB7, 'state_dict': state_dict}, os.path.join(weights_path, 'checkpoint.pth.tar'))
