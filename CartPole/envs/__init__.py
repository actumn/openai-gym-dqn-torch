import gym
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image

import torch
import torchvision.transforms as T

env = gym.make('CartPole-v0').unwrapped
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a named tuple representing a single transition in our environment
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Pre-process
# extracting, processing rendered images from environment.
resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])
screen_width = 600
view_width = 320
