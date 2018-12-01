import torch.nn as nn
import torch.nn.functional as F

from config import IMG_CHANNELS, NUM_OF_ACTIONS


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, NUM_OF_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # <-- flatten
        x = F.relu(self.fc1(x))
        x = self.head(x)
        return x
