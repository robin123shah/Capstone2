import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45","L22","FW","R22","R45"]

class QNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.net(x)

DEVICE = torch.device("cpu")

MODEL = QNet().to(DEVICE)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights (7).pth")
MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))

MODEL.eval()

def policy(obs, rng):

    obs = np.array(obs, dtype=np.float32)
    obs = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        q_values = MODEL(obs)
        action_idx = torch.argmax(q_values, dim=1).item()

    return ACTIONS[action_idx]