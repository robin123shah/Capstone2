import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45","L22","FW","R22","R45"]

# Q Network (must match training exactly)
class QNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(18, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.head = nn.Linear(64, 5)

    def forward(self, x):
        x = self.feature(x)
        return self.head(x)

# Use CPU for evaluation
DEVICE = torch.device("cpu")

# Load model
MODEL = QNet().to(DEVICE)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights_101.pth")
MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))

MODEL.eval()

def policy(obs, rng):

    obs = np.array(obs, dtype=np.float32)
    obs = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        q_values = MODEL(obs)
        action_idx = torch.argmax(q_values, dim=1).item()

    # 🔥 convert index → action string
    return ACTIONS[action_idx]