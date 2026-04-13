import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45","L22","FW","R22","R45"]

# Q Network (must match training exactly)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(74,256),
        nn.ReLU(),
        nn.Linear(256,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,5)
        )

    def forward(self, x):
        return self.net(x)
# Device (CPU only for submission)

DEVICE = torch.device("cpu")

# Initialize model

MODEL = QNet().to(DEVICE)

# Load weights

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights (5).pth")
MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))

MODEL.eval()

def policy(obs, rng):
    """
    obs: numpy array of shape (74,)
    rng: random generator (not used)
    """

    # Ensure correct shape
    obs = np.array(obs, dtype=np.float32)

    # Convert to tensor
    obs = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        q_values = MODEL(obs)
        action = torch.argmax(q_values, dim=1).item()

    return action
