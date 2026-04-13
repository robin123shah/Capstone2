"""
Submission agent using trained weights (DQN 64-64 architecture)

Make sure:
- weights.pth is in the same folder
- weights are trained using this EXACT architecture
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    import torch
    import torch.nn as nn

    # ✅ EXACT architecture expected
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(18, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
            )

        def forward(self, x):
            return self.net(x)

    model = Net()

    # load weights safely
    state_dict = torch.load(wpath, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    _MODEL = model

    print("✅ Model loaded successfully")


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q_values = _MODEL(x).squeeze(0).numpy()

    action = int(np.argmax(q_values))
    return ACTIONS[action]