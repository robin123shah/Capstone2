
"""
Submission agent for OBELIX.

Loads trained weights and returns actions using the neural network policy.
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
    weights_path = os.path.join(submission_dir, "weights.pth")

    import torch
    import torch.nn as nn

    class QNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(18,64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,5)
            )

        def forward(self,x):
            return self.net(x)

    model = QNetwork()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Called by evaluator every step.
    Returns an action string.
    """

    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q_values = _MODEL(x).squeeze(0).numpy()

    action_index = int(np.argmax(q_values))

    return ACTIONS[action_index]
