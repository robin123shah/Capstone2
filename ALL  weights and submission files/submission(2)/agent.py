import torch
import torch.nn as nn
import numpy as np

ACTIONS = ["L45","L22","FW","R22","R45"]

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        # MUST match saved keys
        self.policy_net = nn.Sequential(
            nn.Linear(18,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh()
        )

        self.action_net = nn.Linear(64,5)

    def forward(self, x):
        x = self.policy_net(x)
        return self.action_net(x)


model = Policy()

ckpt = torch.load("weights (4).pth", map_location="cpu")
model.policy_net.load_state_dict(ckpt["policy_net"])
model.action_net.load_state_dict(ckpt["action_net"])

model.eval()


def policy(obs, rng=None):
    state = np.array(obs, dtype=np.float32)
    state = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        logits = model(state)
        action = torch.argmax(logits, dim=1).item()

    return ACTIONS[action]