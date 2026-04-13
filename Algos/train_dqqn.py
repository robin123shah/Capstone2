import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from obelix import OBELIX

ACTIONS = ["L45","L22","FW","R22","R45"]

# Q Network
class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(18,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,5)
        )

    def forward(self,x):
        return self.net(x)


# Hyperparameters
gamma = 0.99
lr = 1e-3
batch_size = 64
buffer_size = 50000
episodes = 1000

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

target_update = 10


# Replay buffer
replay_buffer = deque(maxlen=buffer_size)


# Select action
def select_action(state, model):

    if random.random() < epsilon:
        return random.randint(0,4)

    state = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        q_values = model(state)

    return torch.argmax(q_values).item()


# Train step
def train_step(model, target_model, optimizer):

    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()

    next_actions = model(next_states).argmax(1)
    next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

    target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main training
def train():

    global epsilon

    env = OBELIX(scaling_factor=1)

    model = QNetwork()
    target_model = QNetwork()

    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(episodes):

        state = env.reset()

        done = False
        total_reward = 0

        while not done:

            action = select_action(state, model)

            next_state, reward, done = env.step(ACTIONS[action], render=False)

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            train_step(model, target_model, optimizer)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode} Reward {total_reward} Epsilon {epsilon}")

    torch.save(model.state_dict(),"weights.pth")

    print("Training complete. weights.pth saved.")


if __name__ == "__main__":
    train()