import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from obelix import OBELIX

ACTIONS = ["L45","L22","FW","R22","R45"]

# -----------------------------
# DUELING NETWORK
# -----------------------------
class QNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(18,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,5)
        )

    def forward(self,x):

        x = self.feature(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q


# -----------------------------
# HYPERPARAMETERS
# -----------------------------
gamma = 0.99
lr = 1e-3
batch_size = 64
buffer_size = 100000
episodes = 3000

epsilon = 1.0
epsilon_decay = 0.998
epsilon_min = 0.05

target_update = 10


# -----------------------------
# REPLAY BUFFER
# -----------------------------
replay_buffer = deque(maxlen=buffer_size)


# -----------------------------
# ACTION SELECTION
# -----------------------------
def select_action(state, model):

    global epsilon

    if random.random() < epsilon:
        return random.randint(0,4)

    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state)

    return torch.argmax(q_values).item()


# -----------------------------
# TRAIN STEP
# -----------------------------
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


# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------
def train():

    global epsilon

    model = QNetwork()
    target_model = QNetwork()

    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(episodes):

        # Randomize difficulty during training
        difficulty = random.choice([0,2,3])
        wall = random.random() < 0.3

        env = OBELIX(
            scaling_factor=1,
            difficulty=difficulty,
            wall_obstacles=wall,
            max_steps=400
        )

        state = env.reset()

        done = False
        total_reward = 0

        while not done:

            action = select_action(state, model)

            next_state, reward, done = env.step(ACTIONS[action], render=False)

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            # train twice per step
            for _ in range(2):
                train_step(model, target_model, optimizer)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode} Reward {total_reward:.2f} Epsilon {epsilon:.4f}")

    torch.save(model.state_dict(),"weights.pth")

    print("\nTraining complete. weights.pth saved.")


# -----------------------------
# RUN TRAINING
# -----------------------------
if __name__ == "__main__":
    train()