import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gc
from collections import deque

from obelix import OBELIX


ACTIONS = ["L45","L22","FW","R22","R45"]

NUM_ENVS = 4


# -----------------------------
# Q NETWORK
# -----------------------------
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


# -----------------------------
# HYPERPARAMETERS
# -----------------------------
gamma = 0.99
lr = 1e-3

batch_size = 128
buffer_size = 30000
episodes = 2000

epsilon = 1.0
epsilon_decay = 0.996
epsilon_min = 0.05

target_update = 20


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

    # faster tensor creation
    states = torch.tensor(np.array(states), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

    actions = torch.tensor(np.array(actions))
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
    dones = torch.tensor(np.array(dones), dtype=torch.float32)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()

    next_actions = model(next_states).argmax(1)
    next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

    target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# -----------------------------
# CREATE ENVIRONMENTS
# -----------------------------
def create_envs(episode):

    envs = []

    for i in range(NUM_ENVS):

        # faster curriculum
        if episode < 400:
            difficulty = 0
        elif episode < 900:
            difficulty = [0,2][i % 2]
        else:
            difficulty = [0,2,3,3][i % 4]

        wall = random.random() < 0.3

        env = OBELIX(
            scaling_factor=1,
            difficulty=difficulty,
            wall_obstacles=wall,
            max_steps=150
        )

        envs.append(env)

    return envs


# -----------------------------
# TRAINING LOOP
# -----------------------------
def train():

    global epsilon

    model = QNetwork()
    target_model = QNetwork()

    # try loading checkpoint
    try:
        model.load_state_dict(torch.load("checkpoint.pth"))
        print("Checkpoint loaded successfully")
    except:
        print("No checkpoint found, starting fresh")

    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)

    step_counter = 0

    for episode in range(episodes):

        # reset exploration when difficulty increases
        if episode == 400 or episode == 900:
            epsilon = 1.0

        envs = create_envs(episode)
        states = [env.reset() for env in envs]

        done_flags = [False]*NUM_ENVS
        total_rewards = [0]*NUM_ENVS

        while not all(done_flags):

            for i, env in enumerate(envs):

                if done_flags[i]:
                    continue

                action = select_action(states[i], model)

                next_state, reward, done = env.step(ACTIONS[action], render=False)

                replay_buffer.append((states[i], action, reward, next_state, done))

                states[i] = next_state
                total_rewards[i] += reward
                done_flags[i] = done

            step_counter += 1

            # train less frequently
            if step_counter % 4 == 0:
                train_step(model, target_model, optimizer)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 200 == 0:
            torch.save(model.state_dict(),"checkpoint.pth")
            gc.collect()

        print(
            f"Episode {episode} | "
            f"Mean Reward {np.mean(total_rewards):.2f} | "
            f"Max Reward {np.max(total_rewards):.2f} | "
            f"Epsilon {epsilon:.3f}"
        )

    torch.save(model.state_dict(),"weights.pth")

    print("\nTraining complete. weights.pth saved.")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    train()
