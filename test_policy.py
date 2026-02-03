# test_policy.py
import gym
import torch
import numpy as np
import os

from models import WorldModel, Actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    obs, _ = env.reset()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    world_model = WorldModel(obs_dim, action_dim, latent_dim=64, hidden_dim=128).to(DEVICE)
    actor = Actor(latent_dim=64, action_dim=action_dim, hidden_dim=128).to(DEVICE)

    # Load models from fixed checkpoint files
    if os.path.exists("world_model.pt"):
        world_model.load_state_dict(torch.load("world_model.pt", map_location=DEVICE))
        print("Loaded world model from world_model.pt")
    else:
        print("Warning: world_model.pt not found!")
        return

    if os.path.exists("actor.pt"):
        actor.load_state_dict(torch.load("actor.pt", map_location=DEVICE))
        print("Loaded actor from actor.pt")
    else:
        print("Warning: actor.pt not found!")
        return

    world_model.eval()
    actor.eval()

    done = False
    total_reward = 0.0

    with torch.no_grad():
        h = world_model.rssm.init_hidden(1, DEVICE)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        z, _, _, h = world_model.rssm.posterior(obs_t, h)

        while True:
            dist = actor(z)
            action = dist.sample().item()
            print(f"Action: {action}, Probs: {dist.probs.cpu().numpy().flatten()}")

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            z, _, _, h = world_model.rssm.posterior(obs_t, h)

            env.render()
            if done:
                print("Episode done, total_reward:", total_reward)
                total_reward = 0.0
                obs, _ = env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                h = world_model.rssm.init_hidden(1, DEVICE)
                z, _, _, h = world_model.rssm.posterior(obs_t, h)

if __name__ == "__main__":
    main()
