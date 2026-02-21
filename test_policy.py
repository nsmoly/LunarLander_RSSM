# test_policy.py
import argparse
import os
import torch
import yaml
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar_lander_module
import pygame
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, K_ESCAPE, QUIT, KEYDOWN

from models import WorldModel, Actor

# IMPORTANT! We set these here and everywhere to these values to have a consistent world
lunar_lander_module.VIEWPORT_W = 600
lunar_lander_module.VIEWPORT_H = 400
lunar_lander_module.SCALE = 30
lunar_lander_module.FPS = 50  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    if not os.path.exists(config_path):
        return 64, 128, 1, 4
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    wm = config.get("world_model", {})
    cap = wm.get("capacity", {})
    latent_dim = int(cap.get("latent_dim", 64))
    hidden_dim = int(cap.get("hidden_dim", 128))
    gru_num_layers = int(cap.get("gru_num_layers", 1))
    action_dim = int(wm.get("action_dim", 4))
    return latent_dim, hidden_dim, gru_num_layers, action_dim


def main():
    parser = argparse.ArgumentParser(description="Test trained policy in LunarLander")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--world_model", default=None,
                        help="World model checkpoint (default: world_model.pt)")
    parser.add_argument("--actor", default=None,
                        help="Actor checkpoint (default: actor.pt)")
    action_mode = parser.add_mutually_exclusive_group()
    action_mode.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Use argmax action selection (default).",
    )
    action_mode.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample actions from the policy distribution.",
    )
    parser.set_defaults(deterministic=True)
    args = parser.parse_args()

    latent_dim, hidden_dim, gru_num_layers, action_dim = load_config(args.config)

    env = gym.make("LunarLander-v3", render_mode="human")
    
    pygame.init()
    pygame.display.set_caption("LunarLander Policy Test")
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]

    world_model = WorldModel(
        obs_dim,
        action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers,
    ).to(DEVICE)
    actor = Actor(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(DEVICE)

    wm_path = args.world_model or "world_model.pt"
    if wm_path and os.path.exists(wm_path):
        world_model.load_state_dict(torch.load(wm_path, map_location=DEVICE))
        print(f"Loaded world model from {wm_path}")
    else:
        print(f"Warning: {wm_path} not found!")
        return

    actor_path = args.actor or "actor.pt"
    if actor_path and os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
        print(f"Loaded actor from {actor_path}")
    else:
        print(f"Warning: {actor_path} not found!")
        return

    world_model.eval()
    actor.eval()
    print(
        f"Action selection mode: {'deterministic (argmax)' if args.deterministic else 'stochastic (sample)'}"
    )

    done = False
    total_reward = 0.0

    with torch.no_grad():
        action_dim = world_model.rssm.action_dim
        h = world_model.rssm.init_hidden(1, DEVICE)
        z_prev = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
        a_prev = torch.zeros(1, action_dim, device=DEVICE)
        h = world_model.rssm.update_hidden(h, z_prev, a_prev)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mean_post, logstd_post = world_model.rssm.posterior(h, obs_t)
        z = world_model.rssm.sample_latent(mean_post, logstd_post)

        running = True
        while running:
            # Handle keyboard events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False

            # Get action from the policy
            action_dist = actor(z)
            if args.deterministic:
                action = torch.argmax(action_dist.probs, dim=-1).item()
            else:
                action = action_dist.sample().item()
            print(f"Action: {action}, Probs: {action_dist.probs.cpu().numpy().flatten()}")

            # Execute this action in the environment and get the next observation, reward, and done flag
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Update world model
            a_onehot = torch.nn.functional.one_hot(
                torch.tensor([action], device=DEVICE), num_classes=action_dim
            ).float()
            h = world_model.rssm.update_hidden(h, z, a_onehot)
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean_post, logstd_post = world_model.rssm.posterior(h, obs_t)
            z = world_model.rssm.sample_latent(mean_post, logstd_post)

            env.render()
            clock.tick(lunar_lander_module.FPS)

            if done:
                print("Episode done, total_reward:", total_reward)
                # Reset the world model and actor states
                total_reward = 0.0
                obs, _ = env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                h = world_model.rssm.init_hidden(1, DEVICE)
                z_prev = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
                a_prev = torch.zeros(1, action_dim, device=DEVICE)
                h = world_model.rssm.update_hidden(h, z_prev, a_prev)
                mean_post, logstd_post = world_model.rssm.posterior(h, obs_t)
                z = world_model.rssm.sample_latent(mean_post, logstd_post)

if __name__ == "__main__":
    main()
