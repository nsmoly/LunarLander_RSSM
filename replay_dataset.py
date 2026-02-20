import argparse
import time
import numpy as np
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar_lander_module
import pygame

# IMPORTANT! We set these here and everywhere to these values to have a consistent world
lunar_lander_module.VIEWPORT_W = 600
lunar_lander_module.VIEWPORT_H = 400
lunar_lander_module.SCALE = 30
lunar_lander_module.FPS = 50  

def main():
    parser = argparse.ArgumentParser(description="Replay dataset episodes in LunarLander")
    parser.add_argument("--dataset", default="lunarlander_dataset.npz",
                        help="Path to dataset .npz")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to replay")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for selecting episodes")
    args = parser.parse_args()

    data = np.load(args.dataset)
    ep_index = data["ep_index"].astype(np.int64)
    step_index = data["step_index"].astype(np.int64)
    actions = data["actions"]
    episode_seed = data["episode_seed"].astype(np.int64) if "episode_seed" in data else None

    if ep_index.size == 0:
        print("No episodes found in dataset.")
        return

    rng = np.random.default_rng(args.seed)
    episode_ids = np.unique(ep_index)
    num = min(args.episodes, len(episode_ids))
    picks = rng.choice(len(episode_ids), size=num, replace=False)

    env = gym.make("LunarLander-v3", render_mode="human")

    pygame.init()
    pygame.display.set_caption("LunarLander Dataset Replay")
    clock = pygame.time.Clock()

    for i, pick_idx in enumerate(picks, start=1):
        ep_id = episode_ids[pick_idx]
        episode_mask = (ep_index == ep_id)
        order = np.argsort(step_index[episode_mask], kind="stable")
        ep_actions = actions[episode_mask][order]
        if episode_seed is not None:
            ep_seed = int(episode_seed[episode_mask][0])
            obs, _ = env.reset(seed=ep_seed)
        else:
            obs, _ = env.reset(seed=args.seed + i)
        total_reward = 0.0

        for a in ep_actions:
            obs, reward, terminated, truncated, _ = env.step(int(a))
            total_reward += reward
            
            env.render()
            clock.tick(lunar_lander_module.FPS)

            if terminated or truncated:
                break

        print(f"Episode {i}/{num} (dataset ep {int(ep_id)}) length={len(ep_actions)} reward={total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
