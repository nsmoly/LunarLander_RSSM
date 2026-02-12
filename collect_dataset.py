# collect_dataset.py
import argparse
import gymnasium as gym
import numpy as np
import pygame
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, K_ESCAPE, QUIT, KEYDOWN

# Dataset format (per-step rows):
# [ep_index, step_index, obs_t, action_t, reward_t, next_obs_t, done_t, episode_seed]
# ep_index is the index of the episode.
# step_index is the index of the step within the episode.
# obs_t is the observation at the current step.
# action_t is the action taken at the current step. 0-3 are the actions: left, right, up, down.
# reward_t is the reward received at the current step.
# next_obs_t is the observation at the next step.
# done_t is a boolean indicating if the episode is done.
# episode_seed is the seed used to reset the environment to the same initial state for each episode.

DATASET_PATH = "lunarlander_dataset.npz"

def main():
    parser = argparse.ArgumentParser(description="Collect LunarLander dataset with episode seeds")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Output dataset .npz path")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for episode surfaces")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = gym.make("LunarLander-v3", render_mode="human")
    env.unwrapped.SCALE = 20  # Zoom out by reducing scale (default 30)
    episode_seed = int(rng.integers(0, 2**31 - 1))
    obs, _ = env.reset(seed=episode_seed)

    pygame.init()
    screen = pygame.display.set_mode((600, 400))  # Match viewport size
    pygame.display.set_caption("LunarLander Keyboard Controller")
    clock = pygame.time.Clock()  # Control frame rate

    current_obs = obs
    done = False
    ep_index = 0
    step_index = 0
    episode_reward = 0.0
    episode_length = 0

    # Flat per-step storage for sequences. All episodes are stored in a single array. 
    ep_indices = []
    step_indices = []
    episode_seeds = []
    obs_list = []
    action_list = []
    reward_list = []
    next_obs_list = []
    done_list = []

    print("Controls: LEFT/RIGHT/UP/DOWN arrows (hold to fire). ESC to end and save.")

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                done = True
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                    done = True

        # Get held keys for continuous control
        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:
            action = 1
        elif keys[K_UP]:
            action = 2
        elif keys[K_RIGHT]:
            action = 3
        else:
            action = 0

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_indices.append(ep_index)
        step_indices.append(step_index)
        episode_seeds.append(episode_seed)
        obs_list.append(np.asarray(current_obs, dtype=np.float32))
        action_list.append(int(action))
        reward_list.append(float(reward))
        next_obs_list.append(np.asarray(next_obs, dtype=np.float32))
        done_list.append(1 if done else 0)

        episode_reward += reward
        episode_length += 1

        current_obs = next_obs
        step_index += 1
        env.render()

        clock.tick(60)  # Limit to 60 FPS for better control

        if done:
            print(f"Episode {ep_index} collected, length={episode_length}, total_reward={episode_reward:.2f}")
            ep_index += 1
            step_index = 0
            episode_reward = 0.0
            episode_length = 0
            episode_seed = int(rng.integers(0, 2**31 - 1))
            current_obs, _ = env.reset(seed=episode_seed)
            done = False

    env.close()
    pygame.quit()

    # Save dataset in per-step format as a set of arrays. All must be the same length.
    lengths = {
        "ep_index": len(ep_indices),
        "step_index": len(step_indices),
        "episode_seed": len(episode_seeds),
        "obs": len(obs_list),
        "actions": len(action_list),
        "rewards": len(reward_list),
        "next_obs": len(next_obs_list),
        "dones": len(done_list),
    }
    if len(set(lengths.values())) != 1:
        print(f"Length mismatch, not saving dataset: {lengths}")
        return

    np.savez(
        args.dataset,
        ep_index=np.asarray(ep_indices, dtype=np.int64),
        step_index=np.asarray(step_indices, dtype=np.int64),
        episode_seed=np.asarray(episode_seeds, dtype=np.int64),
        obs=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(action_list, dtype=np.int64),
        rewards=np.asarray(reward_list, dtype=np.float32),
        next_obs=np.asarray(next_obs_list, dtype=np.float32),
        dones=np.asarray(done_list, dtype=np.int64),
    )
    print(f"Saved dataset to {args.dataset}")

if __name__ == "__main__":
    main()
