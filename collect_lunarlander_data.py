# collect_lunarlander_data.py
import gym
import numpy as np
import pygame
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, K_ESCAPE, QUIT, KEYDOWN

def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    env.unwrapped.SCALE = 20  # Zoom out by reducing scale (default 30)
    obs, _ = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((600, 400))  # Match viewport size
    pygame.display.set_caption("LunarLander Keyboard Controller")
    clock = pygame.time.Clock()  # Control frame rate

    episodes = []
    current_obs = obs
    done = False
    episode = {"obs": [], "actions": [], "rewards": [], "dones": []}

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

        episode["obs"].append(current_obs)
        episode["actions"].append(action)
        episode["rewards"].append(reward)
        episode["dones"].append(done)

        current_obs = next_obs
        env.render()

        clock.tick(60)  # Limit to 60 FPS for better control

        if done:
            episodes.append(episode)
            print(f"Episode collected, length={len(episode['obs'])}, total_reward={sum(episode['rewards'])}")
            episode = {"obs": [], "actions": [], "rewards": [], "dones": []}
            current_obs, _ = env.reset()
            done = False

    env.close()
    pygame.quit()

    # Save dataset
    all_obs = np.concatenate([np.array(ep["obs"]) for ep in episodes], axis=0)
    all_actions = np.concatenate([np.array(ep["actions"]) for ep in episodes], axis=0)
    all_rewards = np.concatenate([np.array(ep["rewards"]) for ep in episodes], axis=0)
    all_dones = np.concatenate([np.array(ep["dones"]) for ep in episodes], axis=0)

    np.savez("lunarlander_dataset.npz",
             obs=all_obs,
             actions=all_actions,
             rewards=all_rewards,
             dones=all_dones)
    print("Saved dataset to lunarlander_dataset.npz")

if __name__ == "__main__":
    main()
