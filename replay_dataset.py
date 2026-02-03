import argparse
import time
import numpy as np
import gym


def split_episodes(actions, dones):
    end_idxs = np.where(dones == 1)[0]
    episodes = []
    start = 0
    for end in end_idxs:
        episodes.append((start, end + 1))
        start = end + 1
    if start < len(actions):
        episodes.append((start, len(actions)))
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Replay dataset episodes in LunarLander")
    parser.add_argument("--dataset", default="lunarlander_train_dataset.npz",
                        help="Path to dataset .npz")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to replay")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for selecting episodes")
    parser.add_argument("--sleep", type=float, default=0.02,
                        help="Sleep between steps (seconds)")
    args = parser.parse_args()

    data = np.load(args.dataset)
    actions = data["actions"].astype(np.int64)
    dones = data["dones"].astype(np.int64)

    episodes = split_episodes(actions, dones)
    if not episodes:
        print("No episodes found in dataset.")
        return

    rng = np.random.default_rng(args.seed)
    num = min(args.episodes, len(episodes))
    picks = rng.choice(len(episodes), size=num, replace=False)

    env = gym.make("LunarLander-v2", render_mode="human")

    for i, ep_idx in enumerate(picks, start=1):
        start, end = episodes[ep_idx]
        ep_actions = actions[start:end]
        obs, _ = env.reset(seed=args.seed + i)
        total_reward = 0.0

        for a in ep_actions:
            obs, reward, terminated, truncated, _ = env.step(int(a))
            total_reward += reward
            time.sleep(args.sleep)
            if terminated or truncated:
                break

        print(f"Episode {i}/{num} (dataset idx {ep_idx}) length={len(ep_actions)} reward={total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
