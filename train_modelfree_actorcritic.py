# ===========================================================================
# MoonLander RSSM — Model-free RL policy training for LunarLander
#
# Copyright (c) 2026 Nikolai Smolyanskiy
# Licensed under the MIT License. See LICENSE file for details.
# ===========================================================================
#
# Model-free actor-critic RL trainer for LunarLander-v3.
# Collects real environment experience and trains ActorObs / CriticObs
# directly on raw observations — no world model required.
#
# The trained actor is compatible with test_policy.py (use --actor_type obs).
# This trainer is self-contained and uses ActorObs / CriticObs define in models.py
# that operate directly on raw observations.
# All training hyperparameters are set in the main function.

import argparse
import os
import random
import datetime
import glob
import numpy as np
import torch
import torch.optim as optim

import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar_lander_module

from models import ActorObs, CriticObs

# Must match test_policy.py settings exactly
lunar_lander_module.VIEWPORT_W = 600
lunar_lander_module.VIEWPORT_H = 400
lunar_lander_module.SCALE = 30
lunar_lander_module.FPS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_message(message, log_path=None):
    print(message)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def save_checkpoint(model, model_name, epoch, directory=CHECKPOINT_DIR, log_path=None):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"{model_name}_{timestamp}_epoch_{epoch}.pt")
    torch.save(model.state_dict(), filename)
    log_message(f"Saved checkpoint: {filename}", log_path)
    return filename


def get_latest_checkpoint(model_name, directory=CHECKPOINT_DIR):
    pattern = os.path.join(directory, f"{model_name}_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_episodes(env, actor, num_episodes, max_steps=600):
    """Run the current actor in the real environment using raw observations.

    Returns a list of episode dicts with keys:
        observations (T, obs_dim)  — raw environment observations
        actions      (T,)         — actions taken
        rewards      (T,)         — environment rewards
        dones        (T,)         — termination flags
    """
    actor.eval()
    episodes = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()

            ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []

            for _ in range(max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                ep_obs.append(obs_t.squeeze(0).cpu())

                action_dist = actor(obs_t)
                action = action_dist.sample().item()
                ep_actions.append(action)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_rewards.append(reward)
                ep_dones.append(float(done))

                if done:
                    break

                obs = next_obs

            episodes.append({
                "observations": torch.stack(ep_obs),
                "actions": torch.tensor(ep_actions, dtype=torch.long),
                "rewards": torch.tensor(ep_rewards, dtype=torch.float32),
                "dones": torch.tensor(ep_dones, dtype=torch.float32),
            })

    return episodes


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------
def compute_gae(episodes, critic, gamma=0.99, lambda_gae=0.95):
    """Compute Generalized Advantage Estimation from collected episodes."""
    critic.eval()

    all_obs, all_actions, all_advantages, all_returns = [], [], [], []

    with torch.no_grad():
        for ep in episodes:
            obs = ep["observations"].to(DEVICE)
            rewards = ep["rewards"].to(DEVICE)
            dones = ep["dones"].to(DEVICE)
            T = len(rewards)

            values = critic(obs)

            advantages = torch.zeros(T, device=DEVICE)
            gae = 0.0
            for t in reversed(range(T)):
                next_val = values[t + 1].item() if t < T - 1 else 0.0
                delta = rewards[t] + gamma * (1.0 - dones[t]) * next_val - values[t]
                gae = delta + gamma * lambda_gae * (1.0 - dones[t]) * gae
                advantages[t] = gae

            returns = advantages + values

            all_obs.append(obs.cpu())
            all_actions.append(ep["actions"])
            all_advantages.append(advantages.cpu())
            all_returns.append(returns.cpu())

    return {
        "observations": torch.cat(all_obs),
        "actions": torch.cat(all_actions),
        "advantages": torch.cat(all_advantages),
        "returns": torch.cat(all_returns),
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_on_batch(actor, critic, data, opt_actor, opt_critic,
                   entropy_coeff=0.01, critic_loss_weight=0.3,
                   advantage_clip=3.0, actor_grad_clip=10.0, critic_grad_clip=100.0):
    """Single full-batch update on all collected data (fully on-policy)."""
    actor.train()
    critic.train()

    obs = data["observations"].to(DEVICE)
    actions = data["actions"].to(DEVICE)
    advantages = data["advantages"].to(DEVICE)
    returns = data["returns"].to(DEVICE)

    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    adv = torch.clamp(adv, -advantage_clip, advantage_clip)

    action_dist = actor(obs)
    log_probs = action_dist.log_prob(actions)
    actor_loss = -(log_probs * adv).mean() - entropy_coeff * action_dist.entropy().mean()

    opt_actor.zero_grad()
    actor_loss.backward()
    ag = torch.nn.utils.clip_grad_norm_(actor.parameters(), actor_grad_clip)
    opt_actor.step()

    value_pred = critic(obs)
    critic_loss = critic_loss_weight * torch.nn.functional.smooth_l1_loss(value_pred, returns)

    opt_critic.zero_grad()
    critic_loss.backward()
    cg = torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_grad_clip)
    opt_critic.step()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": action_dist.entropy().mean().item(),
        "max_action_prob": action_dist.probs.max(dim=-1).values.mean().item(),
        "actor_grad_norm": float(ag),
        "critic_grad_norm": float(cg),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model-free actor-critic RL training for LunarLander"
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Training epochs (each = collect episodes + train)")
    parser.add_argument("--episodes_per_epoch", type=int, default=50,
                        help="Episodes to collect per epoch (~5K transitions at ~100 steps)")
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden layer size for ActorObs/CriticObs")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_gae", type=float, default=0.95)
    parser.add_argument("--entropy_coeff", type=float, default=0.2,
                        help="Entropy bonus start (decays linearly to entropy_coeff_end)")
    parser.add_argument("--entropy_coeff_end", type=float, default=0.01)
    parser.add_argument("--critic_loss_weight", type=float, default=0.3,
                        help="Scaling factor for critic loss (matches WM trainer)")
    parser.add_argument("--advantage_clip", type=float, default=0.9)
    parser.add_argument("--actor_grad_clip", type=float, default=1.8)
    parser.add_argument("--critic_grad_clip", type=float, default=100.0)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--render", action="store_true",
                        help="Render one episode per epoch (slows training)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest actor_mf / critic_mf checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)
    log_path = "train_modelfree_actorcritic_logs.txt"

    # ---- Environment ----
    render_mode = "human" if args.render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ---- Create actor and critic ----
    actor = ActorObs(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(DEVICE)
    critic = CriticObs(obs_dim, hidden_dim=args.hidden_dim).to(DEVICE)

    start_epoch = 0
    if args.resume:
        latest_actor = get_latest_checkpoint("actor_mf")
        latest_critic = get_latest_checkpoint("critic_mf")

        if latest_actor and not latest_critic:
            raise RuntimeError(
                f"Found actor checkpoint ({latest_actor}) but no matching "
                f"critic checkpoint. Cannot resume with mismatched pair."
            )
        if latest_critic and not latest_actor:
            raise RuntimeError(
                f"Found critic checkpoint ({latest_critic}) but no matching "
                f"actor checkpoint. Cannot resume with mismatched pair."
            )

        if latest_actor and latest_critic:
            actor_epoch = int(latest_actor.split("_epoch_")[-1].split(".")[0])
            critic_epoch = int(latest_critic.split("_epoch_")[-1].split(".")[0])
            if actor_epoch != critic_epoch:
                raise RuntimeError(
                    f"Actor checkpoint is epoch {actor_epoch} ({latest_actor}) "
                    f"but critic checkpoint is epoch {critic_epoch} ({latest_critic}). "
                    f"Cannot resume with mismatched epochs."
                )
            actor.load_state_dict(torch.load(latest_actor, map_location=DEVICE))
            critic.load_state_dict(torch.load(latest_critic, map_location=DEVICE))
            start_epoch = actor_epoch
            log_message(f"Resumed actor from {latest_actor}", log_path)
            log_message(f"Resumed critic from {latest_critic}", log_path)

    opt_actor = optim.AdamW(actor.parameters(), lr=args.lr)
    opt_critic = optim.AdamW(critic.parameters(), lr=args.lr)

    log_message(
        f"A2C training (obs-space actor): epochs={args.epochs}, "
        f"episodes_per_epoch={args.episodes_per_epoch}, max_steps={args.max_steps}, "
        f"lr={args.lr}, gamma={args.gamma}, lambda_gae={args.lambda_gae}, "
        f"entropy={args.entropy_coeff:.4f}->{args.entropy_coeff_end:.4f}, "
        f"critic_w={args.critic_loss_weight}, adv_clip={args.advantage_clip}, "
        f"hidden_dim={args.hidden_dim}, seed={args.seed}",
        log_path,
    )

    # ---- Collapse detection state ----
    collapse_entropy_threshold = 0.15
    collapse_max_prob_threshold = 0.98
    collapse_patience = 5
    low_entropy_streak = 0
    high_prob_streak = 0

    # ---- LR reduction guard ----
    low_entropy_lr_threshold = 0.25
    reduced_lr = args.lr * (1/6)
    lr_reduced = False

    # ---- Training loop ----
    best_mean_return = -float("inf")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Linear entropy decay
        if args.epochs <= 1:
            entropy_coeff = args.entropy_coeff_end
        else:
            progress = (epoch - 1) / float(args.epochs - 1)
            entropy_coeff = (
                args.entropy_coeff
                + (args.entropy_coeff_end - args.entropy_coeff) * progress
            )

        # Collect fresh on-policy episodes
        episodes = collect_episodes(
            env, actor,
            num_episodes=args.episodes_per_epoch,
            max_steps=args.max_steps,
        )

        ep_returns = [float(ep["rewards"].sum()) for ep in episodes]
        ep_lengths = [len(ep["rewards"]) for ep in episodes]
        total_transitions = sum(ep_lengths)
        mean_ret = np.mean(ep_returns)
        median_ret = np.median(ep_returns)
        min_ret = np.min(ep_returns)
        max_ret = np.max(ep_returns)
        mean_len = np.mean(ep_lengths)

        if mean_ret > best_mean_return:
            best_mean_return = mean_ret

        # Compute GAE from real rewards
        data = compute_gae(
            episodes, critic, gamma=args.gamma, lambda_gae=args.lambda_gae,
        )

        # Train actor and critic (single full-batch update, fully on-policy)
        metrics = train_on_batch(
            actor, critic, data, opt_actor, opt_critic,
            entropy_coeff=entropy_coeff,
            critic_loss_weight=args.critic_loss_weight,
            advantage_clip=args.advantage_clip,
            actor_grad_clip=args.actor_grad_clip,
            critic_grad_clip=args.critic_grad_clip,
        )

        log_message(
            f"[Epoch {epoch}] transitions={total_transitions}, "
            f"mean_return={mean_ret:+.2f}, median={median_ret:+.2f}, "
            f"min={min_ret:+.2f}, max={max_ret:+.2f}, avg_steps={mean_len:.0f}",
            log_path,
        )
        if metrics:
            log_message(
                f"  actor_loss={metrics['actor_loss']:.4f}, "
                f"critic_loss={metrics['critic_loss']:.4f}, "
                f"entropy={metrics['entropy']:.4f}, "
                f"max_prob={metrics['max_action_prob']:.4f}, "
                f"entropy_coeff={entropy_coeff:.4f}",
                log_path,
            )
            log_message(
                f"  grad_norms: actor={metrics['actor_grad_norm']:.4f}, "
                f"critic={metrics['critic_grad_norm']:.4f}",
                log_path,
            )

        # ---- LR reduction guard ----
        if (
            metrics
            and not lr_reduced
            and metrics["entropy"] < low_entropy_lr_threshold
            and reduced_lr < opt_actor.param_groups[0]["lr"]
        ):
            old_lr = opt_actor.param_groups[0]["lr"]
            for pg in opt_actor.param_groups:
                pg["lr"] = reduced_lr
            lr_reduced = True
            log_message(
                f"  [LR Guard] Reduced actor LR {old_lr:.2e} -> {reduced_lr:.2e} "
                f"(entropy={metrics['entropy']:.4f})",
                log_path,
            )

        # ---- Collapse detection ----
        if metrics:
            low_entropy_streak = (
                low_entropy_streak + 1
                if metrics["entropy"] < collapse_entropy_threshold
                else 0
            )
            high_prob_streak = (
                high_prob_streak + 1
                if metrics["max_action_prob"] > collapse_max_prob_threshold
                else 0
            )
            if (
                low_entropy_streak >= collapse_patience
                or high_prob_streak >= collapse_patience
            ):
                reason = (
                    f"entropy<{collapse_entropy_threshold} "
                    f"for {low_entropy_streak} epochs"
                    if low_entropy_streak >= collapse_patience
                    else f"max_prob>{collapse_max_prob_threshold} "
                    f"for {high_prob_streak} epochs"
                )
                log_message(
                    f"[Epoch {epoch}] Early stop: collapse detected ({reason})",
                    log_path,
                )
                save_checkpoint(actor, "actor_mf", epoch, log_path=log_path)
                save_checkpoint(critic, "critic_mf", epoch, log_path=log_path)
                break

        # ---- Checkpoints ----
        if epoch % args.checkpoint_freq == 0:
            save_checkpoint(actor, "actor_mf", epoch, log_path=log_path)
            save_checkpoint(critic, "critic_mf", epoch, log_path=log_path)

    env.close()
    log_message(
        f"Training complete. Best mean return: {best_mean_return:+.2f}", log_path,
    )


if __name__ == "__main__":
    main()
