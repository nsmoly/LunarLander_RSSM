# ===========================================================================
# MoonLander RSSM — Dreamer-style World Models for LunarLander Policy Training
#
# Copyright (c) 2026 Nikolai Smolyanskiy
# Licensed under the MIT License. See LICENSE file for details.
# ===========================================================================

import argparse
import os
import numpy as np
import torch
import yaml
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar_lander_module
import pygame
from pygame.locals import K_ESCAPE, QUIT, KEYDOWN

from models import WorldModel, Actor, ActorObs

lunar_lander_module.VIEWPORT_W = 600
lunar_lander_module.VIEWPORT_H = 400
lunar_lander_module.SCALE = 30
lunar_lander_module.FPS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    if not os.path.exists(config_path):
        return 16, 256, 256, 1, 4
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    wm = config.get("world_model", {})
    cap = wm.get("capacity", {})
    latent_dim = int(cap.get("latent_dim", 16))
    hidden_dim = int(cap.get("hidden_dim", 256))
    mlp_hidden_dim = int(cap.get("mlp_hidden_dim", hidden_dim))
    gru_num_layers = int(cap.get("gru_num_layers", 1))
    action_dim = int(wm.get("action_dim", 4))
    return latent_dim, hidden_dim, mlp_hidden_dim, gru_num_layers, action_dim


def main():
    parser = argparse.ArgumentParser(description="Test trained policy in LunarLander")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--world_model", default=None,
                        help="World model checkpoint (default: world_model.pt)")
    parser.add_argument("--actor", default=None,
                        help="Actor checkpoint (default: actor.pt for latent, actor_mf.pt for obs)")
    parser.add_argument("--actor_type", choices=["latent", "obs"], default="latent",
                        help="Actor type: 'latent' uses RSSM (h,z), 'obs' uses raw observations")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=600, help="Max steps per episode")
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

    latent_dim, hidden_dim, mlp_hidden_dim, gru_num_layers, action_dim = load_config(args.config)
    use_wm_actor = (args.actor_type == "latent")

    env = gym.make("LunarLander-v3", render_mode="human")
    
    pygame.init()
    pygame.display.set_caption("LunarLander Policy Test")
    clock = pygame.time.Clock()

    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]

    world_model = None
    if use_wm_actor:
        world_model = WorldModel(
            obs_dim,
            action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            gru_num_layers=gru_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        ).to(DEVICE)
        wm_path = args.world_model or "world_model.pt"
        if not os.path.exists(wm_path):
            print(f"Warning: {wm_path} not found!")
            return
        world_model.load_state_dict(torch.load(wm_path, map_location=DEVICE))
        print(f"Loaded world model from {wm_path}")
        world_model.eval()

        actor = Actor(latent_dim=latent_dim, rssm_hidden_dim=hidden_dim,
                      action_dim=action_dim, actor_hidden_dim=mlp_hidden_dim).to(DEVICE)
    else:
        actor = ActorObs(obs_dim, action_dim, hidden_dim=mlp_hidden_dim).to(DEVICE)

    actor_path = args.actor or ("actor.pt" if use_wm_actor else "actor_mf.pt")
    if not os.path.exists(actor_path):
        print(f"Warning: {actor_path} not found!")
        return
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    print(f"Loaded actor ({args.actor_type}) from {actor_path}")

    actor.eval()
    mode_str = "deterministic (argmax)" if args.deterministic else "stochastic (sample)"
    print(f"Action selection mode: {mode_str}")
    print(f"Actor type: {args.actor_type}")
    print(f"Seed: {args.seed}, Episodes: {args.episodes}")

    # ---- Run-level accumulators ----
    run_episode_count = 0
    run_return_sum = 0.0
    run_steps_sum = 0
    run_action_counts = np.zeros(action_dim, dtype=np.int64)
    run_action_counts_air = np.zeros(action_dim, dtype=np.int64)
    run_air_steps = 0
    run_probs_sum = np.zeros(action_dim, dtype=np.float64)
    run_entropy_sum = 0.0
    run_touchdown_abs_ang_vel = []

    with torch.no_grad():
        episode_idx = 0

        h, z = None, None
        if use_wm_actor:
            def init_rssm_state(obs_np):
                h = world_model.rssm.init_hidden(1, DEVICE)
                z_prev = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
                a_prev = torch.zeros(1, action_dim, device=DEVICE)
                h = world_model.rssm.update_hidden(h, z_prev, a_prev)
                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                mean_post, logstd_post = world_model.rssm.posterior(h, obs_t)
                z = mean_post if args.deterministic else world_model.rssm.sample_latent(mean_post, logstd_post)
                return h, z
            h, z = init_rssm_state(obs)

        # ---- Episode-level accumulators ----
        total_reward = 0.0
        episode_steps = 0
        action_counts = np.zeros(action_dim, dtype=np.int64)
        action_counts_air = np.zeros(action_dim, dtype=np.int64)
        air_steps = 0
        probs_sum = np.zeros(action_dim, dtype=np.float64)
        entropy_sum = 0.0
        touchdown_snapshot = None

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False

            landed = (len(obs) >= 8 and obs[6] >= 0.5 and obs[7] >= 0.5
                      and abs(obs[2]) < 0.05 and abs(obs[3]) < 0.05)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            if use_wm_actor:
                action_dist = actor(h, z)
            else:
                action_dist = actor(obs_t)
            probs = action_dist.probs.squeeze(0).detach().cpu().numpy()
            if landed:
                action = 0
            elif args.deterministic:
                action = int(np.argmax(probs))
            else:
                action = action_dist.sample().item()

            action_counts[action] += 1
            probs_sum += probs
            entropy_sum += float(action_dist.entropy().item())
            if obs[6] < 0.5 and obs[7] < 0.5:
                action_counts_air[action] += 1
                air_steps += 1
            episode_steps += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated or episode_steps >= args.max_steps
            total_reward += reward

            if done and len(next_obs) >= 6:
                touchdown_snapshot = (
                    float(next_obs[0]), float(next_obs[1]),
                    float(next_obs[2]), float(next_obs[3]),
                    float(next_obs[4]), float(next_obs[5]),
                )

            obs = next_obs

            if use_wm_actor:
                a_onehot = torch.nn.functional.one_hot(
                    torch.tensor([action], device=DEVICE), num_classes=action_dim
                ).float()
                h = world_model.rssm.update_hidden(h, z, a_onehot)
                obs_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                mean_post, logstd_post = world_model.rssm.posterior(h, obs_t)
                z = mean_post if args.deterministic else world_model.rssm.sample_latent(mean_post, logstd_post)

            env.render()
            clock.tick(lunar_lander_module.FPS)

            if done:
                ep_num = episode_idx + 1
                action_mix = " ".join(
                    f"{a}:{100.0 * action_counts[a] / max(1, episode_steps):.1f}%"
                    for a in range(action_dim)
                )
                action_mix_air = " ".join(
                    f"{a}:{100.0 * action_counts_air[a] / max(1, air_steps):.1f}%"
                    for a in range(action_dim)
                ) if air_steps > 0 else "n/a"
                avg_probs = probs_sum / max(1, episode_steps)
                avg_probs_str = " ".join(f"{a}:{100.0 * avg_probs[a]:.1f}%" for a in range(action_dim))
                entropy_avg = entropy_sum / max(1, episode_steps)

                print(f"[Episode {ep_num}] return={total_reward:+.2f}, steps={episode_steps}")
                print(f"[Episode {ep_num}] action_mix {action_mix}")
                print(f"[Episode {ep_num}] action_mix_air {action_mix_air}")
                print(f"[Episode {ep_num}] avg_policy_probs {avg_probs_str}")
                print(f"[Episode {ep_num}] avg_entropy={entropy_avg:.3f}")
                if touchdown_snapshot is not None:
                    tx, ty, tvx, tvy, ta, tav = touchdown_snapshot
                    print(
                        f"[Episode {ep_num}] touchdown x={tx:+.3f} y={ty:+.3f} "
                        f"vx={tvx:+.3f} vy={tvy:+.3f} angle={ta:+.3f} ang_vel={tav:+.3f}"
                    )

                # Accumulate into run-level stats
                run_episode_count += 1
                run_return_sum += total_reward
                run_steps_sum += episode_steps
                run_action_counts += action_counts
                run_action_counts_air += action_counts_air
                run_air_steps += air_steps
                run_probs_sum += probs_sum
                run_entropy_sum += entropy_sum
                if touchdown_snapshot is not None:
                    run_touchdown_abs_ang_vel.append(abs(touchdown_snapshot[5]))

                if run_episode_count >= args.episodes:
                    running = False
                    continue

                # Reset for next episode
                total_reward = 0.0
                episode_idx += 1
                episode_steps = 0
                action_counts.fill(0)
                action_counts_air.fill(0)
                air_steps = 0
                probs_sum.fill(0.0)
                entropy_sum = 0.0
                touchdown_snapshot = None
                obs, _ = env.reset(seed=args.seed + episode_idx)
                if use_wm_actor:
                    h, z = init_rssm_state(obs)

    # ---- Run Summary (after all episodes) ----
    if run_episode_count > 0:
        avg_return = run_return_sum / run_episode_count
        avg_steps = run_steps_sum / run_episode_count
        r_action_mix = " ".join(
            f"{a}:{100.0 * run_action_counts[a] / max(1, run_steps_sum):.1f}%"
            for a in range(action_dim)
        )
        r_action_mix_air = " ".join(
            f"{a}:{100.0 * run_action_counts_air[a] / max(1, run_air_steps):.1f}%"
            for a in range(action_dim)
        ) if run_air_steps > 0 else "n/a"
        r_avg_probs = run_probs_sum / max(1, run_steps_sum)
        r_avg_probs_str = " ".join(f"{a}:{100.0 * r_avg_probs[a]:.1f}%" for a in range(action_dim))
        r_entropy_avg = run_entropy_sum / max(1, run_steps_sum)

        print(f"[Run Summary] episodes={run_episode_count}, seed={args.seed}")
        print(f"[Run Summary] avg_return={avg_return:+.2f}, avg_steps={avg_steps:.1f}")
        print(f"[Run Summary] action_mix {r_action_mix}")
        print(f"[Run Summary] action_mix_air {r_action_mix_air}")
        print(f"[Run Summary] avg_policy_probs {r_avg_probs_str}")
        print(f"[Run Summary] avg_entropy={r_entropy_avg:.3f}")
        if run_touchdown_abs_ang_vel:
            td_med = float(np.median(run_touchdown_abs_ang_vel))
            print(f"[Run Summary] touchdown median_abs_ang_vel={td_med:.3f}")


if __name__ == "__main__":
    main()
