# ===========================================================================
# MoonLander RSSM — Dreamer-style World Models for LunarLander Policy Training
#
# Copyright (c) 2026 Nikolai Smolyanskiy
# Licensed under the MIT License. See LICENSE file for details.
#
# Tests a trained actor (world-model-based or model-free) on LunarLander-v3.
# Mirrors wm_mpc_policy.py: headless by default, --render to enable, 'R' to
# toggle render live, 'Q'/ESC to quit. Prints per-episode lines and a final
# Scorecard line plus an AC-specific counters line for grep-friendly logs.
# ===========================================================================

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

import pygame
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar_lander_module

from models import WorldModel, Actor, ActorObs

# IMPORTANT! Match the (modified) constants used during dataset collection
lunar_lander_module.VIEWPORT_W = 600
lunar_lander_module.VIEWPORT_H = 400
lunar_lander_module.SCALE = 30
lunar_lander_module.FPS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    if not os.path.exists(config_path):
        return 16, 256, 256, 1, 4
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    wm_cfg = config.get("world_model", {})
    cap = wm_cfg.get("capacity", {})
    latent_dim = int(cap.get("latent_dim", 16))
    hidden_dim = int(cap.get("hidden_dim", 256))
    mlp_hidden_dim = int(cap.get("mlp_hidden_dim", hidden_dim))
    gru_num_layers = int(cap.get("gru_num_layers", 1))
    action_dim = int(wm_cfg.get("action_dim", 4))
    return latent_dim, hidden_dim, mlp_hidden_dim, gru_num_layers, action_dim


def load_world_model(config_path, checkpoint_path, obs_dim):
    latent_dim, hidden_dim, mlp_hidden_dim, gru_num_layers, action_dim = load_config(config_path)
    world_model = WorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"World model checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    world_model.load_state_dict(state_dict)
    world_model.eval()
    return world_model, action_dim, latent_dim, hidden_dim, mlp_hidden_dim


def load_actor(actor_type, checkpoint_path, *, obs_dim=None, action_dim=4,
               latent_dim=16, hidden_dim=256, mlp_hidden_dim=256):
    if actor_type == "latent":
        actor = Actor(latent_dim=latent_dim, rssm_hidden_dim=hidden_dim,
                      action_dim=action_dim, actor_hidden_dim=mlp_hidden_dim).to(DEVICE)
    elif actor_type == "obs":
        if obs_dim is None:
            raise ValueError("obs_dim required for actor_type='obs'")
        actor = ActorObs(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=mlp_hidden_dim).to(DEVICE)
    else:
        raise ValueError(f"unknown actor_type: {actor_type}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Actor checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def handle_events(render_enabled):
    if pygame is None:
        return False, render_enabled
    quit_requested = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_requested = True
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                quit_requested = True
            elif event.key == pygame.K_r:
                render_enabled = not render_enabled
                state = "ON" if render_enabled else "OFF"
                print(f"[UI] Render toggled {state}")
    return quit_requested, render_enabled


def draw_frame(screen, font, frame, lines, render_enabled):
    if pygame is None or screen is None:
        return
    if render_enabled and frame is not None:
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen_w, screen_h = screen.get_size()
        target_w, target_h = screen_w, screen_h
        surface = pygame.transform.smoothscale(surface, (target_w, target_h))
        screen.fill((0, 0, 0))
        x_off = (screen_w - target_w) // 2
        y_off = (screen_h - target_h) // 2
        screen.blit(surface, (x_off, y_off))
    else:
        screen.fill((0, 0, 0))

    overlay = list(lines)
    if not render_enabled:
        overlay.append("Render paused. Press R to enable.")
    y = 8
    for line in overlay:
        txt = font.render(line, True, (255, 255, 255))
        screen.blit(txt, (8, y))
        y += 22
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Test trained actor (WM-based or model-free) on LunarLander")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--actor_type", choices=["latent", "obs"], required=True,
                        help="'latent': WM-based AC (needs --world_model); 'obs': model-free AC")
    parser.add_argument("--actor", required=True, help="Actor checkpoint path")
    parser.add_argument("--world_model", default=None,
                        help="World model checkpoint (required for --actor_type latent)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Enable render window (default: OFF)")
    parser.add_argument("--render_fps", type=float, default=30.0,
                        help="Target frames-per-second cap for the render window (default: 30). "
                             "Lower = slower / easier to watch. Headless runs ignore this.")
    action_mode = parser.add_mutually_exclusive_group()
    action_mode.add_argument("--deterministic", dest="deterministic", action="store_true",
                             help="Argmax action selection (default).")
    action_mode.add_argument("--stochastic", dest="deterministic", action="store_false",
                             help="Sample actions from the policy distribution.")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--y_near", type=float, default=0.35, help="Near-ground threshold (for scorecard)")
    args = parser.parse_args()

    use_wm = (args.actor_type == "latent")
    if use_wm and not args.world_model:
        parser.error("--world_model is required when --actor_type=latent")

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]

    world_model = None
    if use_wm:
        world_model, action_dim, latent_dim, hidden_dim, mlp_hidden_dim = load_world_model(
            args.config, args.world_model, obs_dim
        )
        print(f"Loaded world model from {args.world_model}")
    else:
        latent_dim, hidden_dim, mlp_hidden_dim, _, action_dim = load_config(args.config)

    actor = load_actor(
        args.actor_type, args.actor,
        obs_dim=obs_dim, action_dim=action_dim,
        latent_dim=latent_dim, hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim,
    )
    print(f"Loaded actor ({args.actor_type}) from {args.actor}")

    # ---- Pygame init (window stays open even if render is off; needed for 'R' toggle) ----
    actor_label = "WM-based" if use_wm else "model-free"
    mode_label = "deterministic" if args.deterministic else "stochastic"
    title = f"LunarLander Policy Test — actor={actor_label}, mode={mode_label}"

    pygame.init()
    pygame.display.set_caption(title)
    screen = pygame.display.set_mode((lunar_lander_module.VIEWPORT_W, lunar_lander_module.VIEWPORT_H))
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()
    render_enabled = bool(args.render and screen is not None)

    # Seed AFTER all init so policy sampling is deterministic regardless of
    # how many random numbers gym/model-init consumed above.
    set_seed(args.seed)

    print(
        "Controls: R toggle render, Q or ESC quit. "
        "Actions: 0=idle, 1=left side, 2=main, 3=right side\n"
        f"actor_type={args.actor_type}, mode={mode_label}, episodes={args.episodes}, seed={args.seed}"
    )

    # ---- Run-level accumulators ----
    run_returns = []
    run_steps = []
    run_action_counts = np.zeros(action_dim, dtype=np.int64)
    run_action_counts_air = np.zeros(action_dim, dtype=np.int64)
    run_step_total = 0
    run_air_steps = 0
    run_near_count = 0
    run_near_abs_angle_sum = 0.0
    run_near_down_speed_sum = 0.0
    run_near_abs_vx_sum = 0.0
    run_touchdown_abs_ang_vel = []
    run_entropy_sum = 0.0

    stop_all = False
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        set_seed(args.seed + ep)

        done = False
        ep_return = 0.0
        prev_action = 0
        action_counts = np.zeros(action_dim, dtype=np.int64)
        action_counts_air = np.zeros(action_dim, dtype=np.int64)
        step_count = 0
        air_steps = 0
        entropy_sum = 0.0

        min_y = float("inf")
        max_abs_angle = 0.0
        max_down_speed = 0.0
        max_abs_vx = 0.0

        near_count = 0
        near_abs_angle_sum = 0.0
        near_down_speed_sum = 0.0
        near_abs_vx_sum = 0.0

        touchdown_snapshot = None

        # Initialize RSSM state (latent actor only)
        if use_wm:
            h = world_model.rssm.init_hidden(batch_size=1, device=DEVICE)
            z = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
        else:
            h = z = None

        t0 = time.perf_counter()

        for step in range(1, args.max_steps + 1):
            step_start = time.perf_counter()
            if pygame is not None and screen is not None:
                quit_requested, render_enabled = handle_events(render_enabled)
                if quit_requested:
                    done = True
                    stop_all = True
                    break

            landed = (len(obs) >= 8 and obs[6] >= 0.5 and obs[7] >= 0.5
                      and abs(obs[2]) < 0.05 and abs(obs[3]) < 0.05)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                if use_wm:
                    prev_action_oh = F.one_hot(
                        torch.tensor([prev_action], device=DEVICE), num_classes=action_dim
                    ).float()
                    h, z, _, _, _, _ = world_model.rssm.step(h, z, prev_action_oh, obs_t)
                    action_dist = actor(h, z)
                else:
                    action_dist = actor(obs_t)

                probs = action_dist.probs.squeeze(0)
                entropy = float(action_dist.entropy().item())
                if landed:
                    action = 0
                elif args.deterministic:
                    action = int(torch.argmax(probs).item())
                else:
                    action = int(action_dist.sample().item())
                chosen_prob = float(probs[action].item())
                probs_np = probs.detach().cpu().numpy()

            step_count += 1
            entropy_sum += entropy
            if 0 <= int(action) < action_dim:
                action_counts[int(action)] += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)

            if len(next_obs) >= 6:
                x = float(next_obs[0])
                y = float(next_obs[1])
                vx = float(next_obs[2])
                vy = float(next_obs[3])
                angle = float(next_obs[4])
                ang_vel = float(next_obs[5])

                min_y = min(min_y, y)
                max_abs_angle = max(max_abs_angle, abs(angle))
                max_down_speed = max(max_down_speed, max(-vy, 0.0))
                max_abs_vx = max(max_abs_vx, abs(vx))

                if y < args.y_near:
                    near_count += 1
                    near_abs_angle_sum += abs(angle)
                    near_down_speed_sum += max(-vy, 0.0)
                    near_abs_vx_sum += abs(vx)

                # Count action mix in the air (exclude on-ground actions)
                if abs(vy) > 0.02 and (y > 0.0) and 0 <= int(action) < action_dim:
                    action_counts_air[int(action)] += 1
                    air_steps += 1

                if done:
                    touchdown_snapshot = (x, y, vx, vy, angle, ang_vel)

            obs = next_obs
            prev_action = action

            elapsed = time.perf_counter() - t0
            fps_est = step / max(elapsed, 1e-6)
            frame = env.render() if render_enabled else None
            probs_str = " ".join(f"{a}={probs_np[a]:.2f}" for a in range(action_dim))
            overlay = [
                f"Episode {ep}/{args.episodes} Step {step}",
                f"Real reward {reward:+.3f}  Episode return {ep_return:+.2f}",
                f"Action {action} (prob {chosen_prob:.3f})  Entropy {entropy:.3f}",
                f"Action probs: {probs_str}",
                f"FPS cap {args.render_fps:.0f}  Measured {fps_est:.1f}",
            ]
            draw_frame(screen, font, frame, overlay, render_enabled)
            # Hard throttle via time.sleep when rendering, so the lander descent
            # is watchable; headless sweeps run as fast as possible.
            if render_enabled and args.render_fps > 0.0:
                target_dt = 1.0 / args.render_fps
                step_elapsed = time.perf_counter() - step_start
                remaining = target_dt - step_elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
                clock.tick()  # keeps pygame's internal clock fresh (no extra throttle)

            if done:
                break

        # ---- Per-episode summary ----
        avg_entropy = entropy_sum / max(1, step_count)
        print(
            f"[Episode {ep}] return={ep_return:+.2f}, steps={step_count}, "
            f"done={done}, elapsed={time.perf_counter() - t0:.1f}s"
        )
        if step_count > 0:
            action_mix = " ".join(
                f"{a}:{(100.0 * action_counts[a] / step_count):.1f}%"
                for a in range(action_dim)
            )
            print(f"[Episode {ep}] action_mix {action_mix}")
            if air_steps > 0:
                action_mix_air = " ".join(
                    f"{a}:{(100.0 * action_counts_air[a] / air_steps):.1f}%"
                    for a in range(action_dim)
                )
                print(f"[Episode {ep}] action_mix_air {action_mix_air}")
            print(f"[Episode {ep}] avg_entropy={avg_entropy:.3f}")
        if min_y != float("inf"):
            print(
                f"[Episode {ep}] extrema min_y={min_y:+.3f} "
                f"max_abs_angle={max_abs_angle:.3f} "
                f"max_down_speed={max_down_speed:.3f} "
                f"max_abs_vx={max_abs_vx:.3f}"
            )
        if near_count > 0:
            print(
                f"[Episode {ep}] near_ground(y<{args.y_near:.2f}) "
                f"avg_abs_angle={near_abs_angle_sum / near_count:.3f} "
                f"avg_down_speed={near_down_speed_sum / near_count:.3f} "
                f"avg_abs_vx={near_abs_vx_sum / near_count:.3f}"
            )
        if touchdown_snapshot is not None:
            tx, ty, tvx, tvy, tangle, tang_vel = touchdown_snapshot
            print(
                f"[Episode {ep}] touchdown "
                f"x={tx:+.3f} y={ty:+.3f} vx={tvx:+.3f} vy={tvy:+.3f} "
                f"angle={tangle:+.3f} ang_vel={tang_vel:+.3f}"
            )

        # Accumulate run-level
        run_returns.append(ep_return)
        run_steps.append(step_count)
        run_action_counts += action_counts
        run_action_counts_air += action_counts_air
        run_step_total += step_count
        run_air_steps += air_steps
        run_near_count += near_count
        run_near_abs_angle_sum += near_abs_angle_sum
        run_near_down_speed_sum += near_down_speed_sum
        run_near_abs_vx_sum += near_abs_vx_sum
        run_entropy_sum += entropy_sum
        if touchdown_snapshot is not None:
            run_touchdown_abs_ang_vel.append(abs(touchdown_snapshot[5]))

        if stop_all:
            break

    # ---- Run Scorecard (matches mpc_eval_logs.txt format for grep parity) ----
    if run_returns:
        mean_return = float(np.mean(run_returns))
        worst_return = float(np.min(run_returns))
        main_mix = (100.0 * run_action_counts[2] / run_step_total) if run_step_total > 0 else 0.0
        main_mix_air = (100.0 * run_action_counts_air[2] / run_air_steps) if run_air_steps > 0 else 0.0
        near_avg_abs_angle = (run_near_abs_angle_sum / run_near_count) if run_near_count > 0 else float("nan")
        near_avg_down_speed = (run_near_down_speed_sum / run_near_count) if run_near_count > 0 else float("nan")
        touchdown_median_abs_ang_vel = (
            float(np.median(np.array(run_touchdown_abs_ang_vel, dtype=np.float64)))
            if run_touchdown_abs_ang_vel else float("nan")
        )
        touchdown_p90_abs_ang_vel = (
            float(np.percentile(np.array(run_touchdown_abs_ang_vel, dtype=np.float64), 90))
            if run_touchdown_abs_ang_vel else float("nan")
        )

        # AC-specific counters
        perfect = int(sum(1 for r in run_returns if r > 200.0))
        negative = int(sum(1 for r in run_returns if r < 0.0))
        catastrophic = int(sum(1 for r in run_returns if r < -100.0))
        avg_entropy = run_entropy_sum / max(1, run_step_total)
        avg_steps = float(np.mean(run_steps)) if run_steps else 0.0

        print("[Run] ===== AC Scorecard =====")
        print(
            f"[Run] mean_return={mean_return:+.2f} worst_return={worst_return:+.2f} "
            f"main_action_mix={main_mix:.1f}% main_action_mix_air={main_mix_air:.1f}% "
            f"near_avg_abs_angle={near_avg_abs_angle:.3f} "
            f"near_avg_down_speed={near_avg_down_speed:.3f} "
            f"touchdown_median_abs_ang_vel={touchdown_median_abs_ang_vel:.3f} "
            f"touchdown_p90_abs_ang_vel={touchdown_p90_abs_ang_vel:.3f}"
        )
        print(
            f"[Run] perfect={perfect} negative={negative} catastrophic={catastrophic} "
            f"avg_entropy={avg_entropy:.3f} avg_steps={avg_steps:.1f}"
        )

    env.close()
    if pygame is not None and screen is not None:
        pygame.quit()


if __name__ == "__main__":
    main()
