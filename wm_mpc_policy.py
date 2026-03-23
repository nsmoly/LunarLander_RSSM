# ===========================================================================
# MoonLander RSSM — Dreamer-style World Models for LunarLander Policy Training
#
# Copyright (c) 2026 Nikolai Smolyanskiy
# Licensed under the MIT License. See LICENSE file for details.
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

from models import WorldModel

# IMPORTANT! We set these here and everywhere to these values to have a consistent world
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
    return world_model, action_dim


def expand_hidden(hidden, batch_size):
    if hidden.dim() == 2:
        return hidden.expand(batch_size, -1).contiguous()
    return hidden.expand(batch_size, -1, -1).contiguous()


def observation_cost(obs_pred, args):
    if obs_pred.shape[-1] < 6:
        return torch.zeros(obs_pred.shape[0], device=obs_pred.device)
    x = obs_pred[:, 0]
    y = obs_pred[:, 1]
    vx = obs_pred[:, 2]
    vy = obs_pred[:, 3]
    angle = obs_pred[:, 4]
    ang_vel = obs_pred[:, 5]

    # 0 far from ground, 1 near ground.
    near = torch.sigmoid((args.y_near - y) / max(args.y_near_temp, 1e-6))
    # Allow faster descent higher up and require softer descent near touchdown.
    vy_allowed = args.vy_allowed_far * (1.0 - near) + args.vy_allowed_near * near
    down_excess = torch.relu((-vy) - vy_allowed)

    cost = (
        args.w_x * torch.abs(x)
        + args.w_y * torch.abs(y)
        + args.w_vx * torch.abs(vx)
        + args.w_vy_down * down_excess
        + args.w_angle * torch.abs(angle)
        + args.w_angle_near * near * torch.abs(angle)
        + args.w_ang_vel * torch.abs(ang_vel)
        + args.w_ang_vel_near * near * torch.abs(ang_vel)
    )
    return cost


@torch.no_grad()
def evaluate_action_sequences(world_model, h0, z0, action_sequences, action_dim, args):
    batch_size = action_sequences.shape[0]
    horizon = action_sequences.shape[1]
    h = expand_hidden(h0, batch_size)
    z = z0.expand(batch_size, -1).contiguous()

    scores = torch.zeros(batch_size, device=DEVICE)
    discount = 1.0
    for t in range(horizon):
        actions_t = action_sequences[:, t]
        action_oh = F.one_hot(actions_t, num_classes=action_dim).float()
        h = world_model.rssm.update_hidden(h, z, action_oh)
        mean_prior, _ = world_model.rssm.prior(h)
        z = mean_prior
        obs_pred = world_model.reconstruct_obs(h, z)
        reward_pred = world_model.predict_reward(h, z)
        step_score = args.reward_weight * reward_pred
        if args.use_obs_cost:
            step_score = step_score - observation_cost(obs_pred, args)
        if args.done_penalty > 0.0 and hasattr(world_model, "predict_done_logits"):
            done_prob = torch.sigmoid(world_model.predict_done_logits(h, z))
            step_score = step_score - args.done_penalty * done_prob
        scores = scores + discount * step_score
        discount *= args.gamma
    return scores


@torch.no_grad()
def cem_plan(world_model, h0, z0, action_dim, args):
    logits = torch.zeros(args.horizon, action_dim, device=DEVICE)
    best_sequence = None
    best_score = None

    for _ in range(args.cem_iters):
        probs = torch.softmax(logits / max(args.temperature, 1e-6), dim=-1)
        dist = torch.distributions.Categorical(probs=probs.unsqueeze(0).expand(args.population, -1, -1))
        action_sequences = dist.sample()
        scores = evaluate_action_sequences(world_model, h0, z0, action_sequences, action_dim, args)

        top_score, top_idx = torch.max(scores, dim=0)
        if best_score is None or top_score.item() > best_score:
            best_score = float(top_score.item())
            best_sequence = action_sequences[top_idx].clone()

        elite_k = min(args.elites, args.population)
        elite_idx = torch.topk(scores, k=elite_k, dim=0).indices
        elites = action_sequences[elite_idx]
        elite_freq = F.one_hot(elites, num_classes=action_dim).float().mean(dim=0)
        elite_logits = torch.log(elite_freq + 1e-6)
        logits = args.cem_alpha * elite_logits + (1.0 - args.cem_alpha) * logits

    if best_sequence is None:
        best_sequence = torch.zeros(args.horizon, device=DEVICE, dtype=torch.long)
        best_score = float("-inf")
    selected_action = int(best_sequence[0].item())

    # Compute diagnostics for the first step of the best selected rollout.
    action0 = best_sequence[0].view(1)
    action0_oh = F.one_hot(action0, num_classes=action_dim).float()
    h1 = world_model.rssm.update_hidden(h0, z0, action0_oh)
    z1, _ = world_model.rssm.prior(h1)
    obs1 = world_model.reconstruct_obs(h1, z1)
    reward1 = world_model.predict_reward(h1, z1)
    best_step0_reward = float((args.reward_weight * reward1).item())
    best_step0_cost = float(observation_cost(obs1, args).item()) if args.use_obs_cost else 0.0

    return selected_action, best_score, best_step0_reward, best_step0_cost



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
    parser = argparse.ArgumentParser(description="MPC policy using world model + CEM on LunarLander")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--world_model", default="world_model.pt", help="World model checkpoint path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Enable render window")
    parser.add_argument("--horizon", type=int, default=25, help="MPC planning horizon")
    parser.add_argument("--population", type=int, default=384, help="CEM population size")
    parser.add_argument("--elites", type=int, default=48, help="Number of elites in CEM")
    parser.add_argument("--cem_iters", type=int, default=4, help="Number of CEM iterations")
    parser.add_argument("--cem_alpha", type=float, default=0.7, help="Logit smoothing (0..1)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--gamma", type=float, default=0.97, help="Planning discount")
    parser.add_argument("--reward_weight", type=float, default=0.2, help="Weight for predicted reward")
    parser.add_argument("--done_penalty", type=float, default=2.5, help="Penalty for predicted done probability")
    parser.add_argument("--obs_cost", dest="use_obs_cost", action="store_true",
                        help="Enable observation cost (default: OFF, use learned rewards only)")
    parser.set_defaults(use_obs_cost=False)

    parser.add_argument("--w_x", type=float, default=1.00, help="Base weight for |x|")
    parser.add_argument("--w_y", type=float, default=0.40, help="Weight for |y| (target y=0)")
    parser.add_argument("--w_vx", type=float, default=1.00, help="Weight for |vx|")
    parser.add_argument("--w_vy_down", type=float, default=1.20, help="Weight for excess downward speed")
    parser.add_argument("--w_angle", type=float, default=0.50, help="Base weight for |angle|")
    parser.add_argument("--w_angle_near", type=float, default=1.50, help="Extra |angle| weight near ground")
    parser.add_argument("--w_ang_vel", type=float, default=0.70, help="Base weight for |angular velocity|")
    parser.add_argument("--w_ang_vel_near", type=float, default=1.20, help="Extra |angular velocity| weight near ground")
    parser.add_argument("--y_near", type=float, default=0.35, help="Near-ground activation threshold")
    parser.add_argument("--y_near_temp", type=float, default=0.08, help="Near-ground gate smoothness")
    parser.add_argument("--vy_allowed_far", type=float, default=0.80, help="Allowed downward speed far from ground")
    parser.add_argument("--vy_allowed_near", type=float, default=0.18, help="Allowed downward speed near ground")
    args = parser.parse_args()

    if args.horizon < 1:
        raise ValueError("horizon must be >= 1")
    if args.population < 2:
        raise ValueError("population must be >= 2")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("elites must be in [1, population]")
    if args.y_near_temp <= 0.0:
        raise ValueError("y_near_temp must be > 0")

    set_seed(args.seed)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    world_model, action_dim = load_world_model(args.config, args.world_model, obs_dim)

    pygame.init()
    pygame.display.set_caption("WorldModel MPC Policy")
    screen = pygame.display.set_mode((lunar_lander_module.VIEWPORT_W, lunar_lander_module.VIEWPORT_H))
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()
    render_enabled = bool(args.render and screen is not None)

    print(
        "Controls: R toggle render, Q or ESC quit. "
        "Actions: 0=idle, 1=left side, 2=main, 3=right side\n"
        f"obs_cost={'ON' if args.use_obs_cost else 'OFF'}"
    )

    run_returns = []
    run_action_counts = np.zeros(action_dim, dtype=np.int64)
    run_action_counts_air = np.zeros(action_dim, dtype=np.int64)
    run_planner_steps = 0
    run_air_steps = 0
    run_near_count = 0
    run_near_abs_angle_sum = 0.0
    run_near_down_speed_sum = 0.0
    run_touchdown_abs_ang_vel = []

    stop_all = False
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_return = 0.0
        prev_action = 0
        action_counts = np.zeros(action_dim, dtype=np.int64)
        action_counts_air = np.zeros(action_dim, dtype=np.int64)
        planner_steps = 0
        air_steps = 0
        planner_score_sum = 0.0
        planner_step0_reward_sum = 0.0
        planner_step0_cost_sum = 0.0

        min_y = float("inf")
        max_abs_angle = 0.0
        max_down_speed = 0.0
        max_abs_vx = 0.0

        near_count = 0
        near_abs_angle_sum = 0.0
        near_down_speed_sum = 0.0
        near_abs_vx_sum = 0.0

        touchdown_snapshot = None
        h = world_model.rssm.init_hidden(batch_size=1, device=DEVICE)
        z = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
        t0 = time.perf_counter()

        for step in range(1, args.max_steps + 1):
            if pygame is not None and screen is not None:
                quit_requested, render_enabled = handle_events(render_enabled)
                if quit_requested:
                    done = True
                    stop_all = True
                    break

            landed = (len(obs) >= 8 and obs[6] >= 0.5 and obs[7] >= 0.5
                      and abs(obs[2]) < 0.05 and abs(obs[3]) < 0.05)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_action_oh = F.one_hot(
                torch.tensor([prev_action], device=DEVICE), num_classes=action_dim
            ).float()
            with torch.no_grad():
                h, z, _, _, _, _ = world_model.rssm.step(h, z, prev_action_oh, obs_t)
                if landed:
                    action = 0
                    best_score, best_step0_reward, best_step0_cost = 0.0, 0.0, 0.0
                else:
                    action, best_score, best_step0_reward, best_step0_cost = cem_plan(
                        world_model, h, z, action_dim, args
                    )

            planner_steps += 1
            planner_score_sum += float(best_score)
            planner_step0_reward_sum += float(best_step0_reward)
            planner_step0_cost_sum += float(best_step0_cost)
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

                # Count action mix in the air (exclude on the ground actions)
                if abs(vy) > 0.02 and (y > 0.0) and 0 <= int(action) < action_dim:
                    action_counts_air[int(action)] += 1
                    air_steps += 1

                if done:
                    touchdown_snapshot = (x, y, vx, vy, angle, ang_vel)
            obs = next_obs
            prev_action = action

            frame = env.render()
            elapsed = time.perf_counter() - t0
            fps_est = step / max(elapsed, 1e-6)
            overlay = [
                f"Episode {ep}/{args.episodes} Step {step}",
                f"Real reward {reward:+.3f}  Episode return {ep_return:+.2f}",
                f"CEM best score {best_score:+.3f}  Chosen action {action}",
                f"Best seq step0: reward={best_step0_reward:+.3f} cost={best_step0_cost:+.3f}",
                f"FPS cap {lunar_lander_module.FPS}  Measured {fps_est:.1f}",
            ]
            draw_frame(screen, font, frame, overlay, render_enabled)
            clock.tick(lunar_lander_module.FPS)  # Must match gym physics FPS

            if done:
                break

        print(
            f"[Episode {ep}] return={ep_return:+.2f}, steps={step}, "
            f"done={done}, elapsed={time.perf_counter() - t0:.1f}s"
        )
        if planner_steps > 0:
            action_mix = " ".join(
                f"{a}:{(100.0 * action_counts[a] / planner_steps):.1f}%"
                for a in range(action_dim)
            )
            print(f"[Episode {ep}] action_mix {action_mix}")
            if air_steps > 0:
                action_mix_air = " ".join(
                    f"{a}:{(100.0 * action_counts_air[a] / air_steps):.1f}%"
                    for a in range(action_dim)
                )
                print(f"[Episode {ep}] action_mix_air {action_mix_air}")
            print(
                f"[Episode {ep}] planner_avg score={planner_score_sum / planner_steps:+.3f} "
                f"step0_reward={planner_step0_reward_sum / planner_steps:+.3f} "
                f"step0_cost={planner_step0_cost_sum / planner_steps:+.3f}"
            )
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

        run_returns.append(ep_return)
        run_action_counts += action_counts
        run_action_counts_air += action_counts_air
        run_planner_steps += planner_steps
        run_air_steps += air_steps
        run_near_count += near_count
        run_near_abs_angle_sum += near_abs_angle_sum
        run_near_down_speed_sum += near_down_speed_sum
        if touchdown_snapshot is not None:
            run_touchdown_abs_ang_vel.append(abs(touchdown_snapshot[5]))

        if stop_all:
            break

    if run_returns:
        mean_return = float(np.mean(run_returns))
        worst_return = float(np.min(run_returns))
        main_mix = (100.0 * run_action_counts[2] / run_planner_steps) if run_planner_steps > 0 else 0.0
        main_mix_air = (100.0 * run_action_counts_air[2] / run_air_steps) if run_air_steps > 0 else 0.0
        near_avg_abs_angle = (run_near_abs_angle_sum / run_near_count) if run_near_count > 0 else float("nan")
        near_avg_down_speed = (run_near_down_speed_sum / run_near_count) if run_near_count > 0 else float("nan")
        touchdown_median_abs_ang_vel = (
            float(np.median(np.array(run_touchdown_abs_ang_vel, dtype=np.float64)))
            if run_touchdown_abs_ang_vel
            else float("nan")
        )
        touchdown_p90_abs_ang_vel = (
            float(np.percentile(np.array(run_touchdown_abs_ang_vel, dtype=np.float64), 90))
            if run_touchdown_abs_ang_vel
            else float("nan")
        )

        checks = {
            "mean_return >= -65": mean_return >= -65.0,
            "worst_return >= -120": worst_return >= -120.0,
            "near_ground avg_down_speed <= 0.75": near_avg_down_speed <= 0.75 if not np.isnan(near_avg_down_speed) else False,
            "near_ground avg_abs_angle <= 0.35": near_avg_abs_angle <= 0.35 if not np.isnan(near_avg_abs_angle) else False,
            "touchdown median abs(ang_vel) <= 2.0": (
                touchdown_median_abs_ang_vel <= 2.0 if not np.isnan(touchdown_median_abs_ang_vel) else False
            ),
            "touchdown p90 abs(ang_vel) <= 3.0": (
                touchdown_p90_abs_ang_vel <= 3.0 if not np.isnan(touchdown_p90_abs_ang_vel) else False
            ),
            "main action mix in [35%, 55%] (air only)": 35.0 <= main_mix_air <= 55.0,
        }
        passed = int(sum(1 for ok in checks.values() if ok))
        required_core = checks["near_ground avg_down_speed <= 0.75"] and checks["near_ground avg_abs_angle <= 0.35"]
        if passed >= 5 and required_core:
            verdict = "PASS"
        elif passed <= 3 or worst_return < -130.0:
            verdict = "FAIL"
        else:
            verdict = "BORDERLINE"

        print("[Run] ===== Checklist Scorecard =====")
        print(
            f"[Run] mean_return={mean_return:+.2f} worst_return={worst_return:+.2f} "
            f"main_action_mix={main_mix:.1f}% main_action_mix_air={main_mix_air:.1f}% "
            f"near_avg_abs_angle={near_avg_abs_angle:.3f} "
            f"near_avg_down_speed={near_avg_down_speed:.3f} "
            f"touchdown_median_abs_ang_vel={touchdown_median_abs_ang_vel:.3f} "
            f"touchdown_p90_abs_ang_vel={touchdown_p90_abs_ang_vel:.3f}"
        )
        for name, ok in checks.items():
            print(f"[Run] {'OK ' if ok else 'BAD'} {name}")
        print(f"[Run] verdict={verdict} ({passed}/{len(checks)} checks passed)")

    env.close()
    if pygame is not None and screen is not None:
        pygame.quit()


if __name__ == "__main__":
    main()
