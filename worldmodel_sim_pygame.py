import argparse
import os
import random
import time

import numpy as np
import torch
import yaml

from models import WorldModel

try:
    import pygame
except ImportError:
    pygame = None


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    if not os.path.exists(config_path):
        return 64, 128, 1, 4
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    wm_cfg = config.get("world_model", {})
    cap = wm_cfg.get("capacity", {})
    latent_dim = int(cap.get("latent_dim", 64))
    hidden_dim = int(cap.get("hidden_dim", 128))
    gru_num_layers = int(cap.get("gru_num_layers", 1))
    action_dim = int(wm_cfg.get("action_dim", 4))
    return latent_dim, hidden_dim, gru_num_layers, action_dim


def load_world_model(world_model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"World model checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    world_model.load_state_dict(state_dict)
    world_model.eval()


def load_validation_data(dataset_path):
    data = np.load(dataset_path)
    required = ["obs", "actions", "ep_index", "step_index"]
    for key in required:
        if key not in data:
            raise KeyError(f"Validation dataset must contain key '{key}'")
    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    ep_index = data["ep_index"].astype(np.int64)
    step_index = data["step_index"].astype(np.int64)
    return obs, actions, ep_index, step_index


def pick_random_episode_from_data(obs, actions, ep_index, step_index, rng):
    episode_ids = np.unique(ep_index)
    episode_id = int(rng.choice(episode_ids.tolist()))
    idxs = np.where(ep_index == episode_id)[0]
    order = np.argsort(step_index[idxs], kind="stable")
    idxs = idxs[order]
    return episode_id, obs[idxs], actions[idxs]


def rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rot.T


def compute_limits(obs):
    x_data = obs[:, 0]
    y_data = obs[:, 1]
    x_low, x_high = np.percentile(x_data, [1, 99])
    y_low, y_high = np.percentile(y_data, [1, 99])
    x_pad = max(0.3, 0.15 * (x_high - x_low + 1e-6))
    y_pad = max(0.25, 0.15 * (y_high - y_low + 1e-6))
    x_min = float(x_low - x_pad)
    x_max = float(x_high + x_pad)
    # Keep some extra room below y=0 so the ground line is clearly above the window bottom.
    y_min = min(-0.5, float(y_low - y_pad))
    y_max = max(1.2, float(y_high + y_pad))

    # Moderate zoom-in while keeping context.
    # zoom_factor > 1.0 means zooming in.
    zoom_factor = 1.8
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_half = 0.5 * (x_max - x_min) / zoom_factor
    y_half = 0.5 * (y_max - y_min) / zoom_factor
    x_min = x_center - x_half
    x_max = x_center + x_half
    y_min = y_center - y_half
    y_max = y_center + y_half

    # Keep y=0 (ground) visible a bit above the bottom edge.
    ground_bottom_margin = 0.2
    if y_min > -ground_bottom_margin:
        shift = y_min + ground_bottom_margin
        y_min -= shift
        y_max -= shift
    return x_min, x_max, y_min, y_max


def world_to_screen(p_world, limits, width, height):
    x_min, x_max, y_min, y_max = limits
    x, y = float(p_world[0]), float(p_world[1])
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    # Use isotropic scaling (same px-per-world-unit for x and y), so rotations stay rigid.
    pad = 24.0
    usable_w = max(width - 2.0 * pad, 1.0)
    usable_h = max(height - 2.0 * pad, 1.0)
    scale = min(usable_w / x_span, usable_h / y_span)

    world_w_px = x_span * scale
    world_h_px = y_span * scale
    x_off = 0.5 * (width - world_w_px)
    y_off = 0.5 * (height - world_h_px)

    px = int(x_off + (x - x_min) * scale)
    py = int(y_off + (y_max - y) * scale)
    return px, py


def draw_poly(screen, color, world_points, limits, width, height, width_line=0):
    points = [world_to_screen(p, limits, width, height) for p in world_points]
    pygame.draw.polygon(screen, color, points, width_line)


def draw_line(screen, color, p0, p1, limits, width, height, thickness=2):
    a = world_to_screen(p0, limits, width, height)
    b = world_to_screen(p1, limits, width, height)
    pygame.draw.line(screen, color, a, b, thickness)


def draw_frame(
    screen,
    font,
    obs,
    action,
    reward_pred,
    episode_reward_pred,
    best_episode_reward_pred,
    model_ms,
    loop_ms,
    avg_model_ms,
    avg_loop_ms,
    step_idx,
    episode_id,
    limits,
):
    width, height = screen.get_size()
    screen.fill((245, 245, 248))

    x = float(obs[0])
    y = float(obs[1])
    angle = float(obs[4]) if len(obs) > 4 else 0.0
    x_vel = float(obs[2]) if len(obs) > 2 else 0.0
    y_vel = float(obs[3]) if len(obs) > 3 else 0.0

    x_min, x_max, _, y_max = limits

    # Ground and top line.
    draw_line(screen, (20, 20, 20), np.array([x_min, 0.0]), np.array([x_max, 0.0]), limits, width, height, 3)
    top_y = y_max - 0.02 * (limits[3] - limits[2])
    draw_line(screen, (120, 120, 120), np.array([x_min, top_y]), np.array([x_max, top_y]), limits, width, height, 1)

    # Lander body.
    body_w = 0.18
    body_h = 0.10
    body_local = np.array(
        [
            [-body_w / 2, -body_h / 2],
            [body_w / 2, -body_h / 2],
            [body_w / 2, body_h / 2],
            [-body_w / 2, body_h / 2],
        ],
        dtype=np.float32,
    )
    body_world = rotate(body_local, angle) + np.array([x, y], dtype=np.float32)
    draw_poly(screen, (78, 121, 167), body_world, limits, width, height, 0)

    # Legs.
    leg_len = 0.10
    left_base = np.array([-body_w * 0.35, -body_h / 2], dtype=np.float32)
    right_base = np.array([body_w * 0.35, -body_h / 2], dtype=np.float32)
    left_tip = left_base + np.array([-0.05, -leg_len], dtype=np.float32)
    right_tip = right_base + np.array([0.05, -leg_len], dtype=np.float32)
    left_base_w = rotate(left_base[None, :], angle)[0] + np.array([x, y], dtype=np.float32)
    right_base_w = rotate(right_base[None, :], angle)[0] + np.array([x, y], dtype=np.float32)
    left_tip_w = rotate(left_tip[None, :], angle)[0] + np.array([x, y], dtype=np.float32)
    right_tip_w = rotate(right_tip[None, :], angle)[0] + np.array([x, y], dtype=np.float32)
    draw_line(screen, (50, 50, 50), left_base_w, left_tip_w, limits, width, height, 3)
    draw_line(screen, (50, 50, 50), right_base_w, right_tip_w, limits, width, height, 3)

    # Engine flames.
    flame_color = (255, 127, 14)
    main_flame_len = 0.12
    side_flame_len = 0.08
    if action == 2:
        nozzle = np.array([0.0, -body_h / 2], dtype=np.float32)
        p1 = nozzle + np.array([-0.03, -0.01], dtype=np.float32)
        p2 = nozzle + np.array([0.03, -0.01], dtype=np.float32)
        p3 = nozzle + np.array([0.0, -main_flame_len], dtype=np.float32)
        flame = rotate(np.vstack([p1, p2, p3]), angle) + np.array([x, y], dtype=np.float32)
        draw_poly(screen, flame_color, flame, limits, width, height, 0)
    if action == 1:
        nozzle = np.array([body_w / 2, 0.0], dtype=np.float32)
        p1 = nozzle + np.array([side_flame_len, 0.0], dtype=np.float32)
        p2 = nozzle + np.array([0.02, 0.02], dtype=np.float32)
        p3 = nozzle + np.array([0.02, -0.02], dtype=np.float32)
        flame = rotate(np.vstack([p1, p2, p3]), angle) + np.array([x, y], dtype=np.float32)
        draw_poly(screen, flame_color, flame, limits, width, height, 0)
    if action == 3:
        nozzle = np.array([-body_w / 2, 0.0], dtype=np.float32)
        p1 = nozzle + np.array([-side_flame_len, 0.0], dtype=np.float32)
        p2 = nozzle + np.array([-0.02, 0.02], dtype=np.float32)
        p3 = nozzle + np.array([-0.02, -0.02], dtype=np.float32)
        flame = rotate(np.vstack([p1, p2, p3]), angle) + np.array([x, y], dtype=np.float32)
        draw_poly(screen, flame_color, flame, limits, width, height, 0)

    # Text overlay.
    action_name = {0: "NO-OP", 1: "LEFT", 2: "MAIN", 3: "RIGHT"}[action]
    lines = [
        f"World Model Sim (pygame) | ep={episode_id} step={step_idx} action={action_name}",
        (
            f"pred_reward={reward_pred:+.4f} ep_pred_return={episode_reward_pred:+.2f} "
            f"best_prev_ep_return={best_episode_reward_pred:+.2f} "
            f"x={x:+.3f} y={y:+.3f} vx={x_vel:+.3f} vy={y_vel:+.3f} angle={angle:+.3f}"
        ),
        f"model_ms={model_ms:.2f} loop_ms={loop_ms:.2f} avg_model_ms={avg_model_ms:.2f} avg_loop_ms={avg_loop_ms:.2f}",
        "Keys: Up=main Left/Right=side engines Space=next episode Q/Esc=quit",
    ]
    for i, text in enumerate(lines):
        surf = font.render(text, True, (20, 20, 20))
        screen.blit(surf, (12, 10 + i * 22))


def run_simulation(
    world_model,
    obs,
    actions,
    ep_index,
    step_index,
    rng,
    fps,
    warmup_steps,
    stochastic,
    max_steps,
    max_episode_steps,
    width,
    height,
    print_every,
):
    limits = compute_limits(obs)
    pygame.init()
    pygame.display.set_caption("WorldModel Sim (pygame)")
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    with torch.no_grad():
        action_dim = world_model.rssm.action_dim
        latent_dim = world_model.rssm.latent_dim
        total_step_count = 0
        print("Action mapping: up=2(main), left=1(left engine), right=3(right engine), no key=0(no-op)")
        print("Controls: Space=next episode, Q/Esc=quit")

        running = True
        next_episode_requested = False

        timing_count = 0
        timing_model_sum = 0.0
        timing_loop_sum = 0.0
        avg_model_ms = 0.0
        avg_loop_ms = 0.0
        best_episode_reward_pred = float("-inf")

        while running:
            if max_steps is not None and total_step_count >= max_steps:
                break

            episode_id, episode_obs, episode_actions = pick_random_episode_from_data(
                obs=obs,
                actions=actions,
                ep_index=ep_index,
                step_index=step_index,
                rng=rng,
            )
            warmup_steps_ep = int(max(1, min(warmup_steps, len(episode_obs))))
            h = world_model.rssm.init_hidden(1, DEVICE)
            z_prev = torch.zeros(1, latent_dim, device=DEVICE)
            a_prev = torch.zeros(1, action_dim, device=DEVICE)

            for t in range(warmup_steps_ep):
                h = world_model.rssm.update_hidden(h, z_prev, a_prev)
                obs_t = torch.tensor(episode_obs[t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                mean_post, logstd_post = world_model.rssm.posterior(h, obs_t)
                z_prev = world_model.rssm.sample_latent(mean_post, logstd_post) if stochastic else mean_post
                a_real = int(episode_actions[t])
                a_prev = torch.nn.functional.one_hot(
                    torch.tensor([a_real], device=DEVICE), num_classes=action_dim
                ).float()

            z = z_prev
            episode_step_count = 0
            episode_reward_pred = 0.0
            print(f"Warm-up complete on episode {episode_id} with {warmup_steps_ep} posterior steps.")

            while running:
                loop_t0 = time.perf_counter()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False
                        elif event.key == pygame.K_SPACE:
                            next_episode_requested = True

                if not running:
                    break
                if max_steps is not None and total_step_count >= max_steps:
                    running = False
                    break

                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    action = 2
                elif keys[pygame.K_LEFT]:
                    action = 1
                elif keys[pygame.K_RIGHT]:
                    action = 3
                else:
                    action = 0

                a_onehot = torch.nn.functional.one_hot(
                    torch.tensor([action], device=DEVICE), num_classes=action_dim
                ).float()

                t0 = time.perf_counter()
                h = world_model.rssm.update_hidden(h, z, a_onehot)
                mean_prior, logstd_prior = world_model.rssm.prior(h)
                z = world_model.rssm.sample_latent(mean_prior, logstd_prior) if stochastic else mean_prior
                obs_pred = world_model.reconstruct_obs(h, z)
                reward_pred = world_model.predict_reward(h, z)
                model_ms = (time.perf_counter() - t0) * 1000.0

                current_obs = obs_pred.squeeze(0).cpu().numpy()
                reward_val = float(reward_pred.item())
                episode_reward_pred += reward_val
                y_pred = float(current_obs[1]) if len(current_obs) > 1 else 1.0

                frame_ms = clock.tick(fps)
                loop_ms = (time.perf_counter() - loop_t0) * 1000.0
                timing_count += 1
                timing_model_sum += model_ms
                timing_loop_sum += loop_ms
                if timing_count > 0:
                    avg_model_ms = timing_model_sum / timing_count
                    avg_loop_ms = timing_loop_sum / timing_count

                draw_frame(
                    screen=screen,
                    font=font,
                    obs=current_obs,
                    action=action,
                    reward_pred=reward_val,
                    episode_reward_pred=episode_reward_pred,
                    best_episode_reward_pred=(
                        best_episode_reward_pred if np.isfinite(best_episode_reward_pred) else 0.0
                    ),
                    model_ms=model_ms,
                    loop_ms=loop_ms,
                    avg_model_ms=avg_model_ms,
                    avg_loop_ms=avg_loop_ms,
                    step_idx=episode_step_count,
                    episode_id=episode_id,
                    limits=limits,
                )
                pygame.display.flip()

                if print_every > 0 and ((episode_step_count + 1) % print_every == 0):
                    print(
                        f"[Sim] ep={episode_id} step={episode_step_count} action={action} "
                        f"pred_reward={reward_val:+.4f} y={y_pred:+.4f} "
                        f"model_ms={model_ms:.2f} loop_ms={loop_ms:.2f} frame_ms={frame_ms:.2f} "
                        f"avg_model_ms={avg_model_ms:.2f} avg_loop_ms={avg_loop_ms:.2f}"
                    )
                    timing_count = 0
                    timing_model_sum = 0.0
                    timing_loop_sum = 0.0

                episode_step_count += 1
                total_step_count += 1

                if y_pred <= 0.0:
                    best_episode_reward_pred = max(best_episode_reward_pred, episode_reward_pred)
                    print(
                        f"Episode {episode_id} ended at imagined step {episode_step_count}: "
                        f"predicted y={y_pred:.4f} <= 0. ep_pred_return={episode_reward_pred:+.2f} "
                        f"best_prev_ep_return={best_episode_reward_pred:+.2f}. Loading another episode..."
                    )
                    break
                if next_episode_requested:
                    next_episode_requested = False
                    best_episode_reward_pred = max(best_episode_reward_pred, episode_reward_pred)
                    print(f"Episode {episode_id} interrupted by user at step {episode_step_count}.")
                    break
                if episode_step_count >= max_episode_steps:
                    best_episode_reward_pred = max(best_episode_reward_pred, episode_reward_pred)
                    print(
                        f"Episode {episode_id} reached max imagined length {max_episode_steps}. "
                        f"ep_pred_return={episode_reward_pred:+.2f} "
                        f"best_prev_ep_return={best_episode_reward_pred:+.2f}. Loading another episode..."
                    )
                    break

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Interactive world-model simulation with pygame")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--world_model", default="world_model.pt", help="World model checkpoint path")
    parser.add_argument(
        "--val_dataset",
        default="lunarlander_val_dataset.npz",
        help="Validation dataset path (must contain obs/actions/ep_index/step_index)",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for picking episode")
    parser.add_argument("--warmup_steps", type=int, default=15, help="Posterior warm-up steps from episode start")
    parser.add_argument("--fps", type=float, default=30.0, help="Target render/update frequency")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum imagined prior steps after warm-up (default: unlimited until quit)",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=500,
        help="Maximum imagined steps per episode before forcing reset (default: 500)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from prior/posterior instead of using means (more stochastic, less stable)",
    )
    parser.add_argument("--width", type=int, default=960, help="Window width in pixels")
    parser.add_argument("--height", type=int, default=640, help="Window height in pixels")
    parser.add_argument("--print_every", type=int, default=10, help="Print stats every N simulated steps")
    args = parser.parse_args()

    if pygame is None:
        raise ImportError(
            "pygame is required for worldmodel_sim_pygame.py. Install with: pip install pygame"
        )
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.max_episode_steps <= 0:
        raise ValueError("--max_episode_steps must be > 0")
    if args.width <= 200 or args.height <= 150:
        raise ValueError("--width/--height are too small")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    latent_dim, hidden_dim, gru_num_layers, action_dim = load_config(args.config)
    obs, actions, ep_index, step_index = load_validation_data(args.val_dataset)
    obs_dim = int(obs.shape[1])

    world_model = WorldModel(
        obs_dim,
        action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers,
    ).to(DEVICE)
    load_world_model(world_model, args.world_model)
    print(f"Loaded world model from {args.world_model}")
    print(f"Loaded validation dataset with {len(np.unique(ep_index))} episodes")

    run_simulation(
        world_model=world_model,
        obs=obs,
        actions=actions,
        ep_index=ep_index,
        step_index=step_index,
        rng=random.Random(args.seed),
        fps=float(args.fps),
        warmup_steps=int(args.warmup_steps),
        stochastic=bool(args.stochastic),
        max_steps=args.max_steps,
        max_episode_steps=int(args.max_episode_steps),
        width=int(args.width),
        height=int(args.height),
        print_every=int(args.print_every),
    )


if __name__ == "__main__":
    main()
