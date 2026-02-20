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


def load_world_model(config_path, checkpoint_path, obs_dim):
    latent_dim, hidden_dim, gru_num_layers, action_dim = load_config(config_path)
    world_model = WorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers,
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
    cost = (
        args.w_x * torch.abs(x)
        + args.w_vx * torch.abs(vx)
        + args.w_vy_down * torch.relu(-vy)
        + args.w_angle * torch.abs(angle)
        + args.w_ang_vel * torch.abs(ang_vel)
        + args.w_low_y * torch.relu(args.y_target - y)
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
        step_score = args.reward_weight * reward_pred - observation_cost(obs_pred, args)
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
    return selected_action, best_score



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
    parser.add_argument("--horizon", type=int, default=15, help="MPC planning horizon")
    parser.add_argument("--population", type=int, default=256, help="CEM population size")
    parser.add_argument("--elites", type=int, default=32, help="Number of elites in CEM")
    parser.add_argument("--cem_iters", type=int, default=4, help="Number of CEM iterations")
    parser.add_argument("--cem_alpha", type=float, default=0.7, help="Logit smoothing (0..1)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--gamma", type=float, default=0.97, help="Planning discount")
    parser.add_argument("--reward_weight", type=float, default=0.2, help="Weight for predicted reward")
    parser.add_argument("--done_penalty", type=float, default=2.0, help="Penalty for predicted done probability")

    parser.add_argument("--w_x", type=float, default=0.15, help="Weight for |x|")
    parser.add_argument("--w_vx", type=float, default=0.60, help="Weight for |vx|")
    parser.add_argument("--w_vy_down", type=float, default=0.85, help="Weight for downward speed")
    parser.add_argument("--w_angle", type=float, default=1.10, help="Weight for |angle|")
    parser.add_argument("--w_ang_vel", type=float, default=0.55, help="Weight for |angular velocity|")
    parser.add_argument("--w_low_y", type=float, default=0.40, help="Penalty for being below y_target")
    parser.add_argument("--y_target", type=float, default=0.20, help="Target minimum y")
    args = parser.parse_args()

    if args.horizon < 1:
        raise ValueError("horizon must be >= 1")
    if args.population < 2:
        raise ValueError("population must be >= 2")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("elites must be in [1, population]")

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
        "Actions: 0=idle, 1=left side, 2=main, 3=right side"
    )

    stop_all = False
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_return = 0.0
        prev_action = 0
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

            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_action_oh = F.one_hot(
                torch.tensor([prev_action], device=DEVICE), num_classes=action_dim
            ).float()
            with torch.no_grad():
                h, z, _, _, _, _ = world_model.rssm.step(h, z, prev_action_oh, obs_t)
                action, best_score = cem_plan(world_model, h, z, action_dim, args)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            obs = next_obs
            prev_action = action

            frame = env.render()
            elapsed = time.perf_counter() - t0
            fps_est = step / max(elapsed, 1e-6)
            overlay = [
                f"Episode {ep}/{args.episodes} Step {step}",
                f"Real reward {reward:+.3f}  Episode return {ep_return:+.2f}",
                f"CEM best score {best_score:+.3f}  Chosen action {action}",
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
        if stop_all:
            break

    env.close()
    if pygame is not None and screen is not None:
        pygame.quit()


if __name__ == "__main__":
    main()
