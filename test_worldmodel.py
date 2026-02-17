import argparse
import os
import yaml
import numpy as np
import torch

from models import WorldModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    if not os.path.exists(config_path):
        return 64, 128, 1, 4
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    capacity = config.get("world_model", {}).get("capacity", {})
    latent_dim = int(capacity.get("latent_dim", 64))
    hidden_dim = int(capacity.get("hidden_dim", 128))
    gru_num_layers = int(capacity.get("gru_num_layers", 1))
    action_dim = int(config.get("world_model", {}).get("action_dim", 4))
    return latent_dim, hidden_dim, gru_num_layers, action_dim


def load_episodes(dataset_path):
    data = np.load(dataset_path)
    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    rewards = data["rewards"].astype(np.float32)
    next_obs = data["next_obs"].astype(np.float32)
    dones = data["dones"].astype(np.int64)
    ep_index = data["ep_index"].astype(np.int64)
    step_index = data["step_index"].astype(np.int64)

    episodes = []
    for ep_id in np.unique(ep_index):
        idxs = np.where(ep_index == ep_id)[0]
        if idxs.size == 0:
            continue
        order = np.argsort(step_index[idxs], kind="stable")
        idxs = idxs[order]
        # obs_full = [obs[0], obs[1], ..., obs[N-1], terminal] for alignment with preds
        obs_full = np.concatenate([obs[idxs], next_obs[idxs[-1:]]], axis=0)
        episodes.append({
            "ep_id": int(ep_id),
            "obs": obs_full,
            "actions": actions[idxs],
            "rewards": rewards[idxs],
            "dones": dones[idxs],
        })
    return episodes


def plot_observations(obs_gt, obs_pred, title, plot_dir=None, filename="plot.png",
                      rewards_gt=None, reward_pred=None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Plotting requested but matplotlib is not available.")
        return

    obs_names = ["x", "y", "vx", "vy", "angle", "ang_vel", "left_contact", "right_contact"]
    obs_dim = obs_gt.shape[1]
    n_rows = obs_dim + (1 if rewards_gt is not None and reward_pred is not None else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    for d in range(obs_dim):
        axes[d].plot(obs_gt[:, d], label="gt")
        axes[d].plot(obs_pred[:, d], label="pred", alpha=0.7)
        label = obs_names[d] if d < len(obs_names) else f"obs[{d}]"
        axes[d].set_ylabel(label)
        axes[d].legend(loc="upper right", fontsize=8)
    ax_reward = axes[obs_dim] if rewards_gt is not None and reward_pred is not None else axes[-1]
    if rewards_gt is not None and reward_pred is not None:
        n = min(len(rewards_gt), len(reward_pred))
        ax_reward.plot(rewards_gt[:n], label="gt")
        ax_reward.plot(reward_pred[:n], label="pred", alpha=0.7)
        ax_reward.set_ylabel("reward")
        ax_reward.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t")
    fig.suptitle(title)
    fig.tight_layout()

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, filename)
        fig.savefig(path)
        print(f"Saved plot to {path}")
        plt.close(fig)
    else:
        plt.show()


def animate_lander(obs_gt, obs_pred=None, title="", plot_dir=None, filename="anim.mp4"):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D
    except Exception:
        print("Animation requested but matplotlib is not available.")
        return

    x_gt = obs_gt[:, 0]
    y_gt = obs_gt[:, 1]
    ang_gt = obs_gt[:, 4]

    if obs_pred is not None:
        x_pr = obs_pred[:, 0]
        y_pr = obs_pred[:, 1]
        ang_pr = obs_pred[:, 4]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    pad = 0.2
    x_all = x_gt if obs_pred is None else np.concatenate([x_gt, x_pr])
    y_all = y_gt if obs_pred is None else np.concatenate([y_gt, y_pr])
    xmin = float(x_all.min() - pad)
    xmax = float(x_all.max() + pad)
    ymin = float(y_all.min() - pad)
    ymax = float(y_all.max() + pad)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    box_size = 0.08
    gt_box = Rectangle((-box_size / 2, -box_size / 2), box_size, box_size,
                       facecolor="none", edgecolor="black", linewidth=2)
    gt_top = plt.Line2D([0, 0], [0, box_size * 0.9], color="black", linewidth=2)
    gt_leg_l = plt.Line2D([0, -box_size * 0.8], [0, -box_size * 0.8], color="black", linewidth=2)
    gt_leg_r = plt.Line2D([0, box_size * 0.8], [0, -box_size * 0.8], color="black", linewidth=2)
    ax.add_patch(gt_box)
    ax.add_line(gt_top)
    ax.add_line(gt_leg_l)
    ax.add_line(gt_leg_r)

    if obs_pred is not None:
        pr_box = Rectangle((-box_size / 2, -box_size / 2), box_size, box_size,
                           facecolor="none", edgecolor="red", linewidth=2)
        pr_top = plt.Line2D([0, 0], [0, box_size * 0.9], color="red", linewidth=2)
        pr_leg_l = plt.Line2D([0, -box_size * 0.8], [0, -box_size * 0.8], color="red", linewidth=2)
        pr_leg_r = plt.Line2D([0, box_size * 0.8], [0, -box_size * 0.8], color="red", linewidth=2)
        ax.add_patch(pr_box)
        ax.add_line(pr_top)
        ax.add_line(pr_leg_l)
        ax.add_line(pr_leg_r)

    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=2, label="gt"),
    ]
    if obs_pred is not None:
        legend_handles.append(plt.Line2D([0], [0], color="red", linewidth=2, label="world_model"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    def set_pose(rect, top, leg_l, leg_r, x, y, angle):
        trans = Affine2D().rotate_around(0, 0, angle).translate(x, y) + ax.transData
        rect.set_transform(trans)
        top.set_transform(trans)
        leg_l.set_transform(trans)
        leg_r.set_transform(trans)

    def update(frame):
        set_pose(gt_box, gt_top, gt_leg_l, gt_leg_r, x_gt[frame], y_gt[frame], ang_gt[frame])
        artists = [gt_box, gt_top, gt_leg_l, gt_leg_r]
        if obs_pred is not None:
            set_pose(pr_box, pr_top, pr_leg_l, pr_leg_r, x_pr[frame], y_pr[frame], ang_pr[frame])
            artists += [pr_box, pr_top, pr_leg_l, pr_leg_r]
        return artists

    ani = animation.FuncAnimation(fig, update, frames=len(x_gt), interval=40, blit=True)

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, filename)
        try:
            ani.save(path, fps=25, dpi=120)
            print(f"Saved animation to {path}")
        except Exception:
            print("Failed to save animation. Install ffmpeg or try without plot_dir.")
        plt.close(fig)
    else:
        plt.show()


def rollout_teacher(world_model, episode):
    obs_gt = episode["obs"]
    actions = episode["actions"]
    rewards_gt = episode["rewards"]
    obs_pred = []
    reward_pred = []

    with torch.no_grad():
        h = world_model.rssm.init_hidden(1, DEVICE)
        z_prev = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
        a_prev = torch.zeros(1, world_model.rssm.action_dim, device=DEVICE)

        for t in range(len(actions)):
            h = world_model.rssm.update_hidden(h, z_prev, a_prev)
            obs_t = torch.tensor(obs_gt[t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean_post, logstd_post = world_model.rssm.posterior(h, obs_t)
            z_t = mean_post
            obs_hat = world_model.reconstruct_obs(h, z_t)
            obs_pred.append(obs_hat.squeeze(0).cpu().numpy())
            r_hat = world_model.predict_reward(h, z_t)
            reward_pred.append(r_hat.item())

            a_prev = torch.nn.functional.one_hot(
                torch.tensor([actions[t]], device=DEVICE),
                num_classes=world_model.rssm.action_dim
            ).float()
            z_prev = z_t

    obs_gt_c = obs_gt[: len(obs_pred)]
    obs_pred_a = np.asarray(obs_pred, dtype=np.float32)
    reward_pred_a = np.asarray(reward_pred, dtype=np.float32)
    rewards_gt_c = rewards_gt[: len(reward_pred)]
    return obs_gt_c, obs_pred_a, rewards_gt_c, reward_pred_a


def rollout_open(world_model, episode):
    obs_gt = episode["obs"]
    actions = episode["actions"]
    rewards_gt = episode["rewards"]
    dones = episode["dones"]
    obs_pred = []
    reward_pred = []

    with torch.no_grad():
        h = world_model.rssm.init_hidden(1, DEVICE)
        z_prev = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
        a_prev = torch.zeros(1, world_model.rssm.action_dim, device=DEVICE)

        # Initialize from posterior at t=0
        h = world_model.rssm.update_hidden(h, z_prev, a_prev)
        obs_t = torch.tensor(obs_gt[0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mean_post, logstd_post = world_model.rssm.posterior(h, obs_t)
        z_prev = mean_post

        for t in range(len(actions)):
            if dones[t]:
                break
            h = world_model.rssm.update_hidden(h, z_prev, a_prev)
            mean_prior, logstd_prior = world_model.rssm.prior(h)
            z_t = mean_prior
            obs_hat = world_model.reconstruct_obs(h, z_t)
            obs_pred.append(obs_hat.squeeze(0).cpu().numpy())
            r_hat = world_model.predict_reward(h, z_t)
            reward_pred.append(r_hat.item())

            a_prev = torch.nn.functional.one_hot(
                torch.tensor([actions[t]], device=DEVICE),
                num_classes=world_model.rssm.action_dim
            ).float()
            z_prev = z_t

    obs_pred_a = np.asarray(obs_pred, dtype=np.float32) if obs_pred else np.empty((0, obs_gt.shape[1]), dtype=np.float32)
    obs_gt_compare = obs_gt[1 : 1 + len(obs_pred)]
    reward_pred_a = np.asarray(reward_pred, dtype=np.float32)
    rewards_gt_c = rewards_gt[: len(reward_pred)]
    return obs_gt_compare, obs_pred_a, rewards_gt_c, reward_pred_a


def rollout_sim(world_model, episode, constant_action):
    obs_gt = episode["obs"]
    dones = episode["dones"]
    obs_pred = []
    prior_stds = []

    with torch.no_grad():
        h = world_model.rssm.init_hidden(1, DEVICE)
        z_prev = torch.zeros(1, world_model.rssm.latent_dim, device=DEVICE)
        a_prev = torch.zeros(1, world_model.rssm.action_dim, device=DEVICE)

        # Initialize from posterior at t=0
        h = world_model.rssm.update_hidden(h, z_prev, a_prev)
        obs_t = torch.tensor(obs_gt[0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mean_post, logstd_post = world_model.rssm.posterior(h, obs_t)
        z_prev = mean_post

        for t in range(len(episode["actions"])):
            if dones[t]:
                break
            h = world_model.rssm.update_hidden(h, z_prev, a_prev)
            mean_prior, logstd_prior = world_model.rssm.prior(h)
            std_prior = torch.exp(logstd_prior)
            prior_stds.append(std_prior.mean().item())
            z_t = mean_prior
            obs_hat = world_model.reconstruct_obs(h, z_t)
            obs_pred.append(obs_hat.squeeze(0).cpu().numpy())

            a_prev = torch.nn.functional.one_hot(
                torch.tensor([constant_action], device=DEVICE),
                num_classes=world_model.rssm.action_dim
            ).float()
            z_prev = z_t

    obs_pred = np.asarray(obs_pred, dtype=np.float32) if obs_pred else np.empty((0, obs_gt.shape[1]), dtype=np.float32)
    prior_stds = np.asarray(prior_stds, dtype=np.float64)
    return obs_pred, prior_stds


def compute_reward_metrics(rewards_gt, reward_pred):
    if len(rewards_gt) == 0 or len(reward_pred) == 0:
        return None
    n = min(len(rewards_gt), len(reward_pred))
    gt = rewards_gt[:n]
    pred = reward_pred[:n]
    mae = np.mean(np.abs(gt - pred))
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    sign_match = np.sum(np.sign(gt) == np.sign(pred))
    sign_acc = sign_match / n
    return {"mae": mae, "rmse": rmse, "sign_acc": sign_acc}


def compute_obs_metrics(obs_gt, obs_pred):
    if obs_gt.size == 0 or obs_pred.size == 0:
        return None
    n = min(len(obs_gt), len(obs_pred))
    gt = obs_gt[:n]
    pred = obs_pred[:n]
    mae = np.mean(np.abs(gt - pred))
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    return {"mae": mae, "rmse": rmse}


def load_checkpoint(world_model, checkpoint_path):
    if checkpoint_path and os.path.exists(checkpoint_path):
        world_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Loaded world model checkpoint: {checkpoint_path}")
        return
    if os.path.exists("world_model.pt"):
        world_model.load_state_dict(torch.load("world_model.pt", map_location=DEVICE))
        print("Loaded world model from world_model.pt")
        return
    raise FileNotFoundError("No world model checkpoint found.")


def main():
    parser = argparse.ArgumentParser(description="Minimal world model evaluation")
    parser.add_argument("--dataset", default="lunarlander_val_dataset.npz",
                        help="Path to dataset .npz")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to world model checkpoint (default: world_model.pt)")
    parser.add_argument("--mode", choices=["teacher", "open", "sim"], default="teacher",
                        help="teacher: posterior reconstruction; open: prior rollout; sim: constant action rollout")
    parser.add_argument("--max_episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for selecting episodes")
    parser.add_argument("--plot_dir", default=None,
                        help="If set, save plots/animations to this directory")
    parser.add_argument("--animate", action="store_true",
                        help="Render animations (GT vs pred for teacher/open, pred only for sim)")
    parser.add_argument("--constant_action", type=int, default=0,
                        help="Action id for sim mode")
    args = parser.parse_args()

    episodes = load_episodes(args.dataset)
    if not episodes:
        print("No episodes found in dataset.")
        return

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(episodes), size=min(args.max_episodes, len(episodes)), replace=False)

    latent_dim, hidden_dim, gru_num_layers, action_dim = load_config(args.config)
    obs_dim = episodes[0]["obs"].shape[1]
    world_model = WorldModel(
        obs_dim,
        action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        gru_num_layers=gru_num_layers,
    ).to(DEVICE)
    load_checkpoint(world_model, args.checkpoint)
    world_model.eval()

    for i, ep_idx in enumerate(picks, start=1):
        ep = episodes[ep_idx]
        ep_id = ep["ep_id"]
        if args.mode == "teacher":
            obs_gt, obs_pred, rewards_gt, reward_pred = rollout_teacher(world_model, ep)
            r_metrics = compute_reward_metrics(rewards_gt, reward_pred)
            o_metrics = compute_obs_metrics(obs_gt, obs_pred)
            r_str = f"reward MAE={r_metrics['mae']:.4f} RMSE={r_metrics['rmse']:.4f} sign_acc={r_metrics['sign_acc']:.4f}" if r_metrics else "reward N/A"
            o_str = f"obs MAE={o_metrics['mae']:.4f} RMSE={o_metrics['rmse']:.4f}" if o_metrics else "obs N/A"
            print(f"Episode {ep_id} (teacher): {r_str} | {o_str}")
            plot_observations(obs_gt, obs_pred, f"Teacher episode {ep_id}",
                              plot_dir=args.plot_dir, filename=f"teacher_{ep_id:03d}.png",
                              rewards_gt=rewards_gt, reward_pred=reward_pred)
            if args.animate:
                animate_lander(obs_gt, obs_pred, title=f"Teacher ep {ep_id}",
                               plot_dir=args.plot_dir, filename=f"teacher_{ep_id:03d}.mp4")
        elif args.mode == "open":
            obs_gt, obs_pred, rewards_gt, reward_pred = rollout_open(world_model, ep)
            r_metrics = compute_reward_metrics(rewards_gt, reward_pred)
            o_metrics = compute_obs_metrics(obs_gt, obs_pred)
            r_str = f"reward MAE={r_metrics['mae']:.4f} RMSE={r_metrics['rmse']:.4f} sign_acc={r_metrics['sign_acc']:.4f}" if r_metrics else "reward N/A"
            o_str = f"obs MAE={o_metrics['mae']:.4f} RMSE={o_metrics['rmse']:.4f}" if o_metrics else "obs N/A"
            print(f"Episode {ep_id} (open): {r_str} | {o_str}")
            plot_observations(obs_gt, obs_pred, f"Open episode {ep_id}",
                              plot_dir=args.plot_dir, filename=f"open_{ep_id:03d}.png",
                              rewards_gt=rewards_gt, reward_pred=reward_pred)
            if args.animate:
                animate_lander(obs_gt, obs_pred, title=f"Open ep {ep_id}",
                               plot_dir=args.plot_dir, filename=f"open_{ep_id:03d}.mp4")
        else:
            obs_pred, prior_stds = rollout_sim(world_model, ep, args.constant_action)
            if prior_stds.size > 0:
                print(f"Sim ep {ep_id}: prior_std min={prior_stds.min():.4f} median={np.median(prior_stds):.4f} "
                      f"mean={prior_stds.mean():.4f} max={prior_stds.max():.4f} (over {prior_stds.size} steps)")
            if args.animate:
                animate_lander(obs_gt=obs_pred, obs_pred=None, title=f"Sim ep {ep_id}",
                               plot_dir=args.plot_dir, filename=f"sim_{ep_id:03d}.mp4")


if __name__ == "__main__":
    main()
