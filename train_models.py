# train_models.py - World model and actor-critic training for LunarLander Dreamer
import argparse
import os
import random
import yaml
import datetime
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models import WorldModel, Actor, Critic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloader_generator(seed):
    """Get a generator for DataLoader reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def log_message(message, log_path=None):
    print(message)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def get_latest_checkpoint(model_name, directory=CHECKPOINT_DIR):
    """Find the most recent checkpoint for a given model name."""
    pattern = os.path.join(directory, f"{model_name}_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def save_checkpoint_with_timestamp(model, model_name, epoch, directory=CHECKPOINT_DIR, log_path=None):
    """Save a model checkpoint with timestamp."""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"{model_name}_{timestamp}_epoch_{epoch}.pt")
    torch.save(model.state_dict(), filename)
    log_message(f"Saved checkpoint: {filename}", log_path)
    return filename


# -----------------------------------------------------------------------------
# World model: SequenceDataset for sequence-based training
# -----------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """Samples sequences from real episodes for world model training."""
    def __init__(self, path, sequence_length, action_dim, random_start=True):
        data = np.load(path)
        self.obs = data["obs"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.rewards = data["rewards"].astype(np.float32)
        self.next_obs = data["next_obs"].astype(np.float32)
        self.dones = data["dones"].astype(np.int64)
        self.ep_index = data["ep_index"].astype(np.int64)
        self.step_index = data["step_index"].astype(np.int64)
        self.sequence_length = int(sequence_length)
        self.random_start = bool(random_start)
        self.obs_dim = self.obs.shape[1]
        self.action_dim = int(action_dim)
        if self.actions.size and self.actions.max() >= self.action_dim:
            raise ValueError(
                f"Dataset actions exceed action_dim={self.action_dim}; "
                f"max action={int(self.actions.max())}"
            )

        self.episode_ids = np.unique(self.ep_index)
        self.episode_indices = []
        self.start_positions = []
        for ep_id in self.episode_ids:
            idxs = np.where(self.ep_index == ep_id)[0]
            if idxs.size == 0:
                continue
            order = np.argsort(self.step_index[idxs], kind="stable")
            idxs = idxs[order]
            ep_pos = len(self.episode_indices)
            self.episode_indices.append(idxs)
            for pos in range(len(idxs)):
                self.start_positions.append((ep_pos, pos))

    def __len__(self):
        return len(self.start_positions)

    def __getitem__(self, idx):
        if self.random_start:
            ep_pos, start = self.start_positions[np.random.randint(len(self.start_positions))]
        else:
            ep_pos, start = self.start_positions[idx]
        idxs = self.episode_indices[ep_pos]

        obs_seq = np.zeros((self.sequence_length, self.obs_dim), dtype=np.float32)
        actions_seq = np.zeros((self.sequence_length,), dtype=np.int64)
        rewards_seq = np.zeros((self.sequence_length,), dtype=np.float32)
        next_obs_seq = np.zeros((self.sequence_length, self.obs_dim), dtype=np.float32)
        dones_seq = np.zeros((self.sequence_length,), dtype=np.int64)
        mask = np.zeros((self.sequence_length,), dtype=np.float32)

        max_len = len(idxs) - start
        for t in range(self.sequence_length):
            if t >= max_len:
                break
            idx_t = idxs[start + t]
            obs_seq[t] = self.obs[idx_t]
            actions_seq[t] = self.actions[idx_t]
            rewards_seq[t] = self.rewards[idx_t]
            next_obs_seq[t] = self.next_obs[idx_t]
            dones_seq[t] = self.dones[idx_t]
            mask[t] = 1.0
            if self.dones[idx_t] == 1:
                break

        return obs_seq, actions_seq, rewards_seq, next_obs_seq, dones_seq, mask


# -----------------------------------------------------------------------------
# Actor-critic: TransitionDataset for single-transition batches
# -----------------------------------------------------------------------------
class TransitionDataset(Dataset):
    """Single transitions for actor-critic imagination init."""
    def __init__(self, path):
        data = np.load(path)
        self.obs = data["obs"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.rewards = data["rewards"].astype(np.float32)
        self.next_obs = data["next_obs"].astype(np.float32)
        self.dones = data["dones"].astype(np.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.next_obs[idx],
        )


# -----------------------------------------------------------------------------
# World model training
# -----------------------------------------------------------------------------
def kl_divergence(mean_q, logstd_q, mean_p, logstd_p):
    var_q = torch.exp(2 * logstd_q)
    var_p = torch.exp(2 * logstd_p)
    return 0.5 * (
        (var_q + (mean_q - mean_p) ** 2) / var_p
        - 1.0
        + 2 * (logstd_p - logstd_q)
    ).sum(-1)


def validate_world_model(world_model, val_dataloader, beta_kl=1.0, loss_weights=(1.0, 1.0, 1.0)):
    world_model.eval()
    obs_dim = world_model.rssm.obs_dim

    total_loss = 0.0
    total_obs_abs = 0.0
    total_obs_sq = 0.0
    total_reward_abs = 0.0
    total_reward_sq = 0.0
    total_reward_sign_correct = 0.0
    total_mask = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            obs_seq, actions_seq, rewards_seq, next_obs_seq, dones_seq, mask = batch
            obs_seq = obs_seq.to(DEVICE)
            actions_seq = actions_seq.to(DEVICE)
            rewards_seq = rewards_seq.to(DEVICE)
            mask = mask.to(DEVICE)

            batch_size, seq_len = obs_seq.shape[:2]
            action_dim = world_model.rssm.action_dim
            latent_dim = world_model.rssm.latent_dim

            h = world_model.rssm.init_hidden(batch_size, DEVICE)
            z_prev = torch.zeros(batch_size, latent_dim, device=DEVICE)
            a_prev = torch.zeros(batch_size, action_dim, device=DEVICE)

            sum_recon = 0.0
            sum_rew = 0.0
            sum_kl = 0.0

            for t in range(seq_len):
                h = world_model.rssm.update_hidden(h, z_prev, a_prev)
                mean_prior, logstd_prior = world_model.rssm.prior(h)
                mean_post, logstd_post = world_model.rssm.posterior(h, obs_seq[:, t])
                z_t = world_model.rssm.sample_latent(mean_post, logstd_post)

                obs_pred = world_model.reconstruct_obs(h, z_t)
                reward_pred = world_model.predict_reward(h, z_t)

                recon_loss = (obs_pred - obs_seq[:, t]).pow(2).sum(-1)
                reward_loss = (reward_pred - rewards_seq[:, t]).pow(2)
                kl = kl_divergence(mean_post, logstd_post, mean_prior, logstd_prior)

                mask_t = mask[:, t]
                sum_recon += (recon_loss * mask_t).sum()
                sum_rew += (reward_loss * mask_t).sum()
                sum_kl += (kl * mask_t).sum()

                obs_diff = (obs_pred - obs_seq[:, t]).abs().sum(-1)
                total_obs_abs += (obs_diff * mask_t).sum().item()
                total_obs_sq += ((obs_pred - obs_seq[:, t]).pow(2).sum(-1) * mask_t).sum().item()
                total_reward_abs += ((reward_pred - rewards_seq[:, t]).abs() * mask_t).sum().item()
                total_reward_sq += (((reward_pred - rewards_seq[:, t]).pow(2)) * mask_t).sum().item()

                true_reward_sign = (rewards_seq[:, t] > 0).float()
                pred_reward_sign = (reward_pred > 0).float()
                total_reward_sign_correct += ((true_reward_sign == pred_reward_sign).float() * mask_t).sum().item()

                total_mask += mask_t.sum().item()

                z_prev = z_t
                a_prev = torch.nn.functional.one_hot(actions_seq[:, t], num_classes=action_dim).float()

            batch_mask = mask.sum().item()
            if batch_mask > 0:
                total_loss += (
                    loss_weights[0] * sum_recon
                    + loss_weights[1] * sum_rew
                    + loss_weights[2] * beta_kl * sum_kl
                ).item()

    if total_mask == 0:
        return {
            'loss': 0.0,
            'obs_mae': 0.0,
            'obs_rmse': 0.0,
            'reward_mae': 0.0,
            'reward_rmse': 0.0,
            'reward_sign_acc': 0.0
        }

    obs_count = total_mask * obs_dim
    avg_loss = total_loss / total_mask
    avg_obs_mae = total_obs_abs / obs_count
    avg_obs_rmse = np.sqrt(total_obs_sq / obs_count)
    avg_reward_mae = total_reward_abs / total_mask
    avg_reward_rmse = np.sqrt(total_reward_sq / total_mask)
    avg_reward_sign_acc = total_reward_sign_correct / total_mask

    return {
        'loss': avg_loss,
        'obs_mae': avg_obs_mae,
        'obs_rmse': avg_obs_rmse,
        'reward_mae': avg_reward_mae,
        'reward_rmse': avg_reward_rmse,
        'reward_sign_acc': avg_reward_sign_acc
    }


def train_world_model(world_model, train_dataloader, val_dataloader, epochs=10, start_epoch=0,
                     checkpoint_freq=100, val_freq=10, lr=3e-4, beta_kl=1.0, loss_weights=(1.0, 1.0, 1.0),
                     log_path=None):
    world_model.train()
    opt = optim.AdamW(world_model.parameters(), lr=lr)

    if start_epoch >= epochs:
        log_message(f"Start epoch {start_epoch} is >= total epochs {epochs}; nothing to train.", log_path)
        return

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss = 0.0
        for batch in train_dataloader:
            obs_seq, actions_seq, rewards_seq, next_obs_seq, dones_seq, mask = batch
            obs_seq = obs_seq.to(DEVICE)
            actions_seq = actions_seq.to(DEVICE)
            rewards_seq = rewards_seq.to(DEVICE)
            mask = mask.to(DEVICE)

            batch_size, seq_len = obs_seq.shape[:2]
            action_dim = world_model.rssm.action_dim
            latent_dim = world_model.rssm.latent_dim

            h = world_model.rssm.init_hidden(batch_size, DEVICE)
            z_prev = torch.zeros(batch_size, latent_dim, device=DEVICE)
            a_prev = torch.zeros(batch_size, action_dim, device=DEVICE)

            sum_recon = 0.0
            sum_rew = 0.0
            sum_kl = 0.0
            total_mask = mask.sum().clamp_min(1.0)

            for t in range(seq_len):
                h = world_model.rssm.update_hidden(h, z_prev, a_prev)
                mean_prior, logstd_prior = world_model.rssm.prior(h)
                mean_post, logstd_post = world_model.rssm.posterior(h, obs_seq[:, t])
                z_t = world_model.rssm.sample_latent(mean_post, logstd_post)

                obs_pred = world_model.reconstruct_obs(h, z_t)
                reward_pred = world_model.predict_reward(h, z_t)

                recon_loss = (obs_pred - obs_seq[:, t]).pow(2).sum(-1)
                reward_loss = (reward_pred - rewards_seq[:, t]).pow(2)
                kl = kl_divergence(mean_post, logstd_post, mean_prior, logstd_prior)

                mask_t = mask[:, t]
                sum_recon += (recon_loss * mask_t).sum()
                sum_rew += (reward_loss * mask_t).sum()
                sum_kl += (kl * mask_t).sum()

                z_prev = z_t
                a_prev = torch.nn.functional.one_hot(actions_seq[:, t], num_classes=action_dim).float()

            loss = (
                loss_weights[0] * sum_recon
                + loss_weights[1] * sum_rew
                + loss_weights[2] * beta_kl * sum_kl
            ) / total_mask

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
            opt.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        if epoch % val_freq == 0:
            val_metrics = validate_world_model(world_model, val_dataloader, beta_kl, loss_weights)
            log_message(f"[WorldModel] Epoch {epoch}, train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}", log_path)
            log_message(f"  Validation Metrics - Obs MAE: {val_metrics['obs_mae']:.4f}, Obs RMSE: {val_metrics['obs_rmse']:.4f}", log_path)
            log_message(f"  Reward MAE: {val_metrics['reward_mae']:.4f}, Reward RMSE: {val_metrics['reward_rmse']:.4f}, Reward Sign Acc: {val_metrics['reward_sign_acc']:.3f}", log_path)
        else:
            log_message(f"[WorldModel] Epoch {epoch}, train_loss={train_loss:.4f}", log_path)

        if epoch % checkpoint_freq == 0:
            save_checkpoint_with_timestamp(world_model, "world_model", epoch, log_path=log_path)


# -----------------------------------------------------------------------------
# Actor-critic: auxiliary rewards and imagination
# -----------------------------------------------------------------------------
def compute_auxiliary_rewards(observations, upright_weight=0.1, downward_speed_weight=0.05):
    """Auxiliary rewards for upright pose and minimal downward speed."""
    angles = observations[..., 4]
    y_velocities = observations[..., 3]
    upright_reward = upright_weight * torch.exp(-torch.abs(angles))
    downward_penalty = downward_speed_weight * torch.clamp(y_velocities, max=0.0)
    return upright_reward + downward_penalty


def imagine_rollout(world_model, actor, start_obs, horizon=15):
    """Imagine latent rollouts from start_obs, returning zs, rewards, actions, observations."""
    world_model.eval()
    actor.eval()
    with torch.no_grad():
        batch_size = start_obs.size(0)
        action_dim = world_model.rssm.action_dim
        latent_dim = world_model.rssm.latent_dim
        h = world_model.rssm.init_hidden(batch_size, DEVICE)
        z_prev = torch.zeros(batch_size, latent_dim, device=DEVICE)
        a_prev = torch.zeros(batch_size, action_dim, device=DEVICE)
        h = world_model.rssm.update_hidden(h, z_prev, a_prev)
        mean_post, logstd_post = world_model.rssm.posterior(h, start_obs)
        z = world_model.rssm.sample_latent(mean_post, logstd_post)

    zs = []
    rewards = []
    actions = []
    observations = []

    for t in range(horizon):
        dist = actor(z)
        a = dist.sample()
        a_onehot = torch.nn.functional.one_hot(a, num_classes=action_dim).float()

        h = world_model.rssm.update_hidden(h, z, a_onehot)
        mean_prior, logstd_prior = world_model.rssm.prior(h)
        z = world_model.rssm.sample_latent(mean_prior, logstd_prior)
        r = world_model.predict_reward(h, z)
        obs_imagined = world_model.reconstruct_obs(h, z)

        zs.append(z)
        rewards.append(r)
        actions.append(a)
        observations.append(obs_imagined)

    zs = torch.stack(zs, dim=1)
    rewards = torch.stack(rewards, dim=1)
    actions = torch.stack(actions, dim=1)
    observations = torch.stack(observations, dim=1)
    return zs, rewards, actions, observations


def train_actor_critic(world_model, actor, critic, dataloader,
                       epochs=10, lr=3e-4, gamma=0.99, lambda_gae=0.95,
                       imagination_horizon=15, loss_weights=(1.0, 1.0), entropy_coeff=0.01,
                       checkpoint_freq=100, aux_rewards_config=None, start_epoch=0, log_path=None):
    actor.train()
    critic.train()
    opt_actor = optim.AdamW(actor.parameters(), lr=lr)
    opt_critic = optim.AdamW(critic.parameters(), lr=lr)

    aux_rewards_config = aux_rewards_config or {}

    for epoch in range(start_epoch + 1, epochs + 1):
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_imagined_reward = 0.0
        total_value_mae = 0.0
        total_entropy = 0.0
        total_actor_grad_norm = 0.0
        total_critic_grad_norm = 0.0
        total_aux_upright = 0.0
        total_aux_downward = 0.0
        num_batches = 0

        for batch in dataloader:
            obs, actions, rewards, dones, next_obs = batch
            obs = obs.to(DEVICE)

            zs, imagined_rewards, imagined_actions, imagined_observations = imagine_rollout(
                world_model, actor, obs, horizon=imagination_horizon
            )

            zs = zs.detach()
            imagined_rewards = imagined_rewards.detach()
            imagined_actions = imagined_actions.detach()
            imagined_observations = imagined_observations.detach()

            aux_rewards = compute_auxiliary_rewards(
                imagined_observations,
                upright_weight=aux_rewards_config.get('upright_pose_weight', 0.1),
                downward_speed_weight=aux_rewards_config.get('downward_speed_weight', 0.05)
            )
            imagined_rewards = imagined_rewards + aux_rewards

            with torch.no_grad():
                values = critic(zs)
                next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, -1:])], dim=1)

                deltas = imagined_rewards + gamma * next_values - values
                adv = torch.zeros_like(deltas)
                gae = torch.zeros_like(deltas[:, 0])
                for t in reversed(range(imagination_horizon)):
                    gae = deltas[:, t] + gamma * lambda_gae * gae
                    adv[:, t] = gae
                returns = adv + values

            dist = actor(zs.reshape(-1, zs.size(-1)))
            log_probs = dist.log_prob(imagined_actions.reshape(-1))
            adv_flat = adv.reshape(-1).detach()
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

            actor_loss = loss_weights[0] * (-(log_probs * adv_flat).mean() - entropy_coeff * dist.entropy().mean())

            value_pred = critic(zs.reshape(-1, zs.size(-1)))
            critic_loss = loss_weights[1] * (value_pred - returns.reshape(-1).detach()).pow(2).mean()

            opt_actor.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in actor.parameters() if p.grad is not None]))
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
            opt_actor.step()

            opt_critic.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in critic.parameters() if p.grad is not None]))
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
            opt_critic.step()

            imagined_reward_mean = imagined_rewards.mean().item()
            value_mae = torch.abs(value_pred - returns.reshape(-1).detach()).mean().item()
            entropy = dist.entropy().mean().item()
            angles = imagined_observations[..., 4]
            y_velocities = imagined_observations[..., 3]
            upright_reward = aux_rewards_config.get('upright_pose_weight', 0.1) * torch.exp(-torch.abs(angles)).mean().item()
            downward_penalty = aux_rewards_config.get('downward_speed_weight', 0.05) * torch.clamp(y_velocities, max=0.0).mean().item()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_imagined_reward += imagined_reward_mean
            total_value_mae += value_mae
            total_entropy += entropy
            total_actor_grad_norm += actor_grad_norm.item()
            total_critic_grad_norm += critic_grad_norm.item()
            total_aux_upright += upright_reward
            total_aux_downward += downward_penalty
            num_batches += 1

        avg_actor_loss = total_actor_loss / num_batches
        avg_critic_loss = total_critic_loss / num_batches
        avg_imagined_reward = total_imagined_reward / num_batches
        avg_value_mae = total_value_mae / num_batches
        avg_entropy = total_entropy / num_batches
        avg_actor_grad_norm = total_actor_grad_norm / num_batches
        avg_critic_grad_norm = total_critic_grad_norm / num_batches
        avg_aux_upright = total_aux_upright / num_batches
        avg_aux_downward = total_aux_downward / num_batches

        log_message(f"[ActorCritic] Epoch {epoch}, actor_loss={avg_actor_loss:.4f}, critic_loss={avg_critic_loss:.4f}", log_path)
        log_message(f"  Metrics - Imagined Reward: {avg_imagined_reward:.4f}, Value MAE: {avg_value_mae:.4f}, Entropy: {avg_entropy:.4f}", log_path)
        log_message(f"  Grad Norms - Actor: {avg_actor_grad_norm:.4f}, Critic: {avg_critic_grad_norm:.4f}", log_path)
        log_message(f"  Aux Rewards - Upright: {avg_aux_upright:.4f}, Downward Penalty: {avg_aux_downward:.4f}", log_path)

        if epoch % checkpoint_freq == 0:
            save_checkpoint_with_timestamp(actor, "actor", epoch, log_path=log_path)
            save_checkpoint_with_timestamp(critic, "critic", epoch, log_path=log_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Dreamer offline on LunarLander")
    parser.add_argument('--phase', choices=['world_model', 'actor_critic'], required=True,
                        help='Which phase to train: world_model or actor_critic')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--train_dataset', type=str, default='lunarlander_train_dataset.npz',
                        help='Path to training dataset')
    parser.add_argument('--val_dataset', type=str, default='lunarlander_val_dataset.npz',
                        help='Path to validation dataset (world model only)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    phase_config = config.get(args.phase, {})

    if args.phase == 'world_model':
        log_path = os.path.join(".", "train_worldmodel_logs.txt")

        sequence_length = phase_config.get('sequence_length', 50)
        action_dim = phase_config.get('action_dim', 4)
        train_dataset = SequenceDataset(args.train_dataset, sequence_length, action_dim, random_start=True)
        val_dataset = SequenceDataset(args.val_dataset, sequence_length, action_dim, random_start=False)

        batch_size = phase_config.get('batch_size', 64)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      generator=get_dataloader_generator(args.seed))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        obs_dim = train_dataset.obs_dim
        capacity = phase_config.get('capacity', {})
        latent_dim = capacity.get('latent_dim', 64)
        hidden_dim = capacity.get('hidden_dim', 128)

        world_model = WorldModel(obs_dim, action_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(DEVICE)

        latest_checkpoint = get_latest_checkpoint("world_model")
        start_epoch = 0
        if latest_checkpoint:
            world_model.load_state_dict(torch.load(latest_checkpoint, map_location=DEVICE))
            log_message(f"Loaded world model checkpoint: {latest_checkpoint}", log_path)
            try:
                epoch_str = latest_checkpoint.split('_epoch_')[-1].split('.')[0]
                start_epoch = int(epoch_str)
            except ValueError:
                pass
        elif os.path.exists("world_model.pt"):
            world_model.load_state_dict(torch.load("world_model.pt", map_location=DEVICE))
            log_message("Loaded world model from world_model.pt", log_path)

        epochs = phase_config.get('epochs', 200)
        lr = phase_config.get('lr', 3e-4)
        beta_kl = phase_config.get('beta_kl', 1.0)
        val_freq = phase_config.get('val_freq', 10)
        checkpoint_freq = phase_config.get('checkpoint_freq', 100)

        loss_weights = phase_config.get('loss_weights', {})
        recon_weight = loss_weights.get('reconstruction', 1.0)
        reward_weight = loss_weights.get('reward', 1.0)
        kl_weight = loss_weights.get('kl', 1.0)

        log_message(f"Training world model for {epochs} epochs with lr={lr}, beta_kl={beta_kl}, sequence_length={sequence_length}, seed={args.seed}...", log_path)
        log_message(f"Model capacity: latent_dim={latent_dim}, hidden_dim={hidden_dim}", log_path)
        log_message(f"Loss weights: recon={recon_weight}, reward={reward_weight}, kl={kl_weight}", log_path)

        train_world_model(world_model, train_dataloader, val_dataloader,
                         epochs=epochs, start_epoch=start_epoch, checkpoint_freq=checkpoint_freq,
                         val_freq=val_freq, lr=lr, beta_kl=beta_kl,
                         loss_weights=(recon_weight, reward_weight, kl_weight),
                         log_path=log_path)

        torch.save(world_model.state_dict(), "world_model.pt")
        log_message("Final world model saved to world_model.pt", log_path)

    elif args.phase == 'actor_critic':
        log_path = os.path.join(".", "train_actorcritic_logs.txt")

        dataset = TransitionDataset(args.train_dataset)
        batch_size = phase_config.get('batch_size', 64)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                generator=get_dataloader_generator(args.seed))

        obs_dim = dataset.obs.shape[1]
        action_dim = phase_config.get('action_dim', 4)
        capacity = phase_config.get('capacity', {})
        latent_dim = capacity.get('latent_dim', 64)
        hidden_dim = capacity.get('hidden_dim', 128)

        world_model = WorldModel(obs_dim, action_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(DEVICE)
        latest_wm = get_latest_checkpoint("world_model")
        if latest_wm:
            world_model.load_state_dict(torch.load(latest_wm, map_location=DEVICE))
            log_message(f"Loaded world model: {latest_wm}", log_path)
        elif os.path.exists("world_model.pt"):
            world_model.load_state_dict(torch.load("world_model.pt", map_location=DEVICE))
            log_message("Loaded world model from world_model.pt", log_path)
        else:
            raise FileNotFoundError("No world model checkpoint found.")
        world_model.eval()

        actor = Actor(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(DEVICE)
        critic = Critic(latent_dim=latent_dim, hidden_dim=hidden_dim).to(DEVICE)

        latest_actor = get_latest_checkpoint("actor")
        start_epoch = 0
        if latest_actor:
            actor.load_state_dict(torch.load(latest_actor, map_location=DEVICE))
            log_message(f"Loaded actor checkpoint: {latest_actor}", log_path)
            try:
                epoch_str = latest_actor.split('_epoch_')[-1].split('.')[0]
                start_epoch = int(epoch_str)
            except ValueError:
                pass
        elif os.path.exists("actor.pt"):
            actor.load_state_dict(torch.load("actor.pt", map_location=DEVICE))
            log_message("Loaded actor from actor.pt", log_path)

        latest_critic = get_latest_checkpoint("critic")
        if latest_critic:
            critic.load_state_dict(torch.load(latest_critic, map_location=DEVICE))
            log_message(f"Loaded critic checkpoint: {latest_critic}", log_path)
        elif os.path.exists("critic.pt"):
            critic.load_state_dict(torch.load("critic.pt", map_location=DEVICE))
            log_message("Loaded critic from critic.pt", log_path)

        epochs = phase_config.get('epochs', 200)
        lr = phase_config.get('lr', 3e-4)
        imagination_horizon = phase_config.get('imagination_horizon', 15)
        checkpoint_freq = phase_config.get('checkpoint_freq', 100)

        loss_weights = phase_config.get('loss_weights', {})
        actor_weight = loss_weights.get('actor', 1.0)
        critic_weight = loss_weights.get('critic', 0.25)
        entropy_coeff = loss_weights.get('entropy', 0.01)
        aux_rewards_config = phase_config.get('auxiliary_rewards', {})

        log_message(f"Training actor-critic for {epochs} epochs with lr={lr}, imagination_horizon={imagination_horizon}, seed={args.seed}...", log_path)
        log_message(f"Model capacity: latent_dim={latent_dim}, hidden_dim={hidden_dim}", log_path)

        train_actor_critic(world_model, actor, critic, dataloader,
                          epochs=epochs, lr=lr, imagination_horizon=imagination_horizon,
                          loss_weights=(actor_weight, critic_weight), entropy_coeff=entropy_coeff,
                          checkpoint_freq=checkpoint_freq, aux_rewards_config=aux_rewards_config,
                          start_epoch=start_epoch, log_path=log_path)

        torch.save(actor.state_dict(), "actor.pt")
        torch.save(critic.state_dict(), "critic.pt")
        log_message("Final actor and critic saved to actor.pt and critic.pt", log_path)


if __name__ == "__main__":
    main()
