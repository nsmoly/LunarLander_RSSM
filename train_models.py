# train_models.py - World model and actor-critic training
import argparse
import os
import random
import yaml
import datetime
import glob
import numpy as np
import torch
import torch.nn.functional as F
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
# SequenceDataset for sequence-based training
# -----------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """Samples sequences from real episodes for world model training."""
    def __init__(self, path, sequence_length, action_dim, random_start=True, dataset_seq_offset=20):
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
        self.dataset_seq_offset = max(1, int(dataset_seq_offset))
        self.obs_dim = self.obs.shape[1]
        self.action_dim = int(action_dim) # need to pass action_dim since we only store action IDs in the dataset
        if self.actions.size and self.actions.max() >= self.action_dim:
            raise ValueError(
                f"Dataset actions exceed action_dim={self.action_dim}; "
                f"max action={int(self.actions.max())}"
            )

        # The dataset is stored as a set of episodes, each with a set of steps.
        # We need to store the indices of the episodes.
        # We also need to sort the steps within each episode and store the indices of the steps.
        # We use this to sample sequences from the dataset.
        self.episode_ids = np.unique(self.ep_index)
        self.episode_indices = []
        self.start_positions = []
        for ep_id in self.episode_ids:
            data_idxs = np.where(self.ep_index == ep_id)[0]
            if data_idxs.size == 0:
                continue
            order = np.argsort(self.step_index[data_idxs], kind="stable")
            data_idxs = data_idxs[order]
            ep_pos = len(self.episode_indices)
            self.episode_indices.append(data_idxs)
            for pos in range(0, len(data_idxs), self.dataset_seq_offset):
                self.start_positions.append((ep_pos, pos))

    def __len__(self):
        return len(self.start_positions)

    def __getitem__(self, idx):
        if self.random_start:
            ep_pos, start = self.start_positions[np.random.randint(len(self.start_positions))]
        else:
            ep_pos, start = self.start_positions[idx]
        data_idxs = self.episode_indices[ep_pos]

        obs_seq = np.zeros((self.sequence_length, self.obs_dim), dtype=np.float32)
        actions_seq = np.zeros((self.sequence_length,), dtype=np.int64)
        rewards_seq = np.zeros((self.sequence_length,), dtype=np.float32)
        next_obs_seq = np.zeros((self.sequence_length, self.obs_dim), dtype=np.float32)
        dones_seq = np.zeros((self.sequence_length,), dtype=np.int64)
        mask = np.zeros((self.sequence_length,), dtype=np.float32)

        max_len = len(data_idxs) - start
        for t in range(self.sequence_length):
            if t >= max_len:
                break
            index_t = data_idxs[start + t]
            obs_seq[t] = self.obs[index_t]
            actions_seq[t] = self.actions[index_t]
            rewards_seq[t] = self.rewards[index_t]
            next_obs_seq[t] = self.next_obs[index_t]
            dones_seq[t] = self.dones[index_t]
            mask[t] = 1.0   # mask is needed to support variable-length sequences
            if self.dones[index_t] == 1:
                break

        return obs_seq, actions_seq, rewards_seq, next_obs_seq, dones_seq, mask


# -----------------------------------------------------------------------------
# ActorCriticWarmupDataset for episode sampling (their starts) for world model warm-up
# -----------------------------------------------------------------------------
class ActorCriticWarmupDataset(Dataset):
    """Samples (episode, start) where start + horizon <= first_done. Returns obs_past, actions_past for warm-up."""
    def __init__(self, path, horizon, past_horizon, future_horizon, action_dim):
        assert past_horizon + future_horizon == horizon, (
            f"past_horizon + future_horizon must equal horizon; got P={past_horizon} + F={future_horizon} != H={horizon}"
        )
        data = np.load(path)
        self.obs = data["obs"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.dones = data["dones"].astype(np.int64)
        self.ep_index = data["ep_index"].astype(np.int64)
        self.step_index = data["step_index"].astype(np.int64)
        self.horizon = int(horizon)
        self.past_horizon = int(past_horizon)
        self.future_horizon = int(future_horizon)
        self.obs_dim = self.obs.shape[1]
        self.action_dim = int(action_dim)

        self.episode_ids = np.unique(self.ep_index)
        self.start_positions = []  # (ep_pos, start_step)
        for ep_id in self.episode_ids:
            data_idxs = np.where(self.ep_index == ep_id)[0]
            if data_idxs.size == 0:
                continue
            order = np.argsort(self.step_index[data_idxs], kind="stable")
            data_idxs = data_idxs[order]
            first_done = np.where(self.dones[data_idxs] == 1)[0]
            first_done_idx = int(first_done[0]) if first_done.size > 0 else len(data_idxs)
            max_start = first_done_idx - self.horizon
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                self.start_positions.append((data_idxs, start))

    def __len__(self):
        return len(self.start_positions)

    def __getitem__(self, idx):
        data_idxs, start = self.start_positions[idx]
        P = self.past_horizon
        obs_past = self.obs[data_idxs[start : start + P]].astype(np.float32)  # (P, obs_dim)
        actions_past = self.actions[data_idxs[start : start + P]].astype(np.int64)  # (P,)
        return obs_past, actions_past


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


def train_world_model(world_model, train_dataloader, val_dataloader, epochs=10, start_epoch=0,
                     checkpoint_freq=100, val_freq=10, lr=3e-4, beta_kl=1.0, loss_weights=(1.0, 1.0, 1.0, 0.5),
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
            dones_seq = dones_seq.to(DEVICE).float()
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
            sum_done = 0.0
            total_mask = mask.sum().clamp_min(1.0)

            for t in range(seq_len):
                h = world_model.rssm.update_hidden(h, z_prev, a_prev)
                mean_prior, logstd_prior = world_model.rssm.prior(h)
                mean_post, logstd_post = world_model.rssm.posterior(h, obs_seq[:, t])
                z_t = world_model.rssm.sample_latent(mean_post, logstd_post)

                obs_pred = world_model.reconstruct_obs(h, z_t)
                reward_pred = world_model.predict_reward(h, z_t)
                done_logits = world_model.predict_done_logits(h, z_t)

                recon_loss = (obs_pred - obs_seq[:, t]).pow(2).sum(-1)
                reward_loss = (reward_pred - rewards_seq[:, t]).pow(2)
                done_loss = F.binary_cross_entropy_with_logits(done_logits, dones_seq[:, t], reduction="none")
                kl = kl_divergence(mean_post, logstd_post, mean_prior, logstd_prior)

                mask_t = mask[:, t]
                sum_recon += (recon_loss * mask_t).sum()
                sum_rew += (reward_loss * mask_t).sum()
                sum_done += (done_loss * mask_t).sum()
                sum_kl += (kl * mask_t).sum()

                z_prev = z_t
                a_prev = torch.nn.functional.one_hot(actions_seq[:, t], num_classes=action_dim).float()

            loss = (
                loss_weights[0] * sum_recon
                + loss_weights[1] * sum_rew
                + loss_weights[3] * sum_done
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
            log_message(f"  Done Acc: {val_metrics['done_acc']:.3f}", log_path)
        else:
            log_message(f"[WorldModel] Epoch {epoch}, train_loss={train_loss:.4f}", log_path)

        if epoch % checkpoint_freq == 0:
            save_checkpoint_with_timestamp(world_model, "world_model", epoch, log_path=log_path)


def validate_world_model(world_model, val_dataloader, beta_kl=1.0, loss_weights=(1.0, 1.0, 1.0, 0.5)):
    
    world_model.eval()
    obs_dim = world_model.rssm.obs_dim

    total_loss = 0.0
    total_obs_abs = 0.0
    total_obs_sq = 0.0
    total_reward_abs = 0.0
    total_reward_sq = 0.0
    total_reward_sign_correct = 0.0
    total_done_correct = 0.0
    total_mask = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            obs_seq, actions_seq, rewards_seq, next_obs_seq, dones_seq, mask = batch
            obs_seq = obs_seq.to(DEVICE)
            actions_seq = actions_seq.to(DEVICE)
            rewards_seq = rewards_seq.to(DEVICE)
            dones_seq = dones_seq.to(DEVICE).float()
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
            sum_done = 0.0

            for t in range(seq_len):
                h = world_model.rssm.update_hidden(h, z_prev, a_prev)
                mean_prior, logstd_prior = world_model.rssm.prior(h)
                mean_post, logstd_post = world_model.rssm.posterior(h, obs_seq[:, t])
                z_t = world_model.rssm.sample_latent(mean_post, logstd_post)

                obs_pred = world_model.reconstruct_obs(h, z_t)
                reward_pred = world_model.predict_reward(h, z_t)
                done_logits = world_model.predict_done_logits(h, z_t)

                recon_loss = (obs_pred - obs_seq[:, t]).pow(2).sum(-1)
                reward_loss = (reward_pred - rewards_seq[:, t]).pow(2)
                done_loss = F.binary_cross_entropy_with_logits(done_logits, dones_seq[:, t], reduction="none")
                kl = kl_divergence(mean_post, logstd_post, mean_prior, logstd_prior)

                mask_t = mask[:, t]
                sum_recon += (recon_loss * mask_t).sum()
                sum_rew += (reward_loss * mask_t).sum()
                sum_done += (done_loss * mask_t).sum()
                sum_kl += (kl * mask_t).sum()

                obs_diff = (obs_pred - obs_seq[:, t]).abs().sum(-1)
                total_obs_abs += (obs_diff * mask_t).sum().item()
                total_obs_sq += ((obs_pred - obs_seq[:, t]).pow(2).sum(-1) * mask_t).sum().item()
                total_reward_abs += ((reward_pred - rewards_seq[:, t]).abs() * mask_t).sum().item()
                total_reward_sq += (((reward_pred - rewards_seq[:, t]).pow(2)) * mask_t).sum().item()

                true_reward_sign = (rewards_seq[:, t] > 0).float()
                pred_reward_sign = (reward_pred > 0).float()
                total_reward_sign_correct += ((true_reward_sign == pred_reward_sign).float() * mask_t).sum().item()
                done_pred = (torch.sigmoid(done_logits) >= 0.5).float()
                total_done_correct += ((done_pred == dones_seq[:, t]).float() * mask_t).sum().item()

                total_mask += mask_t.sum().item()

                z_prev = z_t
                a_prev = torch.nn.functional.one_hot(actions_seq[:, t], num_classes=action_dim).float()

            batch_mask = mask.sum().item()
            if batch_mask > 0:
                total_loss += (
                    loss_weights[0] * sum_recon
                    + loss_weights[1] * sum_rew
                    + loss_weights[3] * sum_done
                    + loss_weights[2] * beta_kl * sum_kl
                ).item()

    if total_mask == 0:
        return {
            'loss': 0.0,
            'obs_mae': 0.0,
            'obs_rmse': 0.0,
            'reward_mae': 0.0,
            'reward_rmse': 0.0,
            'reward_sign_acc': 0.0,
            'done_acc': 0.0,
        }

    obs_count = total_mask * obs_dim
    avg_loss = total_loss / total_mask
    avg_obs_mae = total_obs_abs / obs_count
    avg_obs_rmse = np.sqrt(total_obs_sq / obs_count)
    avg_reward_mae = total_reward_abs / total_mask
    avg_reward_rmse = np.sqrt(total_reward_sq / total_mask)
    avg_reward_sign_acc = total_reward_sign_correct / total_mask
    avg_done_acc = total_done_correct / total_mask

    return {
        'loss': avg_loss,
        'obs_mae': avg_obs_mae,
        'obs_rmse': avg_obs_rmse,
        'reward_mae': avg_reward_mae,
        'reward_rmse': avg_reward_rmse,
        'reward_sign_acc': avg_reward_sign_acc,
        'done_acc': avg_done_acc,
    }


# -----------------------------------------------------------------------------
# Actor-critic: imagination
# -----------------------------------------------------------------------------
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
        action_distr = actor(z)
        a = action_distr.sample()
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


def imagine_rollout_with_warmup(world_model, actor, obs_past, actions_past, future_horizon):
    """Warm-up with real (obs, action) for P steps, then imagine F steps. Returns zs, rewards, actions, observations."""
    world_model.eval()
    actor.eval()
    batch_size = obs_past.size(0)
    action_dim = world_model.rssm.action_dim
    latent_dim = world_model.rssm.latent_dim

    with torch.no_grad():
        h = world_model.rssm.init_hidden(batch_size, DEVICE)
        z_prev = torch.zeros(batch_size, latent_dim, device=DEVICE)
        a_prev = torch.zeros(batch_size, action_dim, device=DEVICE)

        for t in range(obs_past.size(1)):
            h = world_model.rssm.update_hidden(h, z_prev, a_prev)
            mean_post, logstd_post = world_model.rssm.posterior(h, obs_past[:, t])
            z_prev = world_model.rssm.sample_latent(mean_post, logstd_post)
            a_prev = torch.nn.functional.one_hot(actions_past[:, t], num_classes=action_dim).float()

        z = z_prev

    zs = []
    rewards = []
    actions = []
    observations = []

    for t in range(future_horizon):
        action_distr = actor(z)
        a = action_distr.sample()
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
                       future_horizon=15, loss_weights=(1.0, 1.0), entropy_coeff=0.01,
                       entropy_coeff_end=None,
                       checkpoint_freq=100, start_epoch=0, log_path=None,
                       use_warmup=True, advantage_clip=3.0, actor_grad_clip=10.0,
                       collapse_entropy_threshold=0.2, collapse_actor_grad_threshold=1e-3,
                       collapse_max_action_prob_threshold=0.98, collapse_patience_epochs=3,
                       low_entropy_actor_lr_threshold=0.9,
                       reduced_actor_lr=None):
    actor.train()
    critic.train()
    opt_actor = optim.AdamW(actor.parameters(), lr=lr)
    opt_critic = optim.AdamW(critic.parameters(), lr=lr)

    if entropy_coeff_end is None:
        entropy_coeff_end = entropy_coeff

    def _grad_norm(parameters):
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=DEVICE)
        return torch.norm(torch.stack([torch.norm(g) for g in grads]))

    low_entropy_streak = 0
    low_actor_grad_streak = 0
    high_action_conf_streak = 0
    actor_lr_reduced = False

    for epoch in range(start_epoch + 1, epochs + 1):
        # Linear entropy decay across the configured actor-critic epochs.
        if epochs <= 1:
            current_entropy_coeff = entropy_coeff_end
        else:
            progress = (epoch - 1) / float(epochs - 1)
            current_entropy_coeff = entropy_coeff + (entropy_coeff_end - entropy_coeff) * progress

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_imagined_reward = 0.0
        total_value_mae = 0.0
        total_entropy = 0.0
        total_max_action_prob = 0.0
        total_actor_grad_norm = 0.0
        total_critic_grad_norm = 0.0
        num_batches = 0

        for batch in dataloader:
            if use_warmup:
                obs_past, actions_past = batch
                obs_past = obs_past.to(DEVICE)
                actions_past = actions_past.to(DEVICE)
                zs, imagined_rewards, imagined_actions, imagined_observations = imagine_rollout_with_warmup(
                    world_model, actor, obs_past, actions_past, future_horizon=future_horizon
                )
            else:
                obs, actions, rewards, dones, next_obs = batch
                obs = obs.to(DEVICE)
                zs, imagined_rewards, imagined_actions, imagined_observations = imagine_rollout(
                    world_model, actor, obs, horizon=future_horizon
                )

            zs = zs.detach()
            imagined_rewards = imagined_rewards.detach()
            imagined_actions = imagined_actions.detach()
            imagined_observations = imagined_observations.detach()

            with torch.no_grad():
                values = critic(zs)
                next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, -1:])], dim=1)

                deltas = imagined_rewards + gamma * next_values - values
                adv = torch.zeros_like(deltas)
                gae = torch.zeros_like(deltas[:, 0])
                for t in reversed(range(future_horizon)):
                    gae = deltas[:, t] + gamma * lambda_gae * gae
                    adv[:, t] = gae
                returns = adv + values

            action_distr = actor(zs.reshape(-1, zs.size(-1)))
            log_probs = action_distr.log_prob(imagined_actions.reshape(-1))
            adv_flat = adv.reshape(-1).detach()
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            adv_flat = torch.clamp(adv_flat, -advantage_clip, advantage_clip)

            actor_loss = loss_weights[0] * (
                -(log_probs * adv_flat).mean() - current_entropy_coeff * action_distr.entropy().mean()
            )

            value_pred = critic(zs.reshape(-1, zs.size(-1)))
            critic_loss = loss_weights[1] * (value_pred - returns.reshape(-1).detach()).pow(2).mean()

            opt_actor.zero_grad()
            actor_loss.backward()
            actor_grad_norm = _grad_norm(actor.parameters())
            torch.nn.utils.clip_grad_norm_(actor.parameters(), actor_grad_clip)
            opt_actor.step()

            opt_critic.zero_grad()
            critic_loss.backward()
            critic_grad_norm = _grad_norm(critic.parameters())
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
            opt_critic.step()

            imagined_reward_mean = imagined_rewards.mean().item()
            value_mae = torch.abs(value_pred - returns.reshape(-1).detach()).mean().item()
            entropy = action_distr.entropy().mean().item()
            max_action_prob = action_distr.probs.max(dim=-1).values.mean().item()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_imagined_reward += imagined_reward_mean
            total_value_mae += value_mae
            total_entropy += entropy
            total_max_action_prob += max_action_prob
            total_actor_grad_norm += actor_grad_norm.item()
            total_critic_grad_norm += critic_grad_norm.item()
            num_batches += 1

        avg_actor_loss = total_actor_loss / num_batches
        avg_critic_loss = total_critic_loss / num_batches
        avg_imagined_reward = total_imagined_reward / num_batches
        avg_value_mae = total_value_mae / num_batches
        avg_entropy = total_entropy / num_batches
        avg_max_action_prob = total_max_action_prob / num_batches
        avg_actor_grad_norm = total_actor_grad_norm / num_batches
        avg_critic_grad_norm = total_critic_grad_norm / num_batches

        log_message(f"[ActorCritic] Epoch {epoch}, actor_loss={avg_actor_loss:.4f}, critic_loss={avg_critic_loss:.4f}", log_path)
        log_message(
            f"  Metrics - Imagined Reward: {avg_imagined_reward:.4f}, Value MAE: {avg_value_mae:.4f}, "
            f"Entropy: {avg_entropy:.4f}, Max Action Prob: {avg_max_action_prob:.4f}, "
            f"Entropy Coef: {current_entropy_coeff:.4f}",
            log_path,
        )
        log_message(f"  Grad Norms - Actor: {avg_actor_grad_norm:.4f}, Critic: {avg_critic_grad_norm:.4f}", log_path)

        # One-way actor LR reduction when policy entropy gets too low.
        if (
            reduced_actor_lr is not None
            and not actor_lr_reduced
            and avg_entropy < low_entropy_actor_lr_threshold
            and reduced_actor_lr < opt_actor.param_groups[0]["lr"]
        ):
            old_lr = opt_actor.param_groups[0]["lr"]
            for pg in opt_actor.param_groups:
                pg["lr"] = reduced_actor_lr
            actor_lr_reduced = True
            log_message(
                f"  [LR Guard] Reduced actor LR from {old_lr:.2e} to {reduced_actor_lr:.2e} "
                f"(entropy={avg_entropy:.4f} < {low_entropy_actor_lr_threshold:.4f})",
                log_path,
            )

        # Early-stop guard against policy collapse.
        low_entropy_streak = low_entropy_streak + 1 if avg_entropy < collapse_entropy_threshold else 0
        low_actor_grad_streak = (
            low_actor_grad_streak + 1 if avg_actor_grad_norm < collapse_actor_grad_threshold else 0
        )
        high_action_conf_streak = (
            high_action_conf_streak + 1
            if avg_max_action_prob > collapse_max_action_prob_threshold
            else 0
        )
        if (
            low_entropy_streak >= collapse_patience_epochs
            or low_actor_grad_streak >= collapse_patience_epochs
            or high_action_conf_streak >= collapse_patience_epochs
        ):
            reason = (
                f"entropy<{collapse_entropy_threshold:.4f} for {low_entropy_streak} epochs"
                if low_entropy_streak >= collapse_patience_epochs
                else (
                    f"actor_grad<{collapse_actor_grad_threshold:.4f} for {low_actor_grad_streak} epochs"
                    if low_actor_grad_streak >= collapse_patience_epochs
                    else (
                        f"max_action_prob>{collapse_max_action_prob_threshold:.4f} "
                        f"for {high_action_conf_streak} epochs"
                    )
                )
            )
            log_message(
                f"[ActorCritic] Early stopping at epoch {epoch} due to collapse guard ({reason}).",
                log_path,
            )
            save_checkpoint_with_timestamp(actor, "actor", epoch, log_path=log_path)
            save_checkpoint_with_timestamp(critic, "critic", epoch, log_path=log_path)
            break

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
        dataset_seq_offset = phase_config.get('dataset_seq_offset', 10)
        action_dim = phase_config.get('action_dim', 4)
        train_dataset = SequenceDataset(
            args.train_dataset,
            sequence_length,
            action_dim,
            random_start=True,
            dataset_seq_offset=dataset_seq_offset,
        )
        val_dataset = SequenceDataset(
            args.val_dataset,
            sequence_length,
            action_dim,
            random_start=False,
            dataset_seq_offset=dataset_seq_offset,
        )

        batch_size = phase_config.get('batch_size', 64)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      generator=get_dataloader_generator(args.seed))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        obs_dim = train_dataset.obs_dim
        capacity = phase_config.get('capacity', {})
        latent_dim = capacity.get('latent_dim', 64)
        hidden_dim = capacity.get('hidden_dim', 128)
        gru_num_layers = int(capacity.get('gru_num_layers', 1))

        world_model = WorldModel(
            obs_dim,
            action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            gru_num_layers=gru_num_layers,
        ).to(DEVICE)

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
        done_weight = loss_weights.get('done', 0.5)

        log_message(f"Train dataset file: {args.train_dataset}", log_path)
        log_message(f"Validation dataset file: {args.val_dataset}", log_path)
        log_message(f"Train dataset episodes: {len(train_dataset.episode_ids)}", log_path)
        log_message(f"Train dataset samples/sequences: {len(train_dataset)}", log_path)
        log_message(f"Validation dataset episodes: {len(val_dataset.episode_ids)}", log_path)
        log_message(f"Validation dataset samples/sequences: {len(val_dataset)}", log_path)

        log_message(
            f"Training world model for {epochs} epochs with lr={lr}, beta_kl={beta_kl}, "
            f"sequence_length={sequence_length}, dataset_seq_offset={dataset_seq_offset}, seed={args.seed}...",
            log_path,
        )
        log_message(
            f"Model capacity: latent_dim={latent_dim}, hidden_dim={hidden_dim}, "
            f"gru_num_layers={gru_num_layers}",
            log_path,
        )
        log_message(
            f"Loss weights: recon={recon_weight}, reward={reward_weight}, kl={kl_weight}, done={done_weight}",
            log_path,
        )

        train_world_model(world_model, train_dataloader, val_dataloader,
                         epochs=epochs, start_epoch=start_epoch, checkpoint_freq=checkpoint_freq,
                         val_freq=val_freq, lr=lr, beta_kl=beta_kl,
                         loss_weights=(recon_weight, reward_weight, kl_weight, done_weight),
                         log_path=log_path)

        #torch.save(world_model.state_dict(), "world_model.pt")
        #log_message("Final world model saved to world_model.pt", log_path)

    elif args.phase == 'actor_critic':
        log_path = os.path.join(".", "train_actorcritic_logs.txt")

        horizon = phase_config.get('horizon', 20)
        past_horizon = phase_config.get('past_horizon', 5)
        future_horizon = phase_config.get('future_horizon', 15)
        assert past_horizon + future_horizon == horizon, (
            f"past_horizon + future_horizon must equal horizon; got P={past_horizon} + F={future_horizon} != H={horizon}"
        )

        action_dim = phase_config.get('action_dim', 4)
        dataset = ActorCriticWarmupDataset(
            args.train_dataset, horizon, past_horizon, future_horizon, action_dim
        )
        if len(dataset) == 0:
            raise ValueError(
                f"No valid (episode, start) positions found. Episodes need at least {horizon} steps before done. "
                f"Try a shorter horizon or a dataset with longer episodes."
            )
        batch_size = phase_config.get('batch_size', 64)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                generator=get_dataloader_generator(args.seed))

        obs_dim = dataset.obs_dim
        capacity = phase_config.get('capacity', {})
        latent_dim = capacity.get('latent_dim', 64)
        hidden_dim = capacity.get('hidden_dim', 128)
        gru_num_layers = int(capacity.get('gru_num_layers', 1))

        world_model = WorldModel(
            obs_dim,
            action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            gru_num_layers=gru_num_layers,
        ).to(DEVICE)
        if os.path.exists("world_model.pt"):
            world_model.load_state_dict(torch.load("world_model.pt", map_location=DEVICE))
            log_message("Loaded world model from world_model.pt", log_path)
        else:
            raise FileNotFoundError("world_model.pt not found. Train the world model first.")
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
        checkpoint_freq = phase_config.get('checkpoint_freq', 100)

        loss_weights = phase_config.get('loss_weights', {})
        actor_weight = loss_weights.get('actor', 1.0)
        critic_weight = loss_weights.get('critic', 0.25)
        entropy_coeff = loss_weights.get('entropy', 0.01)
        entropy_coeff_end = loss_weights.get('entropy_final', entropy_coeff)
        lambda_gae = phase_config.get('lambda_gae', 0.95)
        advantage_clip = phase_config.get('advantage_clip', 3.0)
        actor_grad_clip = phase_config.get('actor_grad_clip', 10.0)
        collapse_entropy_threshold = phase_config.get('collapse_entropy_threshold', 0.2)
        collapse_actor_grad_threshold = phase_config.get('collapse_actor_grad_threshold', 1e-3)
        collapse_max_action_prob_threshold = phase_config.get('collapse_max_action_prob_threshold', 0.98)
        collapse_patience_epochs = phase_config.get('collapse_patience_epochs', 3)
        low_entropy_actor_lr_threshold = phase_config.get('low_entropy_actor_lr_threshold', 0.9)
        reduced_actor_lr = phase_config.get('reduced_actor_lr', None)

        log_message(
            f"Training actor-critic for {epochs} epochs with lr={lr}, horizon={horizon} "
            f"(P={past_horizon}, F={future_horizon}), lambda_gae={lambda_gae}, "
            f"entropy_decay={entropy_coeff}->{entropy_coeff_end}, adv_clip={advantage_clip}, "
            f"seed={args.seed}...",
            log_path,
        )
        log_message(
            f"Model capacity: latent_dim={latent_dim}, hidden_dim={hidden_dim}, "
            f"gru_num_layers={gru_num_layers}",
            log_path,
        )
        log_message(f"Dataset: {len(dataset)} valid (episode, start) positions", log_path)

        train_actor_critic(world_model, actor, critic, dataloader,
                          epochs=epochs, lr=lr, future_horizon=future_horizon,
                          lambda_gae=lambda_gae,
                          loss_weights=(actor_weight, critic_weight), entropy_coeff=entropy_coeff,
                          entropy_coeff_end=entropy_coeff_end,
                          checkpoint_freq=checkpoint_freq,
                          start_epoch=start_epoch, log_path=log_path, use_warmup=True,
                          advantage_clip=advantage_clip, actor_grad_clip=actor_grad_clip,
                          collapse_entropy_threshold=collapse_entropy_threshold,
                          collapse_actor_grad_threshold=collapse_actor_grad_threshold,
                          collapse_max_action_prob_threshold=collapse_max_action_prob_threshold,
                          collapse_patience_epochs=collapse_patience_epochs,
                          low_entropy_actor_lr_threshold=low_entropy_actor_lr_threshold,
                          reduced_actor_lr=reduced_actor_lr)

        #torch.save(actor.state_dict(), "actor.pt")
        #torch.save(critic.state_dict(), "critic.pt")
        #log_message("Final actor and critic saved to actor.pt and critic.pt", log_path)


if __name__ == "__main__":
    main()
