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
            next_obs_seq = next_obs_seq.to(DEVICE)
            dones_seq = dones_seq.to(DEVICE).float()
            mask = mask.to(DEVICE)

            batch_size, seq_len = obs_seq.shape[:2]
            action_dim = world_model.rssm.action_dim
            latent_dim = world_model.rssm.latent_dim

            h = world_model.rssm.init_hidden(batch_size, DEVICE)
            z = torch.zeros(batch_size, latent_dim, device=DEVICE)
            a_init = torch.zeros(batch_size, action_dim, device=DEVICE)

            sum_recon = 0.0
            sum_rew = 0.0
            sum_kl = 0.0
            sum_done = 0.0
            total_mask = mask.sum().clamp_min(1.0)

            # Bootstrap: encode obs_seq[:, 0] to get initial RSSM state (h_0, z_0)
            # This gives us the starting state s_0 from which transitions are predicted.
            h = world_model.rssm.update_hidden(h, z, a_init)
            mean_post_0, logstd_post_0 = world_model.rssm.posterior(h, obs_seq[:, 0])
            z = world_model.rssm.sample_latent(mean_post_0, logstd_post_0)

            # Transition loop: at each step t, we have state (h_t, z_t) encoding s_t.
            # We take action a_t, roll the GRU to get h_{t+1}, then:
            #   - prior(h_{t+1})    = what the model predicts about s_{t+1}
            #   - posterior(h_{t+1}, next_obs[t]) = ground-truth-informed s_{t+1}
            #   - decode(h_{t+1}, z_{t+1})  → reconstruct next_obs[t] (= s_{t+1})
            #   - predict_reward(h_{t+1}, z_{t+1}) → reward for (s_t, a_t) transition
            #   - predict_done(h_{t+1}, z_{t+1})   → done flag for (s_t, a_t) transition
            #   - KL(posterior || prior) at t+1
            for t in range(seq_len):
                a_t = torch.nn.functional.one_hot(actions_seq[:, t], num_classes=action_dim).float()

                # Roll hidden state forward: h_{t+1} = GRU(h_t, z_t, a_t)
                h = world_model.rssm.update_hidden(h, z, a_t)

                # Prior at t+1: model's prediction of s_{t+1} given history + a_t
                mean_prior, logstd_prior = world_model.rssm.prior(h)

                # Posterior at t+1: informed by actual next observation s_{t+1}
                mean_post, logstd_post = world_model.rssm.posterior(h, next_obs_seq[:, t])
                z = world_model.rssm.sample_latent(mean_post, logstd_post)

                # Decode from (h_{t+1}, z_{t+1}) — should reconstruct s_{t+1}
                physics_pred, contact_logits, done_logits_2d = world_model.decode_heads(h, z)
                reward_pred = world_model.predict_reward(h, z)
                done_logits = done_logits_2d.squeeze(-1)

                # Targets are all from the transition (s_t, a_t) → s_{t+1}
                physics_tgt = next_obs_seq[:, t, :6]
                contact_tgt = next_obs_seq[:, t, 6:8]
                physics_loss = (physics_pred - physics_tgt).pow(2).sum(-1)
                contact_loss = F.binary_cross_entropy_with_logits(
                    contact_logits, contact_tgt, reduction="none"
                ).sum(-1)
                recon_loss = physics_loss + contact_loss
                reward_loss = (reward_pred - rewards_seq[:, t]).pow(2)
                done_loss = F.binary_cross_entropy_with_logits(done_logits, dones_seq[:, t], reduction="none")
                kl = kl_divergence(mean_post, logstd_post, mean_prior, logstd_prior)

                mask_t = mask[:, t]
                sum_recon += (recon_loss * mask_t).sum()
                sum_rew += (reward_loss * mask_t).sum()
                sum_done += (done_loss * mask_t).sum()
                sum_kl += (kl * mask_t).sum()

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
    was_training = world_model.training
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
            next_obs_seq = next_obs_seq.to(DEVICE)
            dones_seq = dones_seq.to(DEVICE).float()
            mask = mask.to(DEVICE)

            batch_size, seq_len = obs_seq.shape[:2]
            action_dim = world_model.rssm.action_dim
            latent_dim = world_model.rssm.latent_dim

            h = world_model.rssm.init_hidden(batch_size, DEVICE)
            z = torch.zeros(batch_size, latent_dim, device=DEVICE)
            a_init = torch.zeros(batch_size, action_dim, device=DEVICE)

            sum_recon = 0.0
            sum_rew = 0.0
            sum_kl = 0.0
            sum_done = 0.0

            # Bootstrap: encode obs_seq[:, 0] to get initial state (h_0, z_0)
            h = world_model.rssm.update_hidden(h, z, a_init)
            mean_post_0, logstd_post_0 = world_model.rssm.posterior(h, obs_seq[:, 0])
            z = world_model.rssm.sample_latent(mean_post_0, logstd_post_0)

            # Transition loop: predict s_{t+1} from (s_t, a_t), supervise with next_obs[t]
            for t in range(seq_len):
                a_t = torch.nn.functional.one_hot(actions_seq[:, t], num_classes=action_dim).float()

                # h_{t+1} = GRU(h_t, z_t, a_t)
                h = world_model.rssm.update_hidden(h, z, a_t)
                mean_prior, logstd_prior = world_model.rssm.prior(h)
                mean_post, logstd_post = world_model.rssm.posterior(h, next_obs_seq[:, t])
                z = world_model.rssm.sample_latent(mean_post, logstd_post)

                # Decode (h_{t+1}, z_{t+1}) → reconstruct s_{t+1} = next_obs[t]
                physics_pred, contact_logits, done_logits_2d = world_model.decode_heads(h, z)
                obs_pred = world_model.make_obs_tensor(physics_pred, contact_logits)
                reward_pred = world_model.predict_reward(h, z)
                done_logits = done_logits_2d.squeeze(-1)

                # Targets from transition (s_t, a_t) → s_{t+1}
                physics_tgt = next_obs_seq[:, t, :6]
                contact_tgt = next_obs_seq[:, t, 6:8]
                physics_loss = (physics_pred - physics_tgt).pow(2).sum(-1)
                contact_loss = F.binary_cross_entropy_with_logits(
                    contact_logits, contact_tgt, reduction="none"
                ).sum(-1)
                recon_loss = physics_loss + contact_loss
                reward_loss = (reward_pred - rewards_seq[:, t]).pow(2)
                done_loss = F.binary_cross_entropy_with_logits(done_logits, dones_seq[:, t], reduction="none")
                kl = kl_divergence(mean_post, logstd_post, mean_prior, logstd_prior)

                mask_t = mask[:, t]
                sum_recon += (recon_loss * mask_t).sum()
                sum_rew += (reward_loss * mask_t).sum()
                sum_done += (done_loss * mask_t).sum()
                sum_kl += (kl * mask_t).sum()

                obs_diff = (obs_pred - next_obs_seq[:, t]).abs().sum(-1)
                total_obs_abs += (obs_diff * mask_t).sum().item()
                total_obs_sq += ((obs_pred - next_obs_seq[:, t]).pow(2).sum(-1) * mask_t).sum().item()
                total_reward_abs += ((reward_pred - rewards_seq[:, t]).abs() * mask_t).sum().item()
                total_reward_sq += (((reward_pred - rewards_seq[:, t]).pow(2)) * mask_t).sum().item()

                true_reward_sign = (rewards_seq[:, t] > 0).float()
                pred_reward_sign = (reward_pred > 0).float()
                total_reward_sign_correct += ((true_reward_sign == pred_reward_sign).float() * mask_t).sum().item()
                done_pred = (torch.sigmoid(done_logits) >= 0.5).float()
                total_done_correct += ((done_pred == dones_seq[:, t]).float() * mask_t).sum().item()

                total_mask += mask_t.sum().item()

            batch_mask = mask.sum().item()
            if batch_mask > 0:
                total_loss += (
                    loss_weights[0] * sum_recon
                    + loss_weights[1] * sum_rew
                    + loss_weights[3] * sum_done
                    + loss_weights[2] * beta_kl * sum_kl
                ).item()

    world_model.train(was_training)

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
def imagine_rollout(world_model, actor, obs_seq, actions_seq,
                    warmup_steps, imagination_steps):
    """Warm-up RSSM with real (obs, action) for warmup_steps, then imagine
    imagination_steps using the actor policy.  Caller must ensure all
    sequences have at least warmup_steps valid steps.

    Convention: reward/done at step t are decoded from the
    post-transition state (h_{t+1}, z_{t+1}), matching WM training where
    predict_reward(h_{t+1}, z_{t+1}) targets r(s_t, a_t).

    Returns (zs, hs, rewards, actions, done_probs) where:
      zs, hs:     shape (B, T+1, ...) — states s_0..s_T+1 (T = imagination_steps)
      rewards:    shape (B, T)        — r_t decoded from (h_{t+1}, z_{t+1})
      actions:    shape (B, T)        — a_t taken at (h_t, z_t)
      done_probs: shape (B, T)        — P(done) decoded from (h_{t+1}, z_{t+1})
    """
    ac_training = actor.training
    actor.eval()
    batch_size = obs_seq.size(0)
    action_dim = world_model.rssm.action_dim
    latent_dim = world_model.rssm.latent_dim
    hidden_dim = world_model.rssm.hidden_dim
    T = imagination_steps

    with torch.no_grad():
        # Warmup: run RSSM with real observations and actions to build
        # an accurate latent state before free-running imagination.
        h = world_model.rssm.init_hidden(batch_size, DEVICE)
        z = torch.zeros(batch_size, latent_dim, device=DEVICE)
        a_prev = torch.zeros(batch_size, action_dim, device=DEVICE)

        # Bootstrap from obs_seq[:, 0] (same convention as WM training)
        h = world_model.rssm.update_hidden(h, z, a_prev)
        mean_post, logstd_post = world_model.rssm.posterior(h, obs_seq[:, 0])
        z = world_model.rssm.sample_latent(mean_post, logstd_post)

        # Continue warmup: transition (s_t, a_t) → s_{t+1} via obs_seq
        for t in range(warmup_steps - 1):
            a_t = torch.nn.functional.one_hot(actions_seq[:, t], num_classes=action_dim).float()
            h = world_model.rssm.update_hidden(h, z, a_t)
            mean_post, logstd_post = world_model.rssm.posterior(h, obs_seq[:, t + 1])
            z = world_model.rssm.sample_latent(mean_post, logstd_post)

        # Pre-allocate: T+1 states (s_0..s_T), T transitions
        zs = torch.zeros(batch_size, T + 1, latent_dim, device=DEVICE)
        hs = torch.zeros(batch_size, T + 1, hidden_dim, device=DEVICE)
        rewards = torch.zeros(batch_size, T, device=DEVICE)
        actions = torch.zeros(batch_size, T, dtype=torch.long, device=DEVICE)
        done_probs = torch.zeros(batch_size, T, device=DEVICE)

        # s_0 = state after warmup
        zs[:, 0] = z
        hs[:, 0] = h

        for t in range(T):
            # At state (h_t, z_t): actor picks a_t
            action_distr = actor(hs[:, t], zs[:, t])
            a = action_distr.sample()
            a_onehot = torch.nn.functional.one_hot(a, num_classes=action_dim).float()

            # Transition: h_{t+1} = GRU(h_t, z_t, a_t), z_{t+1} = prior(h_{t+1})
            h = world_model.rssm.update_hidden(h, zs[:, t], a_onehot)
            mean_prior, _ = world_model.rssm.prior(h)
            z = mean_prior

            # Store s_{t+1} and transition outputs
            zs[:, t + 1] = z
            hs[:, t + 1] = h
            actions[:, t] = a
            # Convention: reward/done decoded from post-transition state
            rewards[:, t] = world_model.predict_reward(h, z)
            done_probs[:, t] = torch.sigmoid(world_model.predict_done_logits(h, z))

    actor.train(ac_training)
    return zs, hs, rewards, actions, done_probs


def train_actor_critic(world_model, actor, critic, dataloader,
                       epochs=10, lr=3e-4, gamma=0.99, lambda_gae=0.95,
                       warmup_steps=5, imagination_steps=15,
                       loss_weights=(1.0, 1.0), entropy_coeff=0.01,
                       entropy_coeff_end=None,
                       checkpoint_freq=100, start_epoch=0, log_path=None,
                       advantage_clip=3.0, actor_grad_clip=10.0, critic_grad_clip=100.0,
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
            obs_seq, actions_seq, rewards_seq, next_obs_seq, dones_seq, mask = batch
            obs_seq = obs_seq.to(DEVICE)
            actions_seq = actions_seq.to(DEVICE)
            mask = mask.to(DEVICE)

            valid = mask.sum(dim=1) >= warmup_steps
            if not valid.any():
                continue
            if not valid.all():
                obs_seq = obs_seq[valid]
                actions_seq = actions_seq[valid]
                mask = mask[valid]

            zs, hs, imagined_rewards, imagined_actions, imagined_done_probs = imagine_rollout(
                world_model, actor, obs_seq, actions_seq,
                warmup_steps=warmup_steps, imagination_steps=imagination_steps,
            )
            # zs, hs: (B, T+1, ...) — states s_0..s_T+1
            # imagined_rewards[t]   = reward for (s_t, a_t) transition, decoded from s_{t+1}
            # imagined_done_probs[t]= P(done) for (s_t, a_t) transition, decoded from s_{t+1}
            # imagined_actions[t]   = a_t taken at s_t

            with torch.no_grad():
                # GAE: V(s_t) from source states, V(s_{t+1}) from next states
                src_h = hs[:, :-1].reshape(-1, hs.size(-1))   # (B*T, hidden_dim)
                src_z = zs[:, :-1].reshape(-1, zs.size(-1))   # (B*T, latent_dim)
                nxt_h = hs[:, 1:].reshape(-1, hs.size(-1))    # (B*T, hidden_dim)
                nxt_z = zs[:, 1:].reshape(-1, zs.size(-1))    # (B*T, latent_dim)
                B = hs.size(0)

                cur_values = critic(src_h, src_z).reshape(B, imagination_steps)
                next_values = critic(nxt_h, nxt_z).reshape(B, imagination_steps)

                continues = 1.0 - imagined_done_probs
                deltas = imagined_rewards + gamma * continues * next_values - cur_values
                adv = torch.zeros_like(deltas)
                gae = torch.zeros_like(deltas[:, 0])
                for t in reversed(range(imagination_steps)):
                    gae = deltas[:, t] + gamma * lambda_gae * continues[:, t] * gae
                    adv[:, t] = gae
                returns = adv + cur_values

            # Actor loss: evaluate at source states (correct state-action pairing)
            action_distr = actor(src_h, src_z)
            log_probs = action_distr.log_prob(imagined_actions.reshape(-1))
            adv_flat = adv.reshape(-1).detach()
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            adv_flat = torch.clamp(adv_flat, -advantage_clip, advantage_clip)

            actor_loss = loss_weights[0] * (
                -(log_probs * adv_flat).mean() - current_entropy_coeff * action_distr.entropy().mean()
            )

            value_pred = critic(src_h, src_z)
            critic_loss = loss_weights[1] * (value_pred - returns.reshape(-1).detach()).pow(2).mean()

            opt_actor.zero_grad()
            actor_loss.backward()
            actor_grad_norm = _grad_norm(actor.parameters())
            torch.nn.utils.clip_grad_norm_(actor.parameters(), actor_grad_clip)
            opt_actor.step()

            opt_critic.zero_grad()
            critic_loss.backward()
            critic_grad_norm = _grad_norm(critic.parameters())
            torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_grad_clip)
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

        if num_batches == 0:
            log_message(f"[ActorCritic] Epoch {epoch}, no valid batches (all sequences too short)", log_path)
            continue

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
    parser = argparse.ArgumentParser(description="Trains world model and actor-critic policy offline from the data collected in LunarLander gym environment")
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
        latent_dim = capacity.get('latent_dim', 16)
        hidden_dim = capacity.get('hidden_dim', 256)
        mlp_hidden_dim = capacity.get('mlp_hidden_dim', hidden_dim)
        gru_num_layers = int(capacity.get('gru_num_layers', 1))

        world_model = WorldModel(
            obs_dim,
            action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            gru_num_layers=gru_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
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
            f"mlp_hidden_dim={mlp_hidden_dim}, gru_num_layers={gru_num_layers}",
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

        action_dim = phase_config.get('action_dim', 4)
        warmup_steps = phase_config.get('warmup_steps', 5)
        imagination_steps = phase_config.get('imagination_steps', 15)
        sequence_length = phase_config.get('sequence_length', warmup_steps)
        dataset_seq_offset = phase_config.get('dataset_seq_offset', 5)

        if sequence_length < warmup_steps:
            raise ValueError(
                f"sequence_length ({sequence_length}) must be >= warmup_steps ({warmup_steps})"
            )

        dataset = SequenceDataset(
            args.train_dataset,
            sequence_length,
            action_dim,
            random_start=True,
            dataset_seq_offset=dataset_seq_offset,
        )
        if len(dataset) == 0:
            raise ValueError(
                f"No valid start positions found in the dataset. "
                f"Try a shorter sequence_length or a dataset with longer episodes."
            )
        batch_size = phase_config.get('batch_size', 64)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                generator=get_dataloader_generator(args.seed))

        obs_dim = dataset.obs_dim
        capacity = phase_config.get('capacity', {})
        latent_dim = capacity.get('latent_dim', 16)
        hidden_dim = capacity.get('hidden_dim', 256)
        mlp_hidden_dim = capacity.get('mlp_hidden_dim', hidden_dim)
        gru_num_layers = int(capacity.get('gru_num_layers', 1))

        world_model = WorldModel(
            obs_dim,
            action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            gru_num_layers=gru_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        ).to(DEVICE)
        if os.path.exists("world_model.pt"):
            world_model.load_state_dict(torch.load("world_model.pt", map_location=DEVICE))
            log_message("Loaded world model from world_model.pt", log_path)
        else:
            raise FileNotFoundError("world_model.pt not found. Train the world model first.")
        world_model.eval()

        actor = Actor(latent_dim=latent_dim, rssm_hidden_dim=hidden_dim,
                      action_dim=action_dim, actor_hidden_dim=mlp_hidden_dim).to(DEVICE)
        critic = Critic(latent_dim=latent_dim, rssm_hidden_dim=hidden_dim,
                        critic_hidden_dim=mlp_hidden_dim).to(DEVICE)

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
        gamma = phase_config.get('gamma', 0.99)
        lambda_gae = phase_config.get('lambda_gae', 0.95)
        advantage_clip = phase_config.get('advantage_clip', 3.0)
        actor_grad_clip = phase_config.get('actor_grad_clip', 10.0)
        critic_grad_clip = phase_config.get('critic_grad_clip', 100.0)
        collapse_entropy_threshold = phase_config.get('collapse_entropy_threshold', 0.2)
        collapse_actor_grad_threshold = phase_config.get('collapse_actor_grad_threshold', 1e-3)
        collapse_max_action_prob_threshold = phase_config.get('collapse_max_action_prob_threshold', 0.98)
        collapse_patience_epochs = phase_config.get('collapse_patience_epochs', 3)
        low_entropy_actor_lr_threshold = phase_config.get('low_entropy_actor_lr_threshold', 0.9)
        reduced_actor_lr = phase_config.get('reduced_actor_lr', None)

        log_message(
            f"Training actor-critic for {epochs} epochs with lr={lr}, "
            f"warmup_steps={warmup_steps}, imagination_steps={imagination_steps}, "
            f"sequence_length={sequence_length}, dataset_seq_offset={dataset_seq_offset}, "
            f"lambda_gae={lambda_gae}, "
            f"entropy_decay={entropy_coeff}->{entropy_coeff_end}, adv_clip={advantage_clip}, "
            f"seed={args.seed}...",
            log_path,
        )
        log_message(
            f"Model capacity: latent_dim={latent_dim}, hidden_dim={hidden_dim}, "
            f"mlp_hidden_dim={mlp_hidden_dim}, gru_num_layers={gru_num_layers}",
            log_path,
        )
        log_message(f"Dataset: {len(dataset)} valid (episode, start) positions", log_path)

        train_actor_critic(world_model, actor, critic, dataloader,
                          epochs=epochs, lr=lr, gamma=gamma,
                          warmup_steps=warmup_steps,
                          imagination_steps=imagination_steps,
                          lambda_gae=lambda_gae,
                          loss_weights=(actor_weight, critic_weight), entropy_coeff=entropy_coeff,
                          entropy_coeff_end=entropy_coeff_end,
                          checkpoint_freq=checkpoint_freq,
                          start_epoch=start_epoch, log_path=log_path,
                          advantage_clip=advantage_clip, actor_grad_clip=actor_grad_clip, critic_grad_clip=critic_grad_clip,
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
