# train_offline_dreamer.py
import argparse
import os
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

def get_latest_checkpoint(model_name):
    """Find the most recent checkpoint for a given model name."""
    pattern = f"{model_name}_*.pt"
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]

def save_checkpoint_with_timestamp(model, model_name, epoch):
    """Save a model checkpoint with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}_epoch_{epoch}.pt"
    torch.save(model.state_dict(), filename)
    print(f"Saved checkpoint: {filename}")
    return filename

class LunarDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.obs = data["obs"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.rewards = data["rewards"].astype(np.float32)
        self.dones = data["dones"].astype(np.float32)

    def __len__(self):
        return len(self.obs) - 1

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.obs[idx + 1],
        )

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
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    
    total_loss = 0.0
    total_obs_mae = 0.0
    total_obs_rmse = 0.0
    total_reward_mae = 0.0
    total_reward_rmse = 0.0
    total_reward_sign_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            obs, actions, rewards, dones, next_obs = batch
            obs = obs.to(DEVICE)
            next_obs = next_obs.to(DEVICE)
            actions = actions.to(DEVICE)
            rewards = rewards.to(DEVICE)

            batch_size = obs.size(0)
            action_onehot = torch.zeros(batch_size, 4, device=DEVICE)
            action_onehot.scatter_(1, actions.unsqueeze(-1), 1.0)

            h = world_model.rssm.init_hidden(batch_size, DEVICE)

            z_post, mean_post, logstd_post, h = world_model.rssm.posterior(obs, h)
            z_prior, mean_prior, logstd_prior = world_model.rssm.prior(h, action_onehot)

            obs_pred = world_model.reconstruct_obs(z_post)
            reward_pred = world_model.predict_reward(z_post)

            # Compute losses
            recon_loss = mse(obs_pred, next_obs)
            reward_loss = mse(reward_pred, rewards)
            kl = kl_divergence(mean_post, logstd_post, mean_prior, logstd_prior).mean()
            loss = loss_weights[0] * recon_loss + loss_weights[1] * reward_loss + loss_weights[2] * beta_kl * kl

            # Compute accuracy metrics
            obs_mae = mae(obs_pred, next_obs)
            obs_rmse = torch.sqrt(mse(obs_pred, next_obs))
            reward_mae = mae(reward_pred, rewards)
            reward_rmse = torch.sqrt(mse(reward_pred, rewards))
            
            # Reward sign accuracy (predicting positive/negative reward correctly)
            true_reward_sign = (rewards > 0).float()
            pred_reward_sign = (reward_pred > 0).float()
            reward_sign_acc = (true_reward_sign == pred_reward_sign).float().mean()

            # Accumulate metrics
            total_loss += loss.item() * batch_size
            total_obs_mae += obs_mae.item() * batch_size
            total_obs_rmse += obs_rmse.item() * batch_size
            total_reward_mae += reward_mae.item() * batch_size
            total_reward_rmse += reward_rmse.item() * batch_size
            total_reward_sign_acc += reward_sign_acc.item() * batch_size
            total_samples += batch_size
    
    # Compute averages
    avg_loss = total_loss / total_samples
    avg_obs_mae = total_obs_mae / total_samples
    avg_obs_rmse = total_obs_rmse / total_samples
    avg_reward_mae = total_reward_mae / total_samples
    avg_reward_rmse = total_reward_rmse / total_samples
    avg_reward_sign_acc = total_reward_sign_acc / total_samples
    
    return {
        'loss': avg_loss,
        'obs_mae': avg_obs_mae,
        'obs_rmse': avg_obs_rmse,
        'reward_mae': avg_reward_mae,
        'reward_rmse': avg_reward_rmse,
        'reward_sign_acc': avg_reward_sign_acc
    }

def train_world_model(world_model, train_dataloader, val_dataloader, epochs=10, lr=3e-4, beta_kl=1.0, val_freq=10, loss_weights=(1.0, 1.0, 1.0), checkpoint_freq=100):
    world_model.train()
    opt = optim.AdamW(world_model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs):
        # Training phase
        train_loss = 0.0
        for batch in train_dataloader:
            obs, actions, rewards, dones, next_obs = batch
            obs = obs.to(DEVICE)
            next_obs = next_obs.to(DEVICE)
            actions = actions.to(DEVICE)
            rewards = rewards.to(DEVICE)

            batch_size = obs.size(0)
            action_onehot = torch.zeros(batch_size, 4, device=DEVICE)
            action_onehot.scatter_(1, actions.unsqueeze(-1), 1.0)

            h = world_model.rssm.init_hidden(batch_size, DEVICE)

            z_post, mean_post, logstd_post, h = world_model.rssm.posterior(obs, h)
            z_prior, mean_prior, logstd_prior = world_model.rssm.prior(h, action_onehot)

            obs_pred = world_model.reconstruct_obs(z_post)
            reward_pred = world_model.predict_reward(z_post)

            recon_loss = mse(obs_pred, next_obs)
            reward_loss = mse(reward_pred, rewards)

            kl = kl_divergence(mean_post, logstd_post, mean_prior, logstd_prior).mean()

            loss = loss_weights[0] * recon_loss + loss_weights[1] * reward_loss + loss_weights[2] * beta_kl * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
            opt.step()

            train_loss += loss.item() * batch_size

        train_loss /= len(train_dataloader.dataset)
        
        # Validation phase every val_freq epochs
        if (epoch + 1) % val_freq == 0:
            val_metrics = validate_world_model(world_model, val_dataloader, beta_kl, loss_weights)
            print(f"[WorldModel] Epoch {epoch+1}, train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}")
            print(f"  Validation Metrics - Obs MAE: {val_metrics['obs_mae']:.4f}, Obs RMSE: {val_metrics['obs_rmse']:.4f}")
            print(f"  Reward MAE: {val_metrics['reward_mae']:.4f}, Reward RMSE: {val_metrics['reward_rmse']:.4f}, Reward Sign Acc: {val_metrics['reward_sign_acc']:.3f}")
        else:
            print(f"[WorldModel] Epoch {epoch+1}, train_loss={train_loss:.4f}")

        # Save checkpoint every checkpoint_freq epochs
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint_with_timestamp(world_model, "world_model", epoch + 1)

def compute_auxiliary_rewards(observations, upright_weight=0.1, downward_speed_weight=0.05):
    """
    Compute auxiliary rewards for upright pose and minimal downward speed.
    
    Args:
        observations: Tensor of shape (batch_size, seq_len, obs_dim)
        upright_weight: Weight for upright pose reward
        downward_speed_weight: Weight for downward speed penalty
        
    Returns:
        auxiliary_rewards: Tensor of shape (batch_size, seq_len)
    """
    # Extract angle (index 4) and y-velocity (index 3) from observations
    angles = observations[..., 4]  # Angle in radians
    y_velocities = observations[..., 3]  # Y velocity
    
    # Upright pose reward: higher when angle is close to 0
    upright_reward = upright_weight * torch.exp(-torch.abs(angles))
    
    # Downward speed penalty: penalize negative y velocities (falling down)
    downward_penalty = downward_speed_weight * torch.clamp(y_velocities, max=0.0)  # Only penalize negative velocities
    
    auxiliary_rewards = upright_reward + downward_penalty
    return auxiliary_rewards

def imagine_rollout(world_model, actor, start_obs, horizon=15):
    world_model.eval()
    actor.eval()
    with torch.no_grad():
        batch_size = start_obs.size(0)
        h = world_model.rssm.init_hidden(batch_size, DEVICE)
        z, _, _, h = world_model.rssm.posterior(start_obs, h)

    zs = []
    rewards = []
    actions = []
    observations = []  # Store imagined observations for auxiliary rewards

    for t in range(horizon):
        dist = actor(z)
        a = dist.sample()
        a_onehot = torch.zeros(batch_size, 4, device=DEVICE)
        a_onehot.scatter_(1, a.unsqueeze(-1), 1.0)

        z, mean_prior, logstd_prior = world_model.rssm.prior(h, a_onehot)
        r = world_model.predict_reward(z)
        obs_imagined = world_model.reconstruct_obs(z)  # Reconstruct observation from latent

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
                       imagination_horizon=15, loss_weights=(1.0, 1.0), entropy_coeff=0.01, checkpoint_freq=100, aux_rewards_config=None, start_epoch=0):
    actor.train()
    critic.train()
    opt_actor = optim.AdamW(actor.parameters(), lr=lr)
    opt_critic = optim.AdamW(critic.parameters(), lr=lr)

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

            # Detach to prevent gradients from flowing back to world_model
            zs = zs.detach()
            imagined_rewards = imagined_rewards.detach()
            imagined_actions = imagined_actions.detach()
            imagined_observations = imagined_observations.detach()

            # Add auxiliary rewards for upright pose and minimal downward speed
            aux_rewards = compute_auxiliary_rewards(imagined_observations, 
                                                   upright_weight=aux_rewards_config.get('upright_pose_weight', 0.1),
                                                   downward_speed_weight=aux_rewards_config.get('downward_speed_weight', 0.05))
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

            # Compute additional metrics
            imagined_reward_mean = imagined_rewards.mean().item()
            value_mae = torch.abs(value_pred - returns.reshape(-1).detach()).mean().item()
            entropy = dist.entropy().mean().item()
            
            # Auxiliary reward components
            upright_aux = aux_rewards.mean().item()  # Since aux_rewards includes both, but we can compute separately if needed
            # For breakdown, compute upright and downward separately
            angles = imagined_observations[..., 4]
            y_velocities = imagined_observations[..., 3]
            upright_reward = aux_rewards_config.get('upright_pose_weight', 0.1) * torch.exp(-torch.abs(angles)).mean().item()
            downward_penalty = aux_rewards_config.get('downward_speed_weight', 0.05) * torch.clamp(y_velocities, max=0.0).mean().item()

            # Accumulate metrics
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

        # Average metrics over batches
        avg_actor_loss = total_actor_loss / num_batches
        avg_critic_loss = total_critic_loss / num_batches
        avg_imagined_reward = total_imagined_reward / num_batches
        avg_value_mae = total_value_mae / num_batches
        avg_entropy = total_entropy / num_batches
        avg_actor_grad_norm = total_actor_grad_norm / num_batches
        avg_critic_grad_norm = total_critic_grad_norm / num_batches
        avg_aux_upright = total_aux_upright / num_batches
        avg_aux_downward = total_aux_downward / num_batches

        print(f"[ActorCritic] Epoch {epoch+1}, actor_loss={avg_actor_loss:.4f}, critic_loss={avg_critic_loss:.4f}")
        print(f"  Metrics - Imagined Reward: {avg_imagined_reward:.4f}, Value MAE: {avg_value_mae:.4f}, Entropy: {avg_entropy:.4f}")
        print(f"  Grad Norms - Actor: {avg_actor_grad_norm:.4f}, Critic: {avg_critic_grad_norm:.4f}")
        print(f"  Aux Rewards - Upright: {avg_aux_upright:.4f}, Downward Penalty: {avg_aux_downward:.4f}")

        # Save checkpoints every checkpoint_freq epochs
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint_with_timestamp(actor, "actor", epoch + 1)
            save_checkpoint_with_timestamp(critic, "critic", epoch + 1)

def main():
    parser = argparse.ArgumentParser(description="Train Dreamer offline on LunarLander")
    parser.add_argument('--phase', choices=['world_model', 'actor_critic'], required=True,
                        help='Which phase to train: world_model or actor_critic')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    phase_config = config[args.phase]

    if args.phase == 'world_model':
        # Load separate train and validation datasets for world model training
        train_dataset = LunarDataset("lunarlander_train_dataset.npz")
        val_dataset = LunarDataset("lunarlander_val_dataset.npz")
        
        batch_size = phase_config.get('batch_size', 64)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        obs_dim = train_dataset.obs.shape[1]
        action_dim = 4

        # Get capacity parameters
        capacity = phase_config.get('capacity', {})
        latent_dim = capacity.get('latent_dim', 64)
        hidden_dim = capacity.get('hidden_dim', 128)

        world_model = WorldModel(obs_dim, action_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(DEVICE)
        
        # Load the most recent world model checkpoint
        latest_checkpoint = get_latest_checkpoint("world_model")
        if latest_checkpoint:
            world_model.load_state_dict(torch.load(latest_checkpoint, map_location=DEVICE))
            print(f"Loaded world model checkpoint: {latest_checkpoint}")
        elif os.path.exists("world_model.pt"):
            # Fallback to old naming scheme
            world_model.load_state_dict(torch.load("world_model.pt", map_location=DEVICE))
            print("Loaded existing world model checkpoint (old naming)")
        
        epochs = phase_config.get('epochs', 20)
        lr = phase_config.get('lr', 3e-4)
        beta_kl = phase_config.get('beta_kl', 1.0)
        val_freq = phase_config.get('val_freq', 10)
        checkpoint_freq = phase_config.get('checkpoint_freq', 100)
        
        # Get loss weights
        loss_weights = phase_config.get('loss_weights', {})
        recon_weight = loss_weights.get('reconstruction', 1.0)
        reward_weight = loss_weights.get('reward', 1.0)
        kl_weight = loss_weights.get('kl', 1.0)
        
        print(f"Training world model for {epochs} epochs with lr={lr}, beta_kl={beta_kl}, val_freq={val_freq}...")
        print(f"Model capacity: latent_dim={latent_dim}, hidden_dim={hidden_dim}")
        print(f"Loss weights: recon={recon_weight}, reward={reward_weight}, kl={kl_weight}")
        print(f"Checkpoint frequency: every {checkpoint_freq} epochs")
        train_world_model(world_model, train_dataloader, val_dataloader, epochs=epochs, lr=lr, 
                         beta_kl=beta_kl, val_freq=val_freq, loss_weights=(recon_weight, reward_weight, kl_weight),
                         checkpoint_freq=checkpoint_freq)
        # Save final model (in addition to checkpoints)
        torch.save(world_model.state_dict(), "world_model.pt")
        print("Final world model saved to world_model.pt")

    elif args.phase == 'actor_critic':
        # For actor-critic, use the combined dataset (or you could use train_dataset)
        dataset = LunarDataset("lunarlander_train_dataset.npz")  # Use train data for actor-critic training
        
        batch_size = phase_config.get('batch_size', 64)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        obs_dim = dataset.obs.shape[1]
        action_dim = 4

        # Get capacity parameters
        capacity = phase_config.get('capacity', {})
        latent_dim = capacity.get('latent_dim', 64)
        hidden_dim = capacity.get('hidden_dim', 128)

        world_model = WorldModel(obs_dim, action_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(DEVICE)
        world_model.load_state_dict(torch.load("world_model.pt", map_location=DEVICE))
        world_model.eval()  # Freeze world model

        actor = Actor(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(DEVICE)
        critic = Critic(latent_dim=latent_dim, hidden_dim=hidden_dim).to(DEVICE)

        # Load the most recent actor and critic checkpoints
        latest_actor = get_latest_checkpoint("actor")
        start_epoch = 0
        if latest_actor:
            actor.load_state_dict(torch.load(latest_actor, map_location=DEVICE))
            print(f"Loaded actor checkpoint: {latest_actor}")
            # Extract epoch from filename, e.g., actor_..._epoch_10.pt -> 10
            epoch_str = latest_actor.split('_epoch_')[-1].split('.')[0]
            start_epoch = int(epoch_str)
        elif os.path.exists("actor.pt"):
            actor.load_state_dict(torch.load("actor.pt", map_location=DEVICE))
            print("Loaded existing actor checkpoint (old naming)")
        else:
            print("No actor checkpoint found, starting from scratch")

        latest_critic = get_latest_checkpoint("critic")
        if latest_critic:
            critic.load_state_dict(torch.load(latest_critic, map_location=DEVICE))
            print(f"Loaded critic checkpoint: {latest_critic}")
        elif os.path.exists("critic.pt"):
            critic.load_state_dict(torch.load("critic.pt", map_location=DEVICE))
            print("Loaded existing critic checkpoint (old naming)")
        else:
            print("No critic checkpoint found, starting from scratch")

        epochs = phase_config.get('epochs', 20)
        lr = phase_config.get('lr', 3e-4)
        imagination_horizon = phase_config.get('imagination_horizon', 15)
        checkpoint_freq = phase_config.get('checkpoint_freq', 100)
        
        # Get loss weights
        loss_weights = phase_config.get('loss_weights', {})
        actor_weight = loss_weights.get('actor', 1.0)
        critic_weight = loss_weights.get('critic', 1.0)
        entropy_coeff = loss_weights.get('entropy', 0.01)
        
        # Get auxiliary rewards config
        aux_rewards_config = phase_config.get('auxiliary_rewards', {})
        
        print(f"Training actor-critic for {epochs} epochs with lr={lr}, imagination_horizon={imagination_horizon}... (starting from epoch {start_epoch})")
        print(f"Model capacity: latent_dim={latent_dim}, hidden_dim={hidden_dim}")
        print(f"Loss weights: actor={actor_weight}, critic={critic_weight}")
        print(f"Auxiliary rewards: upright={aux_rewards_config.get('upright_pose_weight', 0.1)}, downward_speed={aux_rewards_config.get('downward_speed_weight', 0.05)}")
        print(f"Checkpoint frequency: every {checkpoint_freq} epochs")
        train_actor_critic(world_model, actor, critic, dataloader,
                           epochs=epochs, lr=lr, imagination_horizon=imagination_horizon, 
                           loss_weights=(actor_weight, critic_weight), entropy_coeff=entropy_coeff, checkpoint_freq=checkpoint_freq,
                           aux_rewards_config=aux_rewards_config, start_epoch=start_epoch)

        # Save final models (in addition to checkpoints)
        torch.save(actor.state_dict(), "actor.pt")
        torch.save(critic.state_dict(), "critic.pt")
        print("Final actor and critic saved to actor.pt and critic.pt")

if __name__ == "__main__":
    main()
