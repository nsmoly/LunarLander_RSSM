# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# you can tune:
# latent_dim (e.g., 32, 64, 128, …)
# hidden_dim in RSSM, WorldModel, Actor, Critic
# Smaller values → lighter GPU load, faster training.

class RSSM(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def update_hidden(self, h, z_prev, action_prev):
        gru_in = torch.cat([z_prev, action_prev], dim=-1)
        return self.gru(gru_in, h)

    def prior(self, h):
        out = self.prior_net(h)
        mean, logstd = out.chunk(2, dim=-1)
        logstd = logstd.clamp(-5, 2)
        return mean, logstd

    def posterior(self, h, obs):
        obs_enc = self.obs_encoder(obs)
        post_in = torch.cat([h, obs_enc], dim=-1)
        out = self.post_net(post_in)
        mean, logstd = out.chunk(2, dim=-1)
        logstd = logstd.clamp(-5, 2)
        return mean, logstd

    def sample_latent(self, mean, logstd):
        std = torch.exp(logstd)
        return mean + std * torch.randn_like(std)

    def step(self, h, z_prev, action_prev, obs):
        h_t = self.update_hidden(h, z_prev, action_prev)
        mean_prior, logstd_prior = self.prior(h_t)
        mean_post, logstd_post = self.posterior(h_t, obs)
        z_t = self.sample_latent(mean_post, logstd_post)
        return h_t, z_t, mean_post, logstd_post, mean_prior, logstd_prior

class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.rssm = RSSM(obs_dim, action_dim, latent_dim, hidden_dim)
        self.obs_decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def reconstruct_obs(self, h, z):
        return self.obs_decoder(torch.cat([h, z], dim=-1))

    def predict_reward(self, h, z):
        return self.reward_head(torch.cat([h, z], dim=-1)).squeeze(-1)

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z):
        logits = self.net(z)
        return torch.distributions.Categorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)
