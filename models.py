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

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.rnn = nn.GRUCell(hidden_dim + action_dim, hidden_dim)

        self.posterior_mean = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logstd = nn.Linear(hidden_dim, latent_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def posterior(self, obs, h):
        x = self.obs_encoder(obs)
        h = self.rnn(torch.cat([x, torch.zeros_like(x[:, :self.action_dim])], dim=-1), h)
        mean = self.posterior_mean(h)
        logstd = self.posterior_logstd(h).clamp(-5, 2)
        std = torch.exp(logstd)
        z = mean + std * torch.randn_like(std)
        return z, mean, logstd, h

    def prior(self, h, action):
        # Update recurrent state using action-only input
        zeros_obs = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        h = self.rnn(torch.cat([zeros_obs, action], dim=-1), h)
        x = torch.cat([h, action], dim=-1)
        out = self.prior_net(x)
        mean, logstd = out.chunk(2, dim=-1)
        logstd = logstd.clamp(-5, 2)
        std = torch.exp(logstd)
        z = mean + std * torch.randn_like(std)
        return z, mean, logstd, h

class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.rssm = RSSM(obs_dim, action_dim, latent_dim, hidden_dim)
        self.obs_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def reconstruct_obs(self, z):
        return self.obs_decoder(z)

    def predict_reward(self, z):
        return self.reward_head(z).squeeze(-1)

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)
