# ===========================================================================
# MoonLander RSSM — Dreamer-style World Models for LunarLander Policy Training
#
# Copyright (c) 2026 Nikolai Smolyanskiy
# Licensed under the MIT License. See LICENSE file for details.
# ===========================================================================

import torch
import torch.nn as nn

# You can tune:
#   latent_dim (e.g., 32, 64, 128, …)
#   hidden_dim in RSSM, WorldModel, Actor, Critic
#   number of layers in the GRU in the RSSM
# Smaller values → lighter GPU load, faster training.

# Recurrent State Space Model (RSSM) for the latent world model
class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_dim=16,
        hidden_dim=256,
        gru_num_layers=1,
        mlp_hidden_dim=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        # Keep hidden_dim as the deterministic state size (h) for compatibility.
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = int(mlp_hidden_dim) if mlp_hidden_dim is not None else int(hidden_dim)
        self.gru_num_layers = int(gru_num_layers)
        if self.gru_num_layers < 1:
            raise ValueError(f"gru_num_layers must be >= 1, got {self.gru_num_layers}")

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
        )

        if self.gru_num_layers == 1:
            # Keep single-layer parameter names for backward-compatible checkpoints.
            self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        else:
            self.gru_layers = nn.ModuleList(
                [nn.GRUCell(latent_dim + action_dim, hidden_dim)]
                + [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(self.gru_num_layers - 1)]
            )

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, 2 * latent_dim),
        )

        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, 2 * latent_dim),
        )

    def init_hidden(self, batch_size, device):
        if self.gru_num_layers == 1:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        return torch.zeros(batch_size, self.gru_num_layers, self.hidden_dim, device=device)

    def top_hidden(self, h):
        return h[:, -1, :] if h.dim() == 3 else h

    def update_hidden(self, h, z_prev, action_prev):
        gru_in = torch.cat([z_prev, action_prev], dim=-1)

        # Run GRU with a single layer
        if self.gru_num_layers == 1:
            if h.dim() == 3:
                h = h[:, 0, :]
            return self.gru(gru_in, h)

        # Run GRU with multiple layers (prepare hidden states if needed)
        if h.dim() == 2:
            h_prev = torch.zeros(
                h.size(0), self.gru_num_layers, self.hidden_dim, device=h.device, dtype=h.dtype
            )
            h_prev[:, 0, :] = h
        else:
            h_prev = h

        states = []
        layer_in = gru_in
        for layer_idx, cell in enumerate(self.gru_layers):
            h_i = cell(layer_in, h_prev[:, layer_idx, :])
            states.append(h_i)
            layer_in = h_i
        return torch.stack(states, dim=1)


    def prior(self, h):
        out = self.prior_net(self.top_hidden(h))
        mean, logstd = out.chunk(2, dim=-1)
        # Clamp logstd to prevent numerical instability
        # values are selected based on experience, but they may need to be changed if training fails
        logstd = logstd.clamp(-5, 2) 
        return mean, logstd

    def posterior(self, h, obs):
        obs_enc = self.obs_encoder(obs)
        post_in = torch.cat([self.top_hidden(h), obs_enc], dim=-1)
        out = self.post_net(post_in)
        mean, logstd = out.chunk(2, dim=-1)
        # Clamp logstd to prevent numerical instability
        # values are selected based on experience, but they may need to be changed if training fails
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
    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_dim=16,
        hidden_dim=256,
        gru_num_layers=1,
        mlp_hidden_dim=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.physics_dim = 6
        self.contact_dim = 2
        self.mlp_hidden_dim = int(mlp_hidden_dim) if mlp_hidden_dim is not None else int(hidden_dim)

        self.rssm = RSSM(
            obs_dim,
            action_dim,
            latent_dim,
            hidden_dim,
            gru_num_layers=gru_num_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )

        dec_in_dim = latent_dim + hidden_dim
        self.decoder_backbone = nn.Sequential(
            nn.Linear(dec_in_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
        )
        self.physics_head = nn.Linear(self.mlp_hidden_dim, self.physics_dim)
        self.contact_head = nn.Linear(self.mlp_hidden_dim, self.contact_dim)
        self.done_head = nn.Linear(self.mlp_hidden_dim, 1)

        self.reward_head = nn.Sequential(
            nn.Linear(dec_in_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.LayerNorm(self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def decode_heads(self, h, z):
        x = torch.cat([self.rssm.top_hidden(h), z], dim=-1)
        feat = self.decoder_backbone(x)
        physics = self.physics_head(feat)      # Linear output
        contact_logits = self.contact_head(feat)  # BCE-with-logits target
        done_logits = self.done_head(feat)     # BCE-with-logits target
        return physics, contact_logits, done_logits

    @staticmethod
    def make_obs_tensor(physics_pred, contact_logits):
        return torch.cat([physics_pred, torch.sigmoid(contact_logits)], dim=-1)

    def reconstruct_obs(self, h, z):
        physics, contact_logits, _ = self.decode_heads(h, z)
        return self.make_obs_tensor(physics, contact_logits)

    def predict_reward(self, h, z):
        return self.reward_head(torch.cat([self.rssm.top_hidden(h), z], dim=-1)).squeeze(-1)

    def predict_done_logits(self, h, z):
        _, _, done_logits = self.decode_heads(h, z)
        return done_logits.squeeze(-1)


class Actor(nn.Module):
    def __init__(self, latent_dim, rssm_hidden_dim, action_dim, actor_hidden_dim=128):
        super().__init__()
        input_dim = latent_dim + rssm_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, action_dim),
        )

    def forward(self, h, z):
        logits = self.net(torch.cat([h, z], dim=-1))
        return torch.distributions.Categorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, latent_dim, rssm_hidden_dim, critic_hidden_dim=128):
        super().__init__()
        input_dim = latent_dim + rssm_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class ActorObs(nn.Module):
    """Actor that operates directly on raw observations (model-free RL)."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)


class CriticObs(nn.Module):
    """Critic that operates directly on raw observations (model-free RL)."""
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
