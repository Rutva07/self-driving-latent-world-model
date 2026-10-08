"""Temporal agent encoder -> scene latents -> multimodal trajectory decoder."""

import torch
from torch import nn

from lwm.physics import rollout


class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * dim, dim))

    def forward(self, x):
        return self.net(x)


class LatentWorldModel(nn.Module):
    def __init__(self, d_model=128, heads=4, temporal_layers=2, latent_layers=2,
                 num_latents=16, num_modes=6, dropout=0.12, history_steps=11,
                 future_steps=80, use_map=True, use_interactions=True):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.use_map = use_map
        self.use_interactions = use_interactions
        self.agent_projection = nn.Sequential(nn.Linear(8, d_model), nn.LayerNorm(d_model),
                                              nn.GELU(), nn.Linear(d_model, d_model))
        self.type_embedding = nn.Embedding(5, d_model, padding_idx=0)
        self.time_embed = nn.Parameter(torch.randn(1, history_steps, d_model) * 0.02)
        time_layer = nn.TransformerEncoderLayer(
            d_model, heads, 4 * d_model, dropout, "gelu", batch_first=True,
            norm_first=True)
        self.temporal = nn.TransformerEncoder(time_layer, temporal_layers,
                                              enable_nested_tensor=False)
        self.agent_norm = nn.LayerNorm(d_model)
        self.map_projection = nn.Sequential(nn.Linear(6, d_model), nn.LayerNorm(d_model),
                                            nn.GELU(), nn.Linear(d_model, d_model))
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)
        self.latent_cross = nn.MultiheadAttention(d_model, heads, dropout=dropout,
                                                  batch_first=True)
        self.latent_norm = nn.LayerNorm(d_model)
        latent_layer = nn.TransformerEncoderLayer(
            d_model, heads, 4 * d_model, dropout, "gelu", batch_first=True,
            norm_first=True)
        self.latent_transformer = nn.TransformerEncoder(
            latent_layer, latent_layers, enable_nested_tensor=False)
        self.decoder_cross = nn.MultiheadAttention(d_model, heads, dropout=dropout,
                                                    batch_first=True)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_mlp = FeedForward(d_model, dropout)
        self.trajectory_head = nn.Linear(d_model, num_modes * future_steps * 2)
        self.mode_head = nn.Linear(d_model, num_modes)
        nn.init.normal_(self.trajectory_head.weight, std=0.005)
        nn.init.zeros_(self.trajectory_head.bias)

    def forward(self, batch):
        history = batch["history"]
        B, A, T, _ = history.shape
        visible = batch["history_mask"]
        agent_valid = visible[:, :, -1]
        # Normalize metric quantities, not sine/cosine of heading.
        scale = history.new_tensor([50., 50., 15., 15., 1., 1., 10., 5.])
        scaled = history / scale
        encoded = self.agent_projection(scaled) + self.time_embed[:, :T]
        encoded = encoded.reshape(B * A, T, -1)
        masks = visible.reshape(B * A, T).clone()
        masks[~masks.any(-1), -1] = True
        encoded = self.temporal(encoded, src_key_padding_mask=~masks)
        # Current frame is always valid for selected agents.
        agents = encoded[:, -1].reshape(B, A, -1)
        agents = self.agent_norm(agents + self.type_embedding(batch["agent_type"].long()))
        agents = agents * agent_valid.unsqueeze(-1)
        map_tokens = batch["map"].clone()
        map_tokens[..., :2] /= 50.
        if self.use_map:
            maps = self.map_projection(map_tokens)
            map_valid = batch["map_mask"]
        else:
            maps = torch.zeros_like(self.map_projection(map_tokens))
            map_valid = torch.zeros_like(batch["map_mask"])
        if self.use_interactions:
            all_agents, all_valid = agents, agent_valid
        else:
            # Single-ego context only: zero out all other agents.
            all_agents = torch.cat([agents[:, :1], torch.zeros_like(agents[:, 1:])], dim=1)
            all_valid = agent_valid.clone()
            all_valid[:, 1:] = False
        scene_tokens = torch.cat([all_agents, maps], dim=1)
        scene_mask = torch.cat([all_valid, map_valid], dim=1)
        latents = self.latent_queries.expand(B, -1, -1)
        delta, _ = self.latent_cross(latents, scene_tokens, scene_tokens,
                                     key_padding_mask=~scene_mask, need_weights=False)
        latents = self.latent_norm(latents + delta)
        latents = self.latent_transformer(latents)
        context, _ = self.decoder_cross(agents, latents, latents, need_weights=False)
        decoded = self.decoder_norm(agents + context)
        decoded = decoded + self.decoder_mlp(decoded)
        decoded = decoded * agent_valid.unsqueeze(-1)
        corrections = self.trajectory_head(decoded).reshape(
            B, A, self.num_modes, self.future_steps, 2)
        times = torch.arange(1, self.future_steps + 1,
                             device=history.device, dtype=history.dtype).view(1, 1, 1, -1, 1) / 10
        # A history-only turn-rate/acceleration prior reduces long-horizon drift.
        prior = rollout(batch, mode="ctrv")[:, :, None]
        predicted = prior + corrections * times * 0.5
        logits = self.mode_head(decoded)
        return {"trajectories": predicted, "logits": logits,
                "latents": latents, "agent_valid": agent_valid}


def build_model(config):
    return LatentWorldModel(**config)
