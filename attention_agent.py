"""
Attention Model Agent for Container Stowage Problem
Based on: "Attention, Learn to Solve Routing Problems!" (Kool et al., ICLR 2019)

Simplified single-layer architecture:
- Encoder: N=1 attention layer
- d_h (embedding dim) = 32
- M (attention heads) = 2
- d_ff (feed-forward) = 64
- Batch Normalization
- Logit clipping C = 10
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) as described in the paper.
    """

    def __init__(self, d_model=32, n_heads=2):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_O(context)


class AttentionEncoderLayer(nn.Module):
    """
    Single encoder layer: h_hat = BN(h + MHA(h)), h_new = BN(h_hat + FF(h_hat))
    """

    def __init__(self, d_model=32, n_heads=2, d_ff=64):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        self._init_weights()

    def _init_weights(self):
        for layer in self.ff:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, mask=None):
        # MHA + skip + BN
        mha_out = self.mha(x, x, x, mask)
        x_hat = self.bn1((x + mha_out).transpose(1, 2)).transpose(1, 2)

        # FF + skip + BN
        ff_out = self.ff(x_hat)
        x_new = self.bn2((x_hat + ff_out).transpose(1, 2)).transpose(1, 2)

        return x_new


class AttentionEncoder(nn.Module):
    """
    Encoder with N attention layers.
    """

    def __init__(self, input_dim, d_model=32, n_layers=1, n_heads=2, d_ff=64):
        super().__init__()

        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            AttentionEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # Initialize
        d = input_dim
        nn.init.uniform_(self.input_proj.weight, -1/np.sqrt(d), 1/np.sqrt(d))
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x, mask=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, mask)
        graph_embedding = h.mean(dim=1)
        return h, graph_embedding


class AttentionDecoder(nn.Module):
    """
    Decoder with glimpse mechanism for computing action logits.
    """

    def __init__(self, d_model=32, n_heads=2, clip_logits=10.0):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.clip_logits = clip_logits
        self.d_k = d_model // n_heads

        self.context_proj = nn.Linear(2 * d_model, d_model)

        # Glimpse attention
        self.glimpse_Q = nn.Linear(d_model, d_model, bias=False)
        self.glimpse_K = nn.Linear(d_model, d_model, bias=False)
        self.glimpse_V = nn.Linear(d_model, d_model, bias=False)
        self.glimpse_O = nn.Linear(d_model, d_model, bias=False)

        # Final attention
        self.final_Q = nn.Linear(d_model, d_model, bias=False)
        self.final_K = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in [self.context_proj, self.glimpse_Q, self.glimpse_K,
                       self.glimpse_V, self.glimpse_O, self.final_Q, self.final_K]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, node_embeddings, graph_embedding, target_embedding, action_mask=None):
        batch_size, n_nodes, _ = node_embeddings.shape

        # Context
        context = self.context_proj(torch.cat([graph_embedding, target_embedding], dim=-1))

        # Glimpse attention
        Q = self.glimpse_Q(context).view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.glimpse_K(node_embeddings).view(batch_size, n_nodes, self.n_heads, self.d_k).transpose(1, 2)
        V = self.glimpse_V(node_embeddings).view(batch_size, n_nodes, self.n_heads, self.d_k).transpose(1, 2)

        glimpse_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if action_mask is not None:
            glimpse_scores = glimpse_scores.masked_fill(
                action_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        glimpse_attn = F.softmax(glimpse_scores, dim=-1)
        glimpse_out = torch.matmul(glimpse_attn, V)
        glimpse_out = self.glimpse_O(glimpse_out.transpose(1, 2).contiguous().view(batch_size, self.d_model))

        # Final attention for logits
        final_Q = self.final_Q(glimpse_out)
        final_K = self.final_K(node_embeddings)

        logits = torch.matmul(final_Q.unsqueeze(1), final_K.transpose(-2, -1)).squeeze(1)
        logits = self.clip_logits * torch.tanh(logits / np.sqrt(self.d_model))

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits


class AttentionAgent(nn.Module):
    """
    Attention Model Agent for Container Stowage with PPO.

    Simplified single-layer configuration:
    - d_model: 32
    - n_heads: 2
    - d_ff: 64
    - n_layers: 1
    """

    def __init__(self, envs, d_model=32, n_layers=1, n_heads=2, d_ff=64, clip_logits=10.0):
        super().__init__()

        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = envs.single_action_space.n

        self.n_slot_attrs = 5
        self.n_nodes = obs_dim // self.n_slot_attrs
        self.action_dim = action_dim
        self.d_model = d_model

        # Encoder
        self.encoder = AttentionEncoder(
            input_dim=self.n_slot_attrs,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff
        )

        # Decoder
        self.decoder = AttentionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            clip_logits=clip_logits
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # Target projection
        self.target_proj = nn.Linear(self.n_slot_attrs, d_model)

        self._init_critic()
        self._print_info()

    def _init_critic(self):
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"Attention Agent (Memory-Efficient)")
        print(f"{'='*50}")
        print(f"  Nodes: {self.n_nodes}")
        print(f"  Features per node: {self.n_slot_attrs}")
        print(f"  Action dimension: {self.action_dim}")
        print(f"  Embedding dim (d_h): {self.d_model}")
        print(f"  Encoder layers (N): {len(self.encoder.layers)}")
        print(f"  Attention heads (M): {self.decoder.n_heads}")
        print(f"  FF dimension: {self.encoder.layers[0].ff[0].out_features}")
        print(f"  Total parameters: {total_params:,}")
        print(f"{'='*50}\n")

    def _reshape_obs(self, obs):
        batch_size = obs.shape[0]
        obs_reshaped = obs.view(batch_size, self.n_nodes, self.n_slot_attrs)
        target_info = obs_reshaped[:, -1, :]
        node_features = obs_reshaped[:, :-1, :]
        return node_features.float(), target_info.float()

    def encode(self, obs):
        node_features, target_info = self._reshape_obs(obs)
        node_embeddings, graph_embedding = self.encoder(node_features)
        target_embedding = self.target_proj(target_info)
        return node_embeddings, graph_embedding, target_embedding

    def get_value(self, obs):
        _, graph_embedding, _ = self.encode(obs)
        return self.critic(graph_embedding)

    def get_action_and_value(self, obs, action=None):
        node_embeddings, graph_embedding, target_embedding = self.encode(obs)
        logits = self.decoder(node_embeddings, graph_embedding, target_embedding)
        logits = self._adjust_logits(logits)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(graph_embedding)

    def get_masked_action_and_value(self, obs, action_mask, action=None):
        node_embeddings, graph_embedding, target_embedding = self.encode(obs)
        logits = self.decoder(node_embeddings, graph_embedding, target_embedding)
        logits = self._adjust_logits(logits)

        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = -1e9

        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(graph_embedding)

    def _adjust_logits(self, logits):
        batch_size = logits.shape[0]
        current_dim = logits.shape[1]

        if current_dim == self.action_dim:
            return logits
        elif current_dim > self.action_dim:
            return logits[:, :self.action_dim]
        else:
            padding = torch.full(
                (batch_size, self.action_dim - current_dim),
                -1e9, device=logits.device, dtype=logits.dtype
            )
            return torch.cat([logits, padding], dim=1)