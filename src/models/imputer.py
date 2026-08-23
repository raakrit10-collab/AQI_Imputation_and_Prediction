import math
import torch
import torch.nn as nn

# Bidirectional GRU with self-attention mechanism, city embeddings, and day-of-year embeddings
class BRATISeasonal(nn.Module):
    def __init__(self, num_pollutants, num_cities, cfg):
        super().__init__()
        self.city_embed = nn.Embedding(num_cities, cfg.embed_city)
        self.meta_fc = nn.Linear(4, cfg.embed_meta_fc)
        self.use_doy_embed = not getattr(cfg, "no_doy_embed", False)
        time_emb_dim = cfg.embed_time if self.use_doy_embed else 0
        if self.use_doy_embed:
            self.doy_embed = nn.Embedding(367, cfg.embed_time)

        in_dim = num_pollutants * 2 + cfg.embed_city + cfg.embed_meta_fc + time_emb_dim
        hidden_size = cfg.hidden_size
        self.input_proj = nn.Linear(in_dim, hidden_size)
        self.encoder = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.q_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.k_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.v_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_pollutants),
        )

    def forward(self, values, mask, meta, pad_mask=None):
        B, T, P = values.shape
        city_idx = meta[:, 0, 4].long()
        city_e = self.city_embed(city_idx).unsqueeze(1).expand(-1, T, -1)

        meta_e = torch.relu(self.meta_fc(meta[:, :, 0:4]))
        if self.use_doy_embed:
            doy_idx = meta[:, :, 5].long().clamp(0, 366)
            x = torch.cat([values, mask, city_e, meta_e, self.doy_embed(doy_idx)], dim=-1)
        else:
            x = torch.cat([values, mask, city_e, meta_e], dim=-1)

        x = torch.relu(self.input_proj(x))
        enc_out, _ = self.encoder(x)

        Q, K, V = self.q_proj(enc_out), self.k_proj(enc_out), self.v_proj(enc_out)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1])
        if pad_mask is not None:
            scores = scores.masked_fill((pad_mask == 0).unsqueeze(1), float("-inf"))
        context = torch.matmul(torch.softmax(scores, dim=-1), V)

        return self.decoder(torch.cat([enc_out, context], dim=-1))
