from __future__ import annotations

import math

import torch
from torch import nn


class EhrEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        ehr_net: str = "gru",
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.25,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.ehr_net = ehr_net
        if ehr_net == "gru":
            self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif ehr_net == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif ehr_net == "rnn":
            self.encoder = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif ehr_net == "transformer":
            self.proj = nn.Linear(input_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ffn_multiplier,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        elif ehr_net == "modern_transformer":
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.positional_encoding = SinusoidalPositionEncoding(hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ffn_multiplier,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unsupported EHR encoder: {ehr_net}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ehr_net == "transformer":
            out = self.encoder(self.proj(x))
            return out[:, -1, :]
        if self.ehr_net == "modern_transformer":
            projected = self.proj(x)
            cls = self.cls_token.expand(projected.size(0), -1, -1)
            encoded = self.positional_encoding(torch.cat([cls, projected], dim=1))
            out = self.encoder(encoded)
            return self.norm(out[:, 0, :])
        out, hidden = self.encoder(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[-1]


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / self.hidden_dim)
        )
        pe = torch.zeros(seq_len, self.hidden_dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return x + pe.unsqueeze(0)


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.25) -> None:
        super().__init__()
        self.text_to_ehr = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ehr_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ehr = nn.LayerNorm(hidden_dim)
        self.norm_text = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, ehr: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        ehr_seq = ehr.unsqueeze(1)
        text_seq = text.unsqueeze(1)
        z_ehr, _ = self.text_to_ehr(query=text_seq, key=ehr_seq, value=ehr_seq)
        z_text, _ = self.ehr_to_text(query=ehr_seq, key=text_seq, value=text_seq)
        z_ehr = self.norm_ehr(z_ehr + text_seq).squeeze(1)
        z_text = self.norm_text(z_text + ehr_seq).squeeze(1)
        return self.ffn(torch.cat([z_ehr, z_text], dim=-1))


class MAGGate(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, 1)
        self.adjust = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Parameter(torch.ones(()))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, base: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.gate(torch.cat([base, aux], dim=-1)))
        adjust = self.adjust(weight * aux)
        scale = torch.norm(base, dim=-1, keepdim=True) / torch.clamp(
            torch.norm(adjust, dim=-1, keepdim=True), min=1e-6
        )
        alpha = torch.clamp(scale * self.beta, max=1.0)
        return self.dropout(self.norm(base + alpha * adjust))


class TensorFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear((hidden_dim + 1) * (hidden_dim + 1), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch_size = first.size(0)
        ones = torch.ones((batch_size, 1), device=first.device, dtype=first.dtype)
        first = torch.cat([first, ones], dim=-1)
        second = torch.cat([second, ones], dim=-1)
        outer = torch.bmm(first.unsqueeze(2), second.unsqueeze(1))
        return self.net(outer.flatten(1))


class Concatenation(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([first, second], dim=-1))


class TokenTransformerFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.25,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_multiplier,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.pool = nn.Linear(hidden_dim, 1)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        ehr: torch.Tensor,
        note: torch.Tensor,
        summary: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = torch.stack([ehr, note, summary], dim=1) + self.modality_embedding
        if token_mask is None:
            token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        scores = self.pool(encoded).squeeze(-1).masked_fill(~token_mask, -torch.inf)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1)
        residual = (tokens * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return self.norm(self.out(pooled) + residual)


class EMERGE(nn.Module):
    def __init__(
        self,
        input_ehr_dim: int,
        input_note_dim: int,
        input_summary_dim: int,
        hidden_dim: int = 128,
        ehr_net: str = "gru",
        text_fusion: str = "concat",
        modality_fusion: str = "ours",
        use_modality: str = "ehr_note_summary",
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_multiplier: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.text_fusion = text_fusion
        self.modality_fusion = modality_fusion
        self.use_modality = use_modality

        self.ehr_encoder = EhrEncoder(
            input_ehr_dim,
            hidden_dim,
            ehr_net=ehr_net,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
        )
        self.note_proj = nn.Linear(input_note_dim, hidden_dim)
        self.summary_proj = nn.Linear(input_summary_dim, hidden_dim)
        self.text_concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.text_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.text_alpha = nn.Parameter(torch.tensor(0.5))
        self.text_mag = MAGGate(hidden_dim, dropout=dropout)

        if modality_fusion == "ours":
            self.fusion = CrossAttentionFusion(hidden_dim, num_heads=num_heads, dropout=dropout)
        elif modality_fusion == "token_transformer":
            self.fusion = TokenTransformerFusion(
                hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                ffn_multiplier=ffn_multiplier,
            )
        elif modality_fusion == "mag":
            self.fusion = MAGGate(hidden_dim, dropout=dropout)
        elif modality_fusion == "concat":
            self.fusion = Concatenation(hidden_dim)
        elif modality_fusion == "tf":
            self.fusion = TensorFusion(hidden_dim)
        else:
            raise ValueError(f"Unsupported modality fusion: {modality_fusion}")

    def forward(
        self,
        x_ehr: torch.Tensor,
        x_note_embedding: torch.Tensor,
        x_summary_embedding: torch.Tensor,
    ) -> torch.Tensor:
        ehr_embedding = self.ehr_encoder(x_ehr)
        note_embedding = self.note_proj(x_note_embedding)
        summary_embedding = self.summary_proj(x_summary_embedding)

        if self.text_fusion == "note_only":
            text_embedding = note_embedding
        elif self.text_fusion == "summary_only":
            text_embedding = summary_embedding
        elif self.text_fusion == "add":
            text_embedding = note_embedding + summary_embedding
        elif self.text_fusion == "concat":
            text_embedding = self.text_concat_proj(torch.cat([note_embedding, summary_embedding], dim=-1))
        elif self.text_fusion == "gated":
            gate = torch.sigmoid(self.text_gate(torch.cat([note_embedding, summary_embedding], dim=-1)))
            text_embedding = gate * note_embedding + (1.0 - gate) * summary_embedding
        elif self.text_fusion == "adaptive":
            alpha = torch.clamp(self.text_alpha, 0.0, 1.0)
            text_embedding = alpha * note_embedding + (1.0 - alpha) * summary_embedding
        elif self.text_fusion == "mag":
            text_embedding = self.text_mag(note_embedding, summary_embedding)
        else:
            raise ValueError(f"Unsupported text fusion: {self.text_fusion}")

        if self.modality_fusion == "token_transformer":
            fused = self.fusion(
                ehr_embedding,
                note_embedding,
                summary_embedding,
                self._modality_mask(x_ehr.size(0), x_ehr.device),
            )
        elif self.modality_fusion == "ours":
            fused = self.fusion(ehr_embedding, text_embedding)
        else:
            fused = self.fusion(ehr_embedding, text_embedding)

        if self.use_modality == "ehr_note_summary":
            return fused
        if self.use_modality == "ehr_only":
            return ehr_embedding
        if self.use_modality == "note_only":
            return note_embedding
        if self.use_modality == "summary_only":
            return summary_embedding
        if self.use_modality in {"ehr_note", "ehr_summary"}:
            return fused
        if self.use_modality == "note_summary":
            return text_embedding
        raise ValueError(f"Unsupported modality selection: {self.use_modality}")

    def _modality_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        active = {
            "ehr_note_summary": (True, True, True),
            "ehr_note": (True, True, False),
            "ehr_summary": (True, False, True),
        }.get(self.use_modality, (True, True, True))
        return torch.tensor(active, device=device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1)


class MultitaskHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.outcome = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.readmission = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        return torch.cat([self.outcome(x), self.readmission(x)], dim=1)


class EMERGEPredictor(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        hidden_dim = int(kwargs["hidden_dim"])
        dropout = float(kwargs.get("dropout", 0.25))
        self.backbone = EMERGE(**kwargs)
        self.head = MultitaskHead(hidden_dim, dropout=dropout)

    def forward(
        self,
        x_ehr: torch.Tensor,
        x_note_embedding: torch.Tensor,
        x_summary_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.backbone(x_ehr, x_note_embedding, x_summary_embedding))
