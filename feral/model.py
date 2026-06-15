import torch
from torch import nn

from feral.backbones import BackboneAdapter


class AttentionPoolingBlockCustom(nn.Module):
    def __init__(self, embed_dim, num_heads, out_tokens, **kwargs):
        """Build the attention pooling block. If out_tokens > 0, allocate that many learnable
        query tokens; if out_tokens == 0, mean-pooling is used as the query at forward time."""
        super().__init__()
        self.out_tokens = out_tokens
        if out_tokens > 0:
            self.x_q = nn.Parameter(torch.empty(out_tokens, embed_dim))
            nn.init.xavier_uniform_(self.x_q.data)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_x = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """Cross-attend learnable (or mean-pooled) queries over the token sequence x and flatten
        the result. x is (B, N, embed_dim); returns (B * num_queries, embed_dim), where num_queries
        is out_tokens (or 1 when out_tokens == 0)."""
        if self.out_tokens == 0:
            x_q = x.mean(1, keepdim=True)
        else:
            x_q = self.x_q.unsqueeze(0).expand(x.size(0), -1, -1)
        x_q = self.ln_q(x_q)
        x_kv = self.ln_x(x)
        attn_output, _ = self.attn(x_q, x_kv, x_kv, need_weights=False)
        attn_output = attn_output.reshape(-1, x.shape[2])
        return attn_output


class FeralModel(nn.Module):
    def __init__(self,
            backbone,
            num_classes,
            predict_per_item,
            fc_drop_rate,
            freeze_encoder_layers=0,
            pretrained=True,
            gradient_checkpointing=False,
            **kwargs):
        """Assemble the model: a backbone encoder, an attention-pooling projector (out_tokens =
        predict_per_item), batch-norm, dropout, and a linear classification head. Freezes the first
        freeze_encoder_layers backbone layers."""
        super().__init__()
        self.backbone = BackboneAdapter(backbone, pretrained=pretrained, gradient_checkpointing=gradient_checkpointing)
        d = self.backbone.hidden_dim

        self.clip_projector = AttentionPoolingBlockCustom(
            embed_dim=d, num_heads=16, out_tokens=predict_per_item
        )
        self.fc_norm = nn.BatchNorm1d(d)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(d, num_classes)
        self.backbone.freeze_encoder(freeze_encoder_layers)

    def forward(self, x):
        """Run input through backbone, attention pooling, norm/dropout, and head; returns class
        logits of shape (B * predict_per_item, num_classes)."""
        x = self.backbone(x)
        x = self.clip_projector(x)
        x = self.fc_norm(x)
        x = self.head(self.fc_dropout(x))
        return x
