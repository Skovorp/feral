import torch
from torch import nn
from transformers import AutoModel

class AttentionPoolingBlockCustom(nn.Module):
    def __init__(self, embed_dim, num_heads, out_tokens, **kwargs):
        super().__init__()
        self.out_tokens = out_tokens
        if out_tokens > 0:
            self.x_q = nn.Parameter(torch.empty(out_tokens, embed_dim))
            nn.init.xavier_uniform_(self.x_q.data)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_x = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        if self.out_tokens == 0:
            x_q = x.mean(1, keepdim=True)
        else:
            x_q = self.x_q.unsqueeze(0).expand(x.size(0), -1, -1)
        x_q = self.ln_q(x_q)
        x_kv = self.ln_x(x)
        attn_output, _ = self.attn(x_q, x_kv, x_kv, need_weights=False)
        attn_output = attn_output.reshape(-1, x.shape[2])
        return attn_output

class HFModel(nn.Module):
    def __init__(self, 
            model_name, 
            num_classes, 
            predict_per_item, 
            fc_drop_rate, 
            freeze_encoder_layers=0,
            **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        backbone_dim = 1024
        self.model.predictor = None

        self.clip_projector = AttentionPoolingBlockCustom(
            embed_dim=backbone_dim, num_heads=16, out_tokens=predict_per_item
        )
        self.fc_norm = nn.BatchNorm1d(backbone_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(backbone_dim, num_classes)
        self.freeze_model(freeze_encoder_layers)

    def freeze_model(self, freeze_encoder_layers: int):
        def freeze(m):
            for param in m.parameters():
                param.requires_grad = False

        freeze(self.model.encoder.embeddings)
        assert freeze_encoder_layers <= len(self.model.encoder.layer), (
            f"Only {len(self.model.encoder.layer)} encoder layers available, got freeze_encoder_layers={freeze_encoder_layers}"
        )
        for i in range(freeze_encoder_layers):
            freeze(self.model.encoder.layer[i])

    def forward(self, x):
        batch, frames, channels, height, width = x.shape
        x = self.model(x, skip_predictor=True).last_hidden_state
        x = self.clip_projector(x)
        x = self.fc_norm(x)
        x = self.head(self.fc_dropout(x))
        return x
