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
    def __init__(self, model_name, num_classes, predict_per_item, fc_drop_rate, 
                 freeze_predictor_layers, freeze_encoder_layers, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        backbone_dim = self.get_backbone_outp_dim()
        self.clip_projector = AttentionPoolingBlockCustom(
            embed_dim=backbone_dim, num_heads=16, out_tokens=predict_per_item
        )
        self.fc_norm = nn.BatchNorm1d(backbone_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(backbone_dim, num_classes)

        self.freeze_model(freeze_predictor_layers, freeze_encoder_layers)

    def get_backbone_outp_dim(self):
        x = torch.zeros((1, 64, 3, 512, 512))
        out = self.model(x).last_hidden_state
        return out.shape[-1]

    def freeze_model(self, freeze_predictor_layers: int, freeze_encoder_layers: int):
        encoder_layers = self.model.encoder.layer
        assert freeze_encoder_layers <= len(encoder_layers), (
            f"Only {len(encoder_layers)} encoder layers available, got freeze_encoder_layers={freeze_encoder_layers}"
        )
        for i in range(freeze_encoder_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False

        predictor_layers = self.model.predictor.layer
        assert freeze_predictor_layers <= len(predictor_layers), (
            f"Only {len(predictor_layers)} predictor layers available, got freeze_predictor_layers={freeze_predictor_layers}"
        )
        for i in range(freeze_predictor_layers):
            for param in predictor_layers[i].parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x).last_hidden_state
        x = self.clip_projector(x)
        x = self.fc_norm(x)
        x = self.head(self.fc_dropout(x))
        return x
