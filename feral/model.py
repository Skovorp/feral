import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from transformers import AutoModelForImageTextToText


class AttentionPoolingBlockCustom(nn.Module):
    def __init__(self, embed_dim, num_heads, out_tokens, **kwargs):
        super().__init__()
        self.out_tokens = out_tokens
        if out_tokens > 0:
            self.x_q = nn.Parameter(torch.empty(out_tokens, embed_dim))
            nn.init.xavier_uniform_(self.x_q.data)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)


    def forward(self, x):
        if self.out_tokens == 0:
            x_q = x.mean(1, keepdim=True)
        else:
            x_q = self.x_q.unsqueeze(0).expand(x.size(0), -1, -1)
        x_kv = x
        attn_output, _ = self.attn(x_q, x_kv, x_kv, need_weights=False)
        attn_output = attn_output.reshape(-1, x.shape[2])
        return attn_output


class HFModel(nn.Module):
    def __init__(self, model_name, num_classes, predict_per_item, **kwargs):
        super().__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).model.vision_model
        backbone_dim = self.get_backbone_outp_dim()
        self.clip_projector = AttentionPoolingBlockCustom(
            embed_dim=backbone_dim, num_heads=16, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.1, 
            norm_layer=partial(nn.LayerNorm, eps=1e-5), out_tokens=predict_per_item
        )
        self.fc_norm = nn.BatchNorm1d(backbone_dim) # nn.LayerNorm(clip_embed_dim)
        # self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.fc_dropout = nn.Identity()
        self.head = nn.Linear(backbone_dim, num_classes)

    def get_backbone_outp_dim(self):
        x = torch.zeros((16, 3, 512, 512))
        out = self.model(x).last_hidden_state
        return out.shape[-1]

    def forward(self, x):
        bs, num_frames, channels, _, _ = x.shape

        x = x.reshape(bs * num_frames, *x.shape[2:])
        x = self.model(x).last_hidden_state
        x = x.reshape(bs, -1, x.shape[-1])
        x = self.clip_projector(x)
        x = self.fc_norm(x)
        x = self.head(self.fc_dropout(x))
        return x
    
if __name__ == "__main__":
    mod = HFModel('HuggingFaceTB/SmolVLM2-2.2B-Instruct', 4)
    mod.get_backbone_outp_dim()