import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from transformers import AutoModelForImageTextToText, AutoModel
from layers_internvideo import AttentiveBlock


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
    
class AttentionPoolingBlockInternvideo(AttentiveBlock):
    def __init__(self, **kwargs):
        kwargs['dim'] = kwargs['embed_dim']
        super().__init__(**kwargs)
        self.x_q = nn.Parameter(torch.empty(kwargs['out_tokens'], kwargs['dim']))
        nn.init.xavier_uniform_(self.x_q.data)

    def forward(self, x):
        x_q = self.x_q.unsqueeze(0).expand(x.size(0), -1, -1)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.reshape(-1, x.shape[2])
        return x

def inject_dropout(module, p):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and 'fc' in name:
            new = nn.Sequential(child, nn.Dropout(p))
            setattr(module, name, new)
        else:
            inject_dropout(child, p)

class HFModel(nn.Module):
    def __init__(self, model_name, num_classes, predict_per_item, fc_drop_rate, freeze_layers, backbone_dropout, head_drop_path, freeze_embeddings, 
                 freeze_predictor_layers, freeze_encoder_layers, **kwargs):
        super().__init__()
        # self.model = AutoModelForImageTextToText.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float32
        # ).model.vision_model # smolvlm

        self.model = AutoModel.from_pretrained(model_name)


        backbone_dim = self.get_backbone_outp_dim()
        self.clip_projector = AttentionPoolingBlockCustom(
            embed_dim=backbone_dim, num_heads=16, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=head_drop_path, 
            norm_layer=partial(nn.LayerNorm, eps=1e-5), out_tokens=predict_per_item
        )
        self.fc_norm = nn.BatchNorm1d(backbone_dim) # nn.LayerNorm(clip_embed_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(backbone_dim, num_classes)

        self.freeze_model(freeze_predictor_layers, freeze_encoder_layers)
        # if backbone_dropout > 0:
        #     inject_dropout(self.model, backbone_dropout)

    def get_backbone_outp_dim(self):
        x = torch.zeros((1, 64, 3, 512, 512))
        out = self.model(x).last_hidden_state
        return out.shape[-1]
    
    # def freeze_model(self, k, freeze_embeddings):
    #     encoder_layers = self.model.encoder.layers
    #     assert k <= len(encoder_layers), f"Only {len(encoder_layers)} layers available, got k={k}"
    #     for i in range(k):
    #         for param in encoder_layers[i].parameters():
    #             param.requires_grad = False
    #     if freeze_embeddings:
    #         for param in self.model.embeddings.parameters():
    #             param.requires_grad = False

    def freeze_model(self, freeze_predictor_layers: int, freeze_encoder_layers: int):
        encoder_layers = self.model.encoder.layer
        assert freeze_encoder_layers <= len(encoder_layers), (
            f"Only {len(encoder_layers)} encoder layers available, got freeze_encoder_layers={freeze_encoder_layers}"
        )
        for i in range(freeze_encoder_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False

        # Freeze predictor layers
        predictor_layers = self.model.predictor.layer
        assert freeze_predictor_layers <= len(predictor_layers), (
            f"Only {len(predictor_layers)} predictor layers available, got freeze_predictor_layers={freeze_predictor_layers}"
        )
        for i in range(freeze_predictor_layers):
            for param in predictor_layers[i].parameters():
                param.requires_grad = False

    def forward(self, x):
        bs, num_frames, channels, _, _ = x.shape

        # x = x.reshape(bs * num_frames, *x.shape[2:])
        x = self.model(x).last_hidden_state
        # x = x.reshape(bs, -1, x.shape[-1])
        x = self.clip_projector(x)
        x = self.fc_norm(x)
        x = self.head(self.fc_dropout(x))
        return x
    
if __name__ == "__main__":
    import yaml
    with open('/home/petr/video_understanding/configs/tarchuna_jepa/camls_vjepa.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    mod = HFModel(model_name=cfg['model_name'], num_classes=4, predict_per_item=64, **cfg['model'])
    print(mod.model)
    print("NEED GRAD")
    for name, param in mod.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    # x = torch.zeros((4, 16, 3, 512, 512))
    # mod.eval()
    # print(mod(x))
    # print(mod(x))
