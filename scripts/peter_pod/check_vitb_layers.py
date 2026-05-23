import sys; sys.path.insert(0, "/root/feral")
from feral.backbones import BackboneAdapter

bb = BackboneAdapter("vjepa2_1_vitb_384", pretrained=False)
n = bb.num_encoder_layers()
print(f"vjepa2_1_vitb_384: {n} encoder layers")
total_params = sum(p.numel() for p in bb.parameters())
print(f"total params: {total_params/1e6:.1f} M")
