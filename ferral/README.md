# Peter reimplemented our finetuning code
Things to add:
- EMA
- saving, loading checkpoints
- set size of frames (currently 512)
- gradient accumulation
- scale lr with layer number
- dropouts & layer drops
- Mixup
- freezing 
- augmentations
- n predictions for each frame, packed target