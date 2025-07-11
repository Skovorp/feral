# How to install 
- make sure you have conda
- conda create -n feral python=3.10
- conda activate feral
- cd to feral directory
- install torch that matches your cuda version
- pip install -r requirements.txt

# Label json structure
```
{
  "class_names": {
    "0": "other",
    "1": "self",
    "2": "larvae"
  },
  "labels": {
    "20250320_callow3_larva_10fps_30minrest.mp4": { # filename of the raw video
      "partition": "train", # could be train or val
      "frame_labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, ...] - very long list of integers
    },
    "20250322_callow2_larva_10fps_30minrest.mp4": {
      "partition": "val",
      "frame_labels": [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, ...] - very long list of integers
    }
  }
}
```

# Peter reimplemented our finetuning code
Things to add:
- dropouts & layer drops
- scale lr with layer number
- ? saving, loading checkpoints
- ? gradient accumulation
- ? freezing 