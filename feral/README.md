# How to install 
- make sure you have conda
- conda create -n feral python=3.10
- conda activate feral
- cd to feral directory
- install torch that matches your cuda version
- pip install -r requirements.txt

# Peter reimplemented our finetuning code
Things to add:
- dropouts & layer drops
- scale lr with layer number
- ? saving, loading checkpoints
- ? gradient accumulation
- ? freezing 