"""Build chair_with_negs dataset:
- Symlink 72 chair-stand videos into /root/data/chair_with_negs/ with canonical filenames
- Symlink 241 FoG negative videos into the same dir
- Write chair_labels_with_negs.json that:
  * keeps original chair labels (z-scored UPDRS arising_from_chair)
  * appends FoG negs with z-scored 0 = (0 - mean) / std
  * splits the FoG negs by patient for val/train
"""
import json, os, glob, random
random.seed(0)

CHAIR_LABELS = "/root/tulip_labels/chair_labels_resplit.json"
CHAIR_RAW    = "/root/data/tulip/_chair_raw"
FOG_RAW      = "/root/data/fog_negs"
OUT_DIR      = "/root/data/chair_with_negs"
OUT_LABELS   = "/root/tulip_labels/chair_labels_with_negs.json"

os.makedirs(OUT_DIR, exist_ok=True)

with open(CHAIR_LABELS) as fp:
    d = json.load(fp)

mean, std = d["normalization"]["mean"][0], d["normalization"]["std"][0]
score_zero_z = (0.0 - mean) / std
print(f"chair z-score for raw=0: {score_zero_z:.4f}  (mean={mean:.4f} std={std:.4f})")

# 1) Symlink chair videos
n_chair_linked = 0
chair_glob = f"{CHAIR_RAW}/Subject_*/Subject_*/20.*chair*/Camera*.mp4"
chair_paths = glob.glob(chair_glob)
print(f"chair videos found via glob: {len(chair_paths)}")
for src in chair_paths:
    parts = src.split("/")
    subj_dirs = [p for p in parts if p.startswith("Subject_")]
    subj_n = int(subj_dirs[0].split("_")[1])
    cam_n = int(os.path.basename(src).replace("Camera", "").replace(".mp4", ""))
    canonical = "subject%02d_camera%d.mp4" % (subj_n, cam_n)
    dst = os.path.join(OUT_DIR, canonical)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        n_chair_linked += 1
print(f"symlinked {n_chair_linked} chair videos -> {OUT_DIR}")

# 2) Symlink FoG negs
n_neg_linked = 0
fog_neg_files = []
for src in glob.glob(f"{FOG_RAW}/*.mp4"):
    name = os.path.basename(src)
    dst = os.path.join(OUT_DIR, name)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        n_neg_linked += 1
    fog_neg_files.append(name)
print(f"symlinked {n_neg_linked} FoG-neg videos; total found: {len(fog_neg_files)}")

# 3) Build labels with negs split by patient
def patient_of(name):
    return name.split("_")[0]

patients = sorted({patient_of(n) for n in fog_neg_files})
random.shuffle(patients)
n_val_p = max(1, len(patients) // 5)
val_patients = set(patients[:n_val_p])
print(f"FoG patients total {len(patients)}, val-only: {len(val_patients)}")
train_negs = [n for n in fog_neg_files if patient_of(n) not in val_patients]
val_negs   = [n for n in fog_neg_files if patient_of(n) in val_patients]
print(f"  -> {len(train_negs)} train negs, {len(val_negs)} val negs")

new_splits = {
    "train": list(d["splits"]["train"]) + train_negs,
    "val":   list(d["splits"]["val"])   + val_negs,
    "test":  d["splits"].get("test", []),
}
new_labels = dict(d["labels"])
for n in fog_neg_files:
    new_labels[n] = [score_zero_z]

out = dict(d)
out["splits"] = new_splits
out["labels"] = new_labels
out["source_note"] = "chair_labels_resplit + 241 FoG negatives at raw_score=0"
with open(OUT_LABELS, "w") as fp:
    json.dump(out, fp, indent=2)
print(f"wrote {OUT_LABELS}")
print("new train:", len(new_splits["train"]), " val:", len(new_splits["val"]))
