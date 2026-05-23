"""Build walking_with_negs dataset:
- Append 241 FoG negative videos to the unified gait labels at z-scored 0.
- Symlink each FoG neg into /root/data_precrop_with_negs/ (mirror the existing precrop tree).
"""
import json, os, glob, random
random.seed(0)

GAIT_LABELS = "/root/labels/feral_gait_labels.json"
GAIT_PRECROP_DIR = "/root/data_precrop"
FOG_RAW = "/root/data/fog_negs"
OUT_DIR = "/root/data_precrop_with_negs"
OUT_LABELS = "/root/labels/feral_gait_labels_with_negs.json"

os.makedirs(OUT_DIR, exist_ok=True)

with open(GAIT_LABELS) as fp:
    d = json.load(fp)
mean, std = d["normalization"]["mean"][0], d["normalization"]["std"][0]
score_zero_z = (0.0 - mean) / std
print("gait z-score for raw=0:", score_zero_z, " (mean=%.4f std=%.4f)" % (mean, std))

# Symlink the entire existing data_precrop tree's contents into OUT_DIR
# by mirroring the subdirs.
n_link = 0
for sub in os.listdir(GAIT_PRECROP_DIR):
    src_sub = os.path.join(GAIT_PRECROP_DIR, sub)
    dst_sub = os.path.join(OUT_DIR, sub)
    if not os.path.exists(dst_sub):
        os.symlink(src_sub, dst_sub)
        n_link += 1
print("symlinked %d subdirs of original precrop tree" % n_link)

# Add a "fog_negs" subdir with symlinks to FoG-neg videos
fog_neg_dst = os.path.join(OUT_DIR, "fog_negs")
os.makedirs(fog_neg_dst, exist_ok=True)
fog_neg_files = []
n_neg_linked = 0
for src in glob.glob(os.path.join(FOG_RAW, "*.mp4")):
    name = os.path.basename(src)
    dst = os.path.join(fog_neg_dst, name)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        n_neg_linked += 1
    fog_neg_files.append("fog_negs/" + name)
print("symlinked %d FoG-neg videos, total %d" % (n_neg_linked, len(fog_neg_files)))

# Split by patient (PDFE\d+) for FoG negs; vlogs become "vlog__"
def patient_of(name):
    base = name.split("/")[-1]
    return base.split("_")[0]
patients = sorted({patient_of(n) for n in fog_neg_files})
random.shuffle(patients)
n_val_p = max(1, len(patients) // 5)
val_patients = set(patients[:n_val_p])
print("FoG patients total %d, val %d" % (len(patients), len(val_patients)))
train_negs = [n for n in fog_neg_files if patient_of(n) not in val_patients]
val_negs = [n for n in fog_neg_files if patient_of(n) in val_patients]
print("  train negs %d, val negs %d" % (len(train_negs), len(val_negs)))

new_splits = {
    "train": list(d["splits"]["train"]) + train_negs,
    "val": list(d["splits"]["val"]) + val_negs,
    "test": d["splits"].get("test", []),
    "inference": d["splits"].get("inference", []),
}
new_labels = dict(d["labels"])
for n in fog_neg_files:
    new_labels[n] = [score_zero_z]

# frame_mask: original videos have it; FoG-negs we assume entire-video valid
new_frame_mask = dict(d.get("frame_mask", {}))
# We don't know frame count of FoG vids without probing; leave them out -> dataset.py treats absence as all-valid

out = dict(d)
out["splits"] = new_splits
out["labels"] = new_labels
out["frame_mask"] = new_frame_mask
out["source_note"] = "feral_gait_labels + 241 FoG negatives at raw_score=0"
with open(OUT_LABELS, "w") as fp:
    json.dump(out, fp, indent=2)
print("wrote", OUT_LABELS)
print("new train:", len(new_splits["train"]), " val:", len(new_splits["val"]))
