"""Build finger-tap (HUBU-FIS) with FoG-neg dataset.
- Symlink the 234 HUBU-FIS videos + 241 FoG-neg videos into /root/data/fingertap_with_negs/
- Build hubu-fis_labels_with_negs.json with FoG negs at z-scored 0
"""
import json, os, glob, random
random.seed(0)

HUBU_LABELS = "/root/data/hubu-fis/hubu-fis_regression_labels.json"
HUBU_VIDS = "/root/data/hubu-fis/videos"
FOG_RAW = "/root/data/fog_negs"
OUT_DIR = "/root/data/fingertap_with_negs"
OUT_LABELS = "/root/labels/hubu-fis_labels_with_negs.json"

os.makedirs(OUT_DIR, exist_ok=True)

with open(HUBU_LABELS) as fp:
    d = json.load(fp)
mean, std = d["normalization"]["mean"][0], d["normalization"]["std"][0]
score_zero_z = (0.0 - mean) / std
print("hubu-fis z-score for raw=0:", score_zero_z, " (mean=%.4f std=%.4f)" % (mean, std))

# Symlink HUBU-FIS videos with original names
n_link_h = 0
for src in glob.glob(os.path.join(HUBU_VIDS, "*.mp4")):
    name = os.path.basename(src)
    dst = os.path.join(OUT_DIR, name)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        n_link_h += 1
print("symlinked %d hubu-fis videos" % n_link_h)

# Symlink FoG negs
n_link_n = 0
fog_neg_files = []
for src in glob.glob(os.path.join(FOG_RAW, "*.mp4")):
    name = os.path.basename(src)
    dst = os.path.join(OUT_DIR, name)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        n_link_n += 1
    fog_neg_files.append(name)
print("symlinked %d FoG negs" % n_link_n)

# Split FoG negs by patient
def patient_of(name):
    return name.split("_")[0]
patients = sorted({patient_of(n) for n in fog_neg_files})
random.shuffle(patients)
n_val_p = max(1, len(patients) // 5)
val_patients = set(patients[:n_val_p])
train_negs = [n for n in fog_neg_files if patient_of(n) not in val_patients]
val_negs = [n for n in fog_neg_files if patient_of(n) in val_patients]
print("FoG patients: total %d, val %d  -> train %d, val %d" % (
    len(patients), len(val_patients), len(train_negs), len(val_negs)))

new_splits = {
    "train": list(d["splits"]["train"]) + train_negs,
    "val": list(d["splits"]["val"]) + val_negs,
    "test": d["splits"].get("test", []),
    "inference": d["splits"].get("inference", []),
}
new_labels = dict(d["labels"])
for n in fog_neg_files:
    new_labels[n] = [score_zero_z]

out = dict(d)
out["splits"] = new_splits
out["labels"] = new_labels
out["source_note"] = "hubu-fis_regression_labels + 241 FoG negatives at raw_score=0"
with open(OUT_LABELS, "w") as fp:
    json.dump(out, fp, indent=2)
print("wrote", OUT_LABELS)
print("new train:", len(new_splits["train"]), " val:", len(new_splits["val"]))
