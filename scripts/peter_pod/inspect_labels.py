import json
from collections import Counter

# auto-gait
d = json.load(open("/root/labels/auto-gait_labels.json"))
print("=== auto-gait ===")
print("class_names:", d["class_names"])
print("is_multilabel:", d["is_multilabel"])
print("splits keys:", list(d["splits"].keys()), "sizes:", {k: len(v) for k, v in d["splits"].items()})
v0 = list(d["labels"].keys())[0]
labels_v0 = d["labels"][v0]
print("sample video", repr(v0), ": type=", type(labels_v0).__name__, "len=", len(labels_v0), "first 10=", labels_v0[:10])
print("  values range:", min(labels_v0), "..", max(labels_v0))
all_label_dist = Counter()
for v in d["labels"].values():
    for x in v: all_label_dist[x] += 1
print("  overall label freqs:", dict(all_label_dist))
print("  patient_id sample:", list(d["patient_id"].items())[:3])
print("  _source:", d.get("_source", "-"))

print()
d = json.load(open("/root/labels/tulip-gait_labels.json"))
print("=== tulip-gait ===")
print("task:", d["task"])
print("target_names:", d["target_names"])
print("normalization:", d["normalization"])
print("source:", d.get("source"))
print("splits keys:", list(d["splits"].keys()), "sizes:", {k: len(v) for k, v in d["splits"].items()})
v0 = list(d["labels"].keys())[0]
print("sample video", repr(v0))
print("  value:", d["labels"][v0])
print("  total videos with labels:", len(d["labels"]))

print()
d = json.load(open("/root/labels/koa-pd-nm-gait_labels.json"))
print("=== koa-pd-nm-gait ===")
print("type:", type(d).__name__)
labels = Counter(d.values())
print("label freqs:", dict(labels))
print("sample:", list(d.items())[:5])
