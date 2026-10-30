#!/usr/bin/env python3
"""Independently check data, label-derived gradients, SGD arithmetic, and run status."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

results, data = map(Path, sys.argv[1:3])
step = json.loads((results / "sgd-step.json").read_text())
state = json.loads((results / "state.json").read_text())
log = (results / "run.log").read_text()
assert state["Status"] == "exited" and state["ExitCode"] == 0 and not state["OOMKilled"]
assert "Final proof size :" in log and "Verification time :" in log
assert not re.search(r"\b(error|failed|wrong|mismatch)\b", log, re.IGNORECASE), log
images = (data / "train-images-idx3-ubyte").read_bytes()
labels = (data / "train-labels-idx1-ubyte").read_bytes()
manifest = json.loads((results / "dataset-manifest.json").read_text())
for name, raw in (("train-images-idx3-ubyte", images), ("train-labels-idx1-ubyte", labels)):
    assert hashlib.sha256(raw).hexdigest() == manifest["files"][name]["sha256"]
assert struct.unpack(">IIII", images[:16]) == (2051, 60000, 28, 28)
assert struct.unpack(">II", labels[:8]) == (2049, 60000)
assert step["Q"] == 15 and step["batch_size"] == 1
f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
reference_losses = []
for i in range(step["batch_size"]):
    index = step["offset"] + i
    label = labels[8 + index]
    assert step["labels"][i] == label
    pixels = images[16 + index * 784:16 + (index + 1) * 784]
    padded = [0] * 1024
    for y in range(28):
        for x in range(28):
            padded[(y + 2) * 32 + x + 2] = int(f32(f32(pixels[y * 28 + x] / 255.0) * 32768))
    assert step["input_q15"][i] == padded, "Input normalization/padding mismatch"
    scores = [x / (1 << 30) for x in step["logits_q30"][i][:10]]
    maximum = max(scores)
    denominator = sum(math.exp(x - maximum) for x in scores)
    loss = maximum + math.log(denominator) - scores[label]
    reference_losses.append(loss)
    expected = [int(f32(f32(math.exp(x - maximum) / denominator - (j == label)) * 32768))
                for j, x in enumerate(scores)] + [0] * 6
    assert step["output_gradient_q15"][i] == expected, "Label-derived softmax gradient mismatch"
    match = re.search(rf"MNIST classification sample={i} label={label} predicted=\d+ cross_entropy=([0-9.eE+-]+)", log)
    assert match and abs(float(match[1]) - loss) < 1e-8, "Cross-entropy mismatch"
    # A different label must change the two affected one-hot gradient slots.
    alternative = (label + 1) % 10
    other = [int(f32(f32(math.exp(x - maximum) / denominator - (j == alternative)) * 32768))
             for j, x in enumerate(scores)] + [0] * 6
    assert other != expected
count = changed = 0
assert [layer["shape"] for layer in step["layers"]] == [
    [4, 1, 5, 5], [16, 4, 5, 5], [128, 16, 5, 5], [32, 128], [16, 32],
]
for layer in step["layers"]:
    before, delta, after = (layer[k] for k in ("before_q15", "update_q15", "after_q15"))
    assert len(before) == len(delta) == len(after) == math.prod(layer["shape"])
    assert all(a == b - d for b, d, a in zip(before, delta, after)), "SGD update mismatch"
    count += len(before)
    changed += sum(d != 0 for d in delta)
assert count == step["parameters"] and changed == step["changed_parameters"] and changed > 0
report = {"passed": True, "dataset": "MNIST train", "offset": step["offset"],
          "labels": step["labels"], "reference_cross_entropy": reference_losses,
          "parameters": count, "changed_parameters": changed,
          "checks": ["IDX headers and SHA256", "real labels", "zero padding and Q15 pixel normalization",
                     "label-dependent softmax gradient", "cross entropy", "SGD subtraction and shapes",
                     "exit status and no reported proof errors"],
          "scope": "Independent host checks; not a proof of loss, dataset membership, or full IVC"}
(results / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
