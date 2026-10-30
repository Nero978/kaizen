#!/usr/bin/env python3
"""Fetch the real MNIST training split and verify the torchvision checksums."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys
import urllib.request

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data/mnist"
BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte": ("f68b3c2dcbeaaa9fbdd348bbdeb94873", 16 + 60000 * 28 * 28),
    "train-labels-idx1-ubyte": ("d53e105ee54ea40749a09fcbcd1e9432", 8 + 60000),
}
ROOT.mkdir(parents=True, exist_ok=True)
manifest = {"dataset": "MNIST", "split": "train", "count": 60000, "files": {}}
for name, (md5, length) in FILES.items():
    archive = ROOT / (name + ".gz")
    url = BASE + archive.name
    if not archive.exists():
        print(f"Downloading {url}", flush=True)
        partial = archive.with_suffix(".gz.part")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        partial.replace(archive)
    compressed = archive.read_bytes()
    if hashlib.md5(compressed).hexdigest() != md5:
        raise SystemExit(f"MNIST checksum mismatch: {archive}")
    raw = gzip.decompress(compressed)
    if len(raw) != length:
        raise SystemExit(f"MNIST IDX size mismatch: {name}")
    expected = (2051, 60000, 28, 28) if "images" in name else (2049, 60000)
    if struct.unpack(">" + "I" * len(expected), raw[:4 * len(expected)]) != expected:
        raise SystemExit(f"MNIST IDX header mismatch: {name}")
    (ROOT / name).write_bytes(raw)
    manifest["files"][name] = {
        "url": url, "archive_md5": md5,
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
    }
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Verified MNIST training images and labels: {ROOT}")
