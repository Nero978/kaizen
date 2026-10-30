# Real MNIST LeNet experiment in Docker

Run one real MNIST training sample through the repository's LeNet forward pass,
label-derived backward pass, existing proof/aggregation/verification path, and
save a host-side SGD checkpoint. This is one batch of size 1, not a full epoch.

## Run

From this directory:

```sh
sh docker-smoke.sh
```

Requirements: Docker Desktop running with more than 6 GiB assigned to its VM,
Python 3.8+ on the host, and internet access for the first dataset/image download.
The script downloads the MNIST training images and labels, verifies their archive
MD5 checksums against torchvision's published values, validates IDX headers and
lengths, and records SHA256 hashes. Files are cached in `data/mnist/` and mounted
read-only into the container. Data and generated results are excluded from Git.

Dataset source: [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist).
Checksum reference: [torchvision MNIST resources](https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html).

The script builds `kaizen:local` for `linux/amd64` and executes:

```sh
./main LENET 1 1 -1 1 /data/mnist 0 /results/sgd-step.json
```

Arguments select LeNet, batch size 1, one channel, no SHA tree levels, Orion
commitment, the real MNIST directory, and zero-based sample offset 0. The final
argument is the SGD checkpoint destination. The container is limited to 2 CPUs,
6 GiB RAM, no container swap, and 10 minutes. Its own container is removed after
the run. Exit 124 means timeout; `state.json` records OOM and exit status.
Missing or invalid MNIST data causes an error rather than a random-data fallback.
The upstream five-argument CLI remains available for legacy synthetic experiments;
the Docker default and this script now select real data.

Results are saved under `docker-results/mnist-lenet-b1-<timestamp>/`.
Set `DATA_DIR` or `RESULTS_DIR` to override those locations. Recheck a saved run:

```sh
python3 validate-mnist-run.py docker-results/mnist-lenet-b1-20261030-183439 data/mnist
```

## Data and model details

- Split: MNIST training set (60,000 images available); this run uses index 0,
  whose true label is **5**. Raw pixel sum is 27,525 with 166 nonzero pixels.
- Preprocessing: normalize uint8 pixels by 255, quantize to Q15, then zero-pad
  28x28 to 32x32 with two pixels on each side.
- Keep the repository's LeNet layout: convolution shapes `[4,1,5,5]`,
  `[16,4,5,5]`, `[128,16,5,5]`, then dense shapes `[32,128]`, `[16,32]`.
  This has 57,508 weights and is not an exact replacement for a standard
  framework LeNet or the paper's stated 61,706-parameter model.
- The first 10 final logits represent digits 0-9; the six extra output slots
  have zero loss gradient. Final dense logits are interpreted as Q30.
- Compute stable 10-class softmax and cross entropy on the host, then quantize
  `softmax(logits) - one_hot(label)` to Q15 for the existing backward pass.
- Initial weights are random, as expected for a new model; the Linux RNG seed
  is fixed at 1. Input images, labels, and output gradients are real-data based.
- Use the upstream quantized learning rate: requested 0.01, represented as
  327/32768. Save each parameter before and after subtracting the upstream
  learning-rate-scaled gradient. This is a host-side update checkpoint.
- The program checks the actual serialized first-convolution input against the
  loaded MNIST pixels. The input-hash path also hashes those actual pixels and
  the true label instead of generating another random sample.

## Verified result: 2026-10-30

Hardware: Apple M3 MacBook Air, 16 GB host RAM. Docker Desktop 4.97.0, ARM64
Linux VM with about 7.75 GiB RAM, running the AMD64 image through translation.
The successful run started at 18:35:21 and finished at 18:36:12 Singapore time.

| Measurement | Single observed run |
| --- | ---: |
| Wall-clock time | 50.76 seconds |
| User CPU time | 50.19 seconds |
| Peak process RSS (GNU time) | 1,843,648 KiB, approximately 1.76 GiB |
| Program-reported verification time | 0.017480 seconds |
| Program-reported proof-size estimate | 251.90625 KiB |
| Cross entropy before the update | 2.792521596 |
| Prediction before the update / true label | 9 / 5 |
| Updated parameters / total | 33,081 / 57,508 |
| Exit code / OOM killed | 0 / false |
| Independent validation | Passed |

Evidence is in `docker-results/mnist-lenet-b1-20261030-183439/`:

- `run.log`: complete program output and GNU time measurements.
- `state.json`: container status, timestamps, and OOM status.
- `dataset-manifest.json`: download URLs, archive checksums, raw SHA256 hashes.
- `sgd-step.json`: real normalized inputs, labels, logits, output gradients,
  layer shapes, old weights, learning-rate-scaled updates, and new weights.
- `validation.json`: independent Python checks of dataset integrity, input
  normalization/padding, label-dependent gradient, loss, SGD subtraction,
  parameter counts, and successful execution without reported proof errors.
- `image.json`: tested image metadata.

Image ID: `sha256:01db56299f1cc73b4e9af89823c9259a374a7dce6c2c21287ff275de01a35a16`.

## Interpretation

The existing proof path completed on a real MNIST input with its label-derived
output gradient. The softmax/loss calculation and the final model update are
checked on the host; they have not been added as new cryptographic proof circuits.
The data-hash trace is not a membership proof against a commitment to all 60,000
samples. The upstream aggregation still reuses witnesses and does not implement
complete multi-step IVC. Its quantization and activation behavior are retained.
This result establishes a real-data single-batch run, not a complete reproduction
of the paper's end-to-end zkPoT or a trained classifier's accuracy.

RSS excludes Docker VM overhead. Proof size is the program's estimate, not a
serialized proof-file measurement. Timings are a single translated execution,
not directly comparable with native x86 server benchmarks.
