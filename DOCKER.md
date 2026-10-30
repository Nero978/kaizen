# Local AMD64 Docker experiment

This image builds the upstream C++ implementation inside Ubuntu 22.04.
On Apple Silicon, always select `linux/amd64`; the source contains x86
assembly and SIMD instructions. Execution depends on the Docker translation
backend supporting those instructions.

From this directory, run:

```sh
sh docker-smoke.sh
```

The script builds `kaizen:local` and runs `./main LENET 1 1 -1 1`:
LeNet, batch size 1, one input channel, no SHA tree levels, Orion commitment.
It limits the container to 2 CPUs and 6 GiB RAM, disables container swap,
and stops the process after 10 minutes. Docker Desktop needs more than
6 GiB allocated to its Linux VM to leave room for the VM itself.

Logs and GNU time's resource measurements are saved under
`docker-results/lenet-b1-<timestamp>/run.log`. The container exit state,
including `OOMKilled`, is saved as `state.json`; the script removes its
own container afterward. Override the output directory using `RESULTS_DIR`.
Exit 124 indicates a timeout; inspect the state and log for other failures.

To build or run manually:

```sh
docker build --platform linux/amd64 -t kaizen:local .
docker run --rm --platform linux/amd64 --memory=6g --memory-swap=6g --cpus=2 kaizen:local
```

The manual command has no time limit; the smoke script is preferred for a
first run. Other upstream arguments can be passed after `kaizen:local`.

This is a synthetic benchmark: the upstream program generates random input,
weights, and output gradients. It does not load MNIST, train to convergence,
or provide a complete multi-step IVC implementation. Successful execution
only establishes that this benchmark path runs under the tested environment.
Emulated execution times should not be compared directly with the paper's
native x86 server timings.

## Verified local result (2026-10-30)

Tested on the Apple M3 MacBook Air with 16 GB host RAM, Docker Desktop
4.97.0, an ARM64 Linux VM with about 7.75 GiB allocated, and AMD64 translation.
No upstream C++ or CMake changes were required.

The smoke script completed successfully with exit code 0 and
`OOMKilled=false`. The log contained no reported errors.

| Measurement | Single observed run |
| --- | ---: |
| Wall-clock time | 52.30 seconds |
| User CPU time | 51.69 seconds |
| Maximum process RSS (GNU time) | 1,843,668 KiB, approximately 1.76 GiB |
| Program-reported verification time | 0.015968 seconds |
| Program-reported final proof size estimate | 251.90625 KiB |

RSS measures the program, not total Docker VM or host memory. The proof-size
value is the upstream `proof_size()` estimate, not a measured serialized
proof file. This is one smoke run, not a statistical benchmark.

Raw evidence is in `docker-results/lenet-b1-20261030-174059/run.log` and
`state.json`. The run started at 17:41:44 and finished at 17:42:36 Singapore
time. Tested image ID:
`sha256:e2d539358995f32b2926996a0280d704600496b013c0005319b18bc70e59a2a8`.
