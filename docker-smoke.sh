#!/bin/sh
# Run one real MNIST LeNet batch with bounded resources.
set -eu
cd "$(dirname "$0")"
DATA_DIR=${DATA_DIR:-"$(pwd)/data/mnist"}
python3 download-mnist.py "$DATA_DIR"
DATA_DIR=$(cd "$DATA_DIR" && pwd)
RESULTS_DIR=${RESULTS_DIR:-"$(pwd)/docker-results/mnist-lenet-b1-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$RESULTS_DIR"
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)

docker build --platform linux/amd64 -t kaizen:local .
cp "$DATA_DIR/manifest.json" "$RESULTS_DIR/dataset-manifest.json"
docker image inspect --format '{{json .}}' kaizen:local >"$RESULTS_DIR/image.json"
container=$(docker create --platform linux/amd64 \
    --memory=6g --memory-swap=6g --cpus=2 \
    --mount "type=bind,src=$DATA_DIR,dst=/data/mnist,readonly" \
    --mount "type=bind,src=$RESULTS_DIR,dst=/results" \
    --entrypoint /bin/sh kaizen:local -c \
    'exec /usr/bin/time -v timeout --signal=TERM --kill-after=10s 10m stdbuf -oL -eL ./main LENET 1 1 -1 1 /data/mnist 0 /results/sgd-step.json')
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Running real MNIST LeNet batch=1; logs: %s/run.log\n' "$RESULTS_DIR"
docker start "$container" >/dev/null
docker wait "$container" >/dev/null
docker logs "$container" >"$RESULTS_DIR/run.log" 2>&1
docker inspect --format '{{json .State}}' "$container" >"$RESULTS_DIR/state.json"
status=$(docker inspect --format '{{.State.ExitCode}}' "$container")
printf 'Container exit code: %s\n' "$status"
tail -30 "$RESULTS_DIR/run.log"
if [ "$status" -eq 0 ]; then
    python3 validate-mnist-run.py "$RESULTS_DIR" "$DATA_DIR"
fi
exit "$status"
