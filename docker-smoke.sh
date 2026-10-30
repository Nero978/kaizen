#!/bin/sh
# Build and run the upstream synthetic LeNet batch-1 benchmark with bounded resources.
set -eu
cd "$(dirname "$0")"
RESULTS_DIR=${RESULTS_DIR:-"$(pwd)/docker-results/lenet-b1-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$RESULTS_DIR"
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)

docker build --platform linux/amd64 -t kaizen:local .
container=$(docker create --platform linux/amd64 \
    --memory=6g --memory-swap=6g --cpus=2 \
    --entrypoint /bin/sh kaizen:local -c \
    'exec /usr/bin/time -v timeout --signal=TERM --kill-after=10s 10m stdbuf -oL -eL ./main LENET 1 1 -1 1')
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Running LeNet batch=1; logs: %s/run.log\n' "$RESULTS_DIR"
docker start -a "$container" >"$RESULTS_DIR/run.log" 2>&1 || true
docker inspect --format '{{json .State}}' "$container" >"$RESULTS_DIR/state.json"
status=$(docker inspect --format '{{.State.ExitCode}}' "$container")
printf 'Container exit code: %s\n' "$status"
tail -30 "$RESULTS_DIR/run.log"
exit "$status"
