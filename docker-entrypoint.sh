#!/usr/bin/env bash
# Fetch the checkpoint on first start, then hand off to the CMD.
#
# Deliberately not a build step: baking the .pt in would put a 20 MB binary
# that changes wholesale on every retrain into an image layer, and would pin
# the image to one checkpoint. Mount a volume at /app/weights to keep the
# download across restarts.
set -euo pipefail

WEIGHTS="${BATTLESIGHT_MODEL:-/app/weights/best.pt}"

if [ ! -f "$WEIGHTS" ]; then
    echo "[entrypoint] no checkpoint at $WEIGHTS; fetching ${DETECSIGHT_WEIGHTS_TAG:-latest}"
    if ! ./scripts/fetch_weights.sh "${DETECSIGHT_WEIGHTS_TAG:-}"; then
        cat >&2 <<'EOF'
[entrypoint] could not fetch the checkpoint.

The service needs one to start. Either give the container network access to
GitHub releases, or mount a checkpoint you already have:

    docker run -p 8000:8000 -v /path/to/weights:/app/weights <image>

EOF
        exit 1
    fi
fi

echo "[entrypoint] starting: $*"
exec "$@"
