#!/usr/bin/env bash
# Fetch the deployed checkpoint into weights/.
#
#   scripts/fetch_weights.sh            # latest release
#   scripts/fetch_weights.sh v1.0.0     # a specific tag
#
# The checkpoint is a release asset and not tracked in git, because it is a 20 MB
# binary that changes completely on every retrain. Set DETECSIGHT_REPO to pull
# from a fork.
set -u

REPO="${DETECSIGHT_REPO:-24f2006988/Detecsight}"
TAG="${1:-}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/weights"
mkdir -p "$DEST" || exit 1

# Use gh when it is there, since a private repo 404s without a token. curl is
# the fallback for a public repo.
if command -v gh >/dev/null 2>&1; then
    echo "[get ] best.pt via gh (${TAG:-latest})"
    # shellcheck disable=SC2086
    gh release download $TAG --repo "$REPO" --pattern 'best.pt*' \
        --dir "$DEST" --clobber || exit 1
else
    if [ -z "$TAG" ]; then
        echo "[warn] gh not found; a tag is required for the curl fallback" >&2
        echo "       usage: $0 v1.0.0" >&2
        exit 2
    fi
    BASE="https://github.com/$REPO/releases/download/$TAG"
    echo "[get ] best.pt via curl ($TAG)"
    curl -L --fail --retry 5 --retry-delay 5 -o "$DEST/best.pt" "$BASE/best.pt" || exit 1
    curl -L --fail --retry 5 -sS -o "$DEST/best.pt.sha256" "$BASE/best.pt.sha256" || true
fi

# Check the sha256 if it came down too. A truncated checkpoint otherwise fails
# deep inside torch.load with an unhelpful error.
if [ -f "$DEST/best.pt.sha256" ] && command -v sha256sum >/dev/null 2>&1; then
    want=$(cut -d' ' -f1 < "$DEST/best.pt.sha256")
    have=$(sha256sum "$DEST/best.pt" | cut -d' ' -f1)
    if [ "$want" = "$have" ]; then
        echo "[ ok ] sha256 verified"
    else
        echo "[FAIL] sha256 mismatch: want $want, got $have" >&2
        echo "       delete weights/best.pt and re-run." >&2
        exit 1
    fi
fi

ls -l "$DEST"
echo
echo "Ready. Start the service with:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
