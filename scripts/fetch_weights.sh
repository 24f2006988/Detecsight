#!/usr/bin/env bash
# Fetch the deployed checkpoint into weights/.
#
#   scripts/fetch_weights.sh            # latest release
#   scripts/fetch_weights.sh v1.0.0     # a specific tag
#
# The checkpoint is a release asset rather than a tracked file: it is 20 MB of
# binary that changes wholesale every time it is retrained, which is what git is
# worst at storing. See ENGINEERING_LOG.md for what each release contains.
set -u

REPO="24f2006988/Detecsight"
TAG="${1:-}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/weights"
mkdir -p "$DEST" || exit 1

# Prefer gh: while the repository is private an unauthenticated download 404s,
# and gh already holds the token. curl is the fallback for a plain clone once
# the repository is public.
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

# Verify, if the published checksum came down with it. A truncated checkpoint
# fails deep inside torch.load with an unhelpful error; catch it here instead.
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
