#!/usr/bin/env bash
# Push the current commit to a Hugging Face Space.
#
#   scripts/deploy_space.sh <hf-username>/<space-name>
#   DRY_RUN=1 scripts/deploy_space.sh me/detecsight     # build the copy, don't push
#
# Make the Space first at huggingface.co/new-space with the Docker SDK, and log in
# with `hf auth login --add-to-git-credential`.
#
# It pushes a fresh one-commit copy and not this repo's history. Hugging Face
# rejects any push that contains a file over 10 MB, and the first commit here has
# an old checkpoint in it. A Space also needs a few lines at the top of its README,
# which the GitHub README shouldn't have, so they are added to the copy only.
set -euo pipefail

SPACE="${1:?usage: $0 <hf-username>/<space-name>}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git -C "$SRC" archive HEAD | tar -x -C "$WORK"
cd "$WORK"

{
    printf -- '---\ntitle: DetecSight\nsdk: docker\napp_port: 8000\n---\n\n'
    cat README.md
} > README.new && mv README.new README.md

git init -q -b main
git add -A
git -c user.name="$(git -C "$SRC" config user.name)" \
    -c user.email="$(git -C "$SRC" config user.email)" \
    commit -q -m "demonstration release"

echo "[space] copy ready: $(git ls-files | wc -l) files, $(du -sh . | cut -f1)"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[space] DRY_RUN set, not pushing"
    head -8 README.md
    exit 0
fi

git push --force "https://huggingface.co/spaces/$SPACE" main
echo "[space] pushed. Watch the build at https://huggingface.co/spaces/$SPACE"
