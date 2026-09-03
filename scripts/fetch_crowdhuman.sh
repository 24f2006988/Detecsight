#!/usr/bin/env bash
# Download CrowdHuman from the author's Hugging Face mirror (sshao0516/CrowdHuman).
# The official crowdhuman.org Google Drive links 404 as of 2026-09-03.
# Resumable: re-run this script to continue any partial file.
set -u
BASE="https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main"
DEST="/g/fusionsight/datasets/CrowdHuman_raw"
cd "$DEST" || exit 1

# filename:expected_bytes  (sizes read from the HF API tree, 2026-09-03)
FILES="
annotation_val.odgt:23323139
annotation_train.odgt:80017502
CrowdHuman_val.zip:2488658160
CrowdHuman_train01.zip:2970597373
CrowdHuman_train02.zip:3092749718
CrowdHuman_train03.zip:2306357030
"

fail=0
for entry in $FILES; do
    f="${entry%%:*}"; want="${entry##*:}"
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$have" = "$want" ]; then
        echo "[skip] $f already complete ($want bytes)"
        continue
    fi
    echo "[get ] $f  (have ${have}, want ${want})  $(date '+%H:%M:%S')"
    # retry a few times; -C - resumes, --retry handles transient CDN drops
    curl -L -C - --retry 5 --retry-delay 10 --retry-all-errors \
         --connect-timeout 30 -sS -o "$f" "$BASE/$f"
    got=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$got" = "$want" ]; then
        echo "[ ok ] $f  $got bytes  $(date '+%H:%M:%S')"
    else
        echo "[FAIL] $f  got $got, want $want  -- re-run this script to resume"
        fail=1
    fi
done

echo
echo "==== final state ===="
ls -l "$DEST"
[ "$fail" = 0 ] && echo "ALL FILES COMPLETE" || echo "INCOMPLETE -- re-run scripts/fetch_crowdhuman.sh"
exit $fail
