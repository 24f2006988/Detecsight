#!/usr/bin/env bash
# Resume a training run until it actually finishes, surviving OOM kills.
#
#   bash scripts/train_until_done.sh battlesight_blend 6 [workers]
#
# WHY THIS EXISTS. On this machine a long run at imgsz 1280 sits at roughly
# 16 GB of commit charge against a 44.5 GB limit, which is fine on its own --
# but a browser left open grows several GB over an evening and pushes the total
# past the watchdog's threshold. The run then dies mid-epoch. Ultralytics writes
# last.pt at the END of each epoch, so each kill costs at most one epoch, and
# `--resume` restores optimizer state, best_fitness and every hyperparameter
# from the checkpoint. That makes the failure cheap and completely recoverable
# -- it just needs someone to type the command again, which is what this loop
# replaces.
#
# It refuses to spin: if an attempt does not increase the completed-epoch count
# in results.csv, that is a real failure rather than a memory kill, so it stops
# and says so instead of retrying forever.
set -u

NAME="${1:?usage: train_until_done.sh <run-name> <total-epochs> [workers]}"
TOTAL="${2:?need the epoch count the run was started with}"
WORKERS="${3:-1}"
MAX_ATTEMPTS=40

RUN="runs/detect/$NAME"
CSV="$RUN/results.csv"
PY="G:/fusionsight/.venv/Scripts/python.exe"

completed() {  # highest epoch number recorded, 0 if none
    [ -f "$CSV" ] || { echo 0; return; }
    awk -F, 'NR>1 && $1+0>m {m=$1+0} END{print m+0}' "$CSV"
}

echo "[loop] $NAME: target $TOTAL epochs, workers=$WORKERS"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    before=$(completed)
    if [ "$before" -ge "$TOTAL" ]; then
        echo "[loop] done: $before/$TOTAL epochs complete"
        exit 0
    fi

    echo "[loop] attempt $attempt -- $before/$TOTAL epochs done, resuming $(date '+%H:%M:%S')"
    PYTHONPATH=. "$PY" scripts/train.py --resume --name "$NAME" --workers "$WORKERS"
    rc=$?

    after=$(completed)
    echo "[loop] attempt $attempt ended rc=$rc, epochs $before -> $after"

    if [ "$after" -ge "$TOTAL" ]; then
        echo "[loop] done: $after/$TOTAL epochs complete"
        exit 0
    fi
    if [ "$after" -le "$before" ]; then
        # No progress. A memory kill always costs at most the epoch in flight,
        # so it still advances eventually; standing still twice means something
        # else is wrong and retrying would only hide it.
        echo "[loop] STOPPING: attempt $attempt made no progress (still $after epochs)."
        echo "[loop] This is not an out-of-memory kill -- read the log in logs/."
        exit 1
    fi
    sleep 10
done
echo "[loop] gave up after $MAX_ATTEMPTS attempts at $(completed)/$TOTAL epochs"
exit 1
