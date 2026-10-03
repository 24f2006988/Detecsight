"""Training job manager. Runs scripts/train.py as a subprocess so a long
training run never blocks the event loop."""
import os
import re
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from app import config

# ultralytics puts ANSI escapes in front of its progress rows
ANSI_RE = re.compile(chr(27) + r"\[[0-9;?]*[a-zA-Z]")
EPOCH_RE = re.compile(r"^\s*(\d+)/(\d+)\s")
# a reused run name gets a number added, so read the real directory from the log
SAVE_DIR_RE = re.compile(r"save_dir=([^,]+)")


class TrainingManager:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _kill_tree(proc: subprocess.Popen):
        """Kill the training process and its children. On Windows terminate() leaves the
        dataloader workers running and holding VRAM."""
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            proc.terminate()

    def _reader(self, job_id: str, proc: subprocess.Popen, log_path):
        """Background thread: reads stdout, keeps the tail, tracks the epoch and writes the log."""
        job = self.jobs[job_id]
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                for raw in proc.stdout:
                    line = ANSI_RE.sub("", raw).rstrip()
                    log_file.write(line + "\n")
                    log_file.flush()
                    job["log_tail"].append(line)

                    m = EPOCH_RE.match(line)
                    if m:
                        job["current_epoch"] = int(m.group(1))
                    elif job["save_dir"] is None:
                        d = SAVE_DIR_RE.search(line)
                        if d:
                            job["save_dir"] = d.group(1).strip()
        except Exception as e:  # noqa: BLE001
            # this thread is the only reader of the child's stdout, so if it dies the child
            # blocks forever. Kill it instead.
            job["log_tail"].append(f"[reader error] {e!r}")
            self._kill_tree(proc)

        proc.wait()
        with self._lock:
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            if job["status"] == "cancelled":
                pass
            elif proc.returncode == 0:
                job["status"] = "completed"
                # use the directory ultralytics reported
                base = (Path(job["save_dir"]) if job["save_dir"]
                        else config.RUNS_DIR / job["run_name"])
                best = base / "weights" / "best.pt"
                job["best_weights"] = str(best) if best.exists() else None
            else:
                job["status"] = "failed"

    def start(self, req) -> dict:
        # check and spawn together, or two requests could both start a training
        with self._lock:
            if self._is_training():
                raise RuntimeError("A training job is already running. "
                                   "Only one at a time: the GPU cannot share 8 GB.")

            job_id = uuid.uuid4().hex[:12]
            config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_path = config.LOGS_DIR / f"{job_id}.log"

            # -u so output isn't buffered, or progress wouldn't update until the end
            cmd = [
                sys.executable, "-u", str(config.TRAIN_SCRIPT),
                "--model", req.model,
                "--data", req.data,
                "--epochs", str(req.epochs),
                "--imgsz", str(req.imgsz),
                "--batch", str(req.batch),
                "--name", req.name,
                "--device", config.DEVICE,
            ]

            proc = subprocess.Popen(
                cmd,
                cwd=str(config.BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                # windows pipes decode as cp1252, which can't read ultralytics' progress bars
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            job = {
                "job_id": job_id,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "run_name": req.name,
                "save_dir": None,
                "current_epoch": 0,
                "total_epochs": req.epochs,
                "log_tail": deque(maxlen=40),
                "best_weights": None,
            }
            self.jobs[job_id] = job
            self._procs[job_id] = proc

            threading.Thread(
                target=self._reader, args=(job_id, proc, log_path), daemon=True
            ).start()

        return self.get(job_id)

    def _is_training(self) -> bool:
        """Running only if the process is alive, so stuck bookkeeping can't block the queue."""
        for jid, j in self.jobs.items():
            if j["status"] != "running":
                continue
            proc = self._procs.get(jid)
            if proc is not None and proc.poll() is None:
                return True
        return False

    def is_training(self) -> bool:
        with self._lock:
            return self._is_training()

    def get(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        out = dict(job)
        out["log_tail"] = list(job["log_tail"])
        return out

    def list_jobs(self):
        return [self.get(jid) for jid in list(self.jobs)]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(job_id)
            if proc is None or proc.poll() is not None:
                return False
            self.jobs[job_id]["status"] = "cancelled"
            self._kill_tree(proc)
            return True


trainer = TrainingManager()
