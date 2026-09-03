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

# Ultralytics prefixes progress rows with ANSI escapes (ESC[K, colours), so
# they must be stripped before the epoch row will match at ^.
ANSI_RE = re.compile(chr(27) + r"\[[0-9;?]*[a-zA-Z]")
EPOCH_RE = re.compile(r"^\s*(\d+)/(\d+)\s")
# Ultralytics increments a reused run name to "<name>2", so the real output
# directory is scraped from its own args dump rather than assumed.
SAVE_DIR_RE = re.compile(r"save_dir=([^,]+)")


class TrainingManager:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _kill_tree(proc: subprocess.Popen):
        """Kill the training process and its children.

        terminate() signals only the launcher. Ultralytics spawns dataloader
        worker processes, which on Windows would be left running and holding
        VRAM after a cancel.
        """
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            proc.terminate()

    def _reader(self, job_id: str, proc: subprocess.Popen, log_path):
        """Runs in a background thread: drains stdout, tails it, scrapes
        the current epoch, and writes the full log to disk."""
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
            # This thread is the only thing draining the child's stdout. If it
            # dies, the pipe fills and the child blocks forever, leaving the job
            # wedged at "running". Kill the child rather than hang.
            job["log_tail"].append(f"[reader error] {e!r}")
            self._kill_tree(proc)

        proc.wait()
        with self._lock:
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            if job["status"] == "cancelled":
                pass
            elif proc.returncode == 0:
                job["status"] = "completed"
                # Trust the directory ultralytics reported; fall back to the
                # conventional one only if the args dump was never seen.
                base = (Path(job["save_dir"]) if job["save_dir"]
                        else config.RUNS_DIR / job["run_name"])
                best = base / "weights" / "best.pt"
                job["best_weights"] = str(best) if best.exists() else None
            else:
                job["status"] = "failed"

    def start(self, req) -> dict:
        # The whole check-then-spawn has to be atomic: two POSTs arriving
        # together would both pass a bare is_training() check and put two
        # trainings on one 8 GB GPU.
        with self._lock:
            if self._is_training():
                raise RuntimeError("A training job is already running. "
                                   "Only one at a time: the GPU cannot share 8 GB.")

            job_id = uuid.uuid4().hex[:12]
            config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_path = config.LOGS_DIR / f"{job_id}.log"

            # -u: without it the child block-buffers stdout into the pipe and the
            # reader thread sees nothing until the run ends, so progress never updates.
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
                # Windows text pipes decode as cp1252, and ultralytics progress
                # bars emit UTF-8 box-drawing bytes that cp1252 cannot map.
                # Without this the reader thread dies on UnicodeDecodeError.
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
        """A job counts as running only if its process is actually alive, so a
        job whose bookkeeping got stuck cannot block the queue forever."""
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
