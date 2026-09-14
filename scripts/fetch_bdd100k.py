"""Download the BDD100K detection subset this project trains ground-level
vehicles on. Resumable: re-run after any interruption and it fetches only what
is missing.

See ENGINEERING_LOG.md for the measurements behind this.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPO_ID = "dgural/bdd100k"
API = f"https://huggingface.co/api/datasets/{REPO_ID}"
RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
UA = {"User-Agent": "detecsight-fetch/1.0"}


def datasets_root() -> Path:
    """Same resolution order as scripts/prepare_training.py -- the datasets
    live wherever this machine keeps them, not inside the checkout."""
    env = os.getenv("BATTLESIGHT_DATASETS")
    if env:
        return Path(env)
    try:
        from ultralytics.utils import SETTINGS
        configured = SETTINGS.get("datasets_dir")
        if configured:
            return Path(configured)
    except Exception:  # noqa: BLE001
        pass
    return REPO / "datasets"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None,
                   help="default: <datasets_dir>/BDD100K_raw")
    p.add_argument("--workers", type=int, default=12,
                   help="12 is measured, not cautious: 32 rate-limited 443 of "
                        "10,002 files with HTTP 429 on this link.")
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--verify", action="store_true",
                   help="re-download any file whose size does not match the API")
    return p.parse_args()


def listing():
    """File names and sizes from the HF API.

    Two traps, both silent. The response is PAGINATED and the page is all you
    get from one request -- the rest is behind a Link: rel="next" header, so a
    single call returns a fraction of this repo and looks like a complete
    answer. And `expand=1`, which is what the docs reach for, drops the page
    size from 1,000 to 50: 200 round trips instead of 11, for a `size` field
    that the unexpanded form already carries.
    """
    url = f"{API}/tree/main?recursive=1"
    files = []
    while url:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            files.extend(json.loads(r.read().decode()))
            link = r.headers.get("Link", "")
        url = None
        for part in link.split(","):
            if 'rel="next"' in part and "<" in part:
                url = part.split("<", 1)[1].split(">", 1)[0]
    return [(f["path"], f.get("size") or f.get("lfs", {}).get("size") or 0)
            for f in files if f.get("type") == "file" and wanted(f["path"])]


def wanted(path: str) -> bool:
    """The repo is 20,006 files; 10,002 of them are this project's business.

    `fields/drivable/*.png` is 10,000 drivable-area segmentation masks -- a
    different task, 40 MB, and half the total request count. Per-file HTTP
    overhead dominates here, not bytes, so skipping them roughly halves the
    wall-clock. Everything detection needs is data/*.jpg plus samples.json;
    metadata.json is kept only because it records which FiftyOne export this
    came from, which is the sort of thing that matters a year later.
    """
    return path.startswith("data/") or path in ("samples.json", "metadata.json")


def fetch(path, size, out, retries, verify):
    dst = out / path
    if dst.exists() and (not verify or dst.stat().st_size == size):
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{RESOLVE}/{path}", headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dst)
            return dst.stat().st_size
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            # 429 is the one that actually happens here, and it needs a
            # different backoff from a dropped connection: at 32 workers this
            # rate-limited 443 of 10,002 files, and a 1/2/4 s ladder is far too
            # short to outlast it. Honour Retry-After when the server sends
            # one, otherwise back off hard and jitter so the pool does not
            # retry in lockstep and re-trip the limit.
            status = getattr(e, "code", None)
            if status == 429:
                wait = float(getattr(e, "headers", {}).get("Retry-After", 0) or 0)
                time.sleep(wait or (5 * 2 ** attempt) + random.uniform(0, 3))
            else:
                time.sleep(2 ** attempt + random.uniform(0, 1))
    return 0


def main():
    args = parse_args()
    out = args.out or (datasets_root() / "BDD100K_raw")
    out.mkdir(parents=True, exist_ok=True)

    print(f"listing {REPO_ID} ...")
    files = listing()
    total = sum(s for _, s in files)
    print(f"{len(files)} files, {total/1e6:.0f} MB -> {out}")

    have = sum(1 for p, s in files
               if (out / p).exists() and (not args.verify or (out / p).stat().st_size == s))
    if have:
        print(f"{have} already present; fetching the remaining {len(files)-have}")

    done = failed = got = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, p, s, out, args.retries, args.verify): p
                for p, s in files}
        for fut in as_completed(futs):
            done += 1
            try:
                got += fut.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"\n  FAILED {futs[fut]}: {e!r}")
            if done % 200 == 0 or done == len(files):
                el = time.time() - t0
                print(f"  {done}/{len(files)}  {got/1e6:6.0f} MB  "
                      f"{got/1e6/max(el,1):.1f} MB/s  {el/60:.1f} min",
                      flush=True)

    print(f"\n{done - failed}/{len(files)} files in {(time.time()-t0)/60:.1f} min")
    if failed:
        print(f"{failed} failed -- re-run this script, it resumes.")
        sys.exit(1)
    print("\nNext:\n    python scripts/convert_bdd100k.py")


if __name__ == "__main__":
    main()
