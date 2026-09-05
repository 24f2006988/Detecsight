"""Extract the CrowdHuman zips into datasets/CrowdHuman_raw/Images/.

All four zips carry an `Images/` prefix and disjoint filenames, so they merge
into a single flat directory -- which is what convert_crowdhuman.py expects.
The .odgt files are what say which image belongs to which split.
Idempotent: an entry already on disk at the right size is skipped.
"""
import sys
import zipfile
from pathlib import Path

RAW = Path("datasets/CrowdHuman_raw")
ZIPS = ["CrowdHuman_train01.zip", "CrowdHuman_train02.zip",
        "CrowdHuman_train03.zip", "CrowdHuman_val.zip"]

total = skipped = 0
for name in ZIPS:
    zp = RAW / name
    if not zp.exists():
        print(f"[skip] {name} not found")
        continue
    with zipfile.ZipFile(zp) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        done = 0
        for i in infos:
            dst = RAW / i.filename
            if dst.exists() and dst.stat().st_size == i.file_size:
                skipped += 1
                done += 1
                continue
            z.extract(i, RAW)
            done += 1
            total += 1
            if done % 1000 == 0:
                print(f"  {name}: {done}/{len(infos)}", flush=True)
        print(f"[ ok ] {name}: {len(infos)} entries", flush=True)

n = len(list((RAW / "Images").glob("*.jpg")))
print(f"\nextracted {total} new, {skipped} already present")
print(f"Images/ now holds {n} jpg (expect 19370)")
sys.exit(0 if n == 19370 else 1)
