"""Chart the size-stratified recall curve -- the metric that limits this system.

Overall mAP50 hides the shape of the problem: near targets are effectively
solved and all the loss is at distance, but distant targets are a minority of
the instance-weighted metric. Plotting recall against target size next to the
instance histogram makes the trade visible in one image.

    python scripts/plot_size_recall.py                      # measure, then plot
    python scripts/plot_size_recall.py --from results.json  # plot a saved run

Writes docs/size_recall.png.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Recorded 2026-09-02 on the deployed checkpoint, 300 WiderPerson val images,
# imgsz 1280, conf 0.10. Re-measured 2026-09-05 after refactoring the measuring
# code and reproduced to within one box in two buckets.
BASELINE = {
    "buckets": ["<16", "16-32", "32-48", "48-64", "64-96", ">96"],
    "gt": [1427, 2214, 1426, 1128, 1481, 1207],
    "found": [267, 1378, 1148, 964, 1388, 1134],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="src", default=None,
                   help="Plot a JSON dump from scripts/evaluate.py instead of measuring")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--images", default="datasets/WiderPerson/images/val")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="docs/size_recall.png")
    p.add_argument("--baseline-only", action="store_true",
                   help="Plot the recorded numbers without touching a GPU")
    return p.parse_args()


def main():
    args = parse_args()

    if args.baseline_only:
        m = dict(BASELINE)
    elif args.src:
        loaded = json.loads(Path(args.src).read_text())
        m = loaded.get("size_recall", loaded)
    else:
        from scripts.eval_size_recall import measure
        m = measure(args.weights, args.images, args.limit, args.imgsz,
                    device=args.device)

    buckets = m["buckets"]
    gt = m["gt"]
    found = m["found"]
    recall = [f / g if g else 0.0 for f, g in zip(found, gt, strict=True)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
    x = range(len(buckets))

    # Instance histogram behind the curve: the point is that the buckets with
    # the worst recall are not rare.
    bars = ax.bar(x, gt, color="#dfe4ea", edgecolor="#c3cad3", zorder=1,
                  label="ground-truth instances")
    ax.set_ylabel("instances in the validation sample", color="#7a8794")
    ax.tick_params(axis="y", labelcolor="#7a8794")
    ax.set_ylim(0, max(gt) * 1.42)

    ax2 = ax.twinx()
    ax2.plot(x, recall, marker="o", linewidth=2.4, markersize=7,
             color="#c0392b", zorder=3, label="recall")
    for xi, r in zip(x, recall, strict=True):
        ax2.annotate(f"{r:.3f}", (xi, r), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=9.5,
                     color="#c0392b", fontweight="bold")
    ax2.set_ylabel("recall @ IoU 0.5", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(0, 1.12)

    # The two buckets that carry the problem.
    small = sum(gt[:2]) / sum(gt)
    ax.axvspan(-0.5, 1.5, color="#c0392b", alpha=0.06, zorder=0)
    ax.annotate(f"{small:.0%} of all people are under 32 px,\n"
                f"and fewer than half of them are found",
                xy=(0.5, max(gt) * 1.16), ha="center", va="bottom",
                fontsize=10, color="#8c2d22")

    ax.set_xticks(list(x))
    ax.set_xticklabels(buckets)
    ax.set_xlabel("ground-truth target size, sqrt(area) in pixels")
    ax.set_title("Recall collapses with target size, not with scene difficulty",
                 fontsize=12.5, pad=14)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#eef1f4", zorder=0)
    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)

    # No legend: both series are already named by their own axis label, in
    # their own colour, and a legend box here only collides with the callout.
    del bars

    sub = (f"{m.get('weights', 'weights/best.pt')} · "
           f"{m.get('n_images', args.limit)} WiderPerson val images · "
           f"imgsz {m.get('imgsz', args.imgsz)} · conf {m.get('conf', 0.10)}")
    fig.text(0.5, 0.005, sub, ha="center", fontsize=8, color="#8d99a6")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
