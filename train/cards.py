"""Print each card's HBM usage and pick the emptiest one.

    python cards.py            # table
    python cards.py --pick 35  # just the id of the first card with >= 35 GB free
"""
import argparse
import re
import subprocess
import sys


def read_cards():
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        if "|" not in line:
            continue
        pairs = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", line)]
        hbm = [p for p in pairs if p[1] > 1000 and abs(p[1] - 65536) <= 1024]
        if hbm:
            rows.append(hbm[-1])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", type=float, default=None, help="min free GB")
    args = ap.parse_args()
    rows = read_cards()
    if not rows:
        print("could not parse npu-smi", file=sys.stderr)
        return 1
    order = []
    print("%-6s %8s %8s %8s" % ("card", "used MB", "free MB", "free GB"))
    for c, (used, total) in enumerate(rows):
        free = total - used
        order.append((free, c))
        print("%-6d %8d %8d %8.1f" % (c, used, free, free / 1024.0))
    order.sort(reverse=True)
    print("emptiest: card %d with %.1f GB free" % (order[0][1], order[0][0] / 1024.0))
    if args.pick is not None:
        need = args.pick * 1024.0
        pick = [c for free, c in order if free >= need]
        if not pick:
            print("none", file=sys.stderr)
            return 1
        print(pick[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
