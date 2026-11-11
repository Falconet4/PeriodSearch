#!/usr/bin/env python3
"""Score period_search outputs against a reference run.

The pass/fail test is the project validator's, taken verbatim from
validator/period_search_validator*.cpp: equal line counts and, for every line,

    |a - b| / (2 * (a + b))  <=  0.1   (period)
                            <=  0.1   (rms)
                            <=  0.5   (chi-square)

Any failing line rejects the pair. Pole agreement is reported for information
only; the validator does not compare poles.

Usage:
    ./score_vs_reference.py <dir-or-files...> --ref <reference-dir>
"""
import argparse
import math
import os
import sys

TOL = (0.1, 0.1, 0.5)
NAMES = ("period", "rms", "chi")


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 6:
                try:
                    rows.append(tuple(float(x) for x in p[:6]))
                except ValueError:
                    pass
    return rows


def rel(a, b):
    d = a + b
    if d == 0.0:
        return 0.0 if a == b else float("inf")
    return abs(a - b) / (2.0 * d)


def angle_deg(x, y):
    la1, be1, la2, be2 = map(math.radians, (x[4], x[5], y[4], y[5]))
    v1 = (math.cos(be1) * math.cos(la1), math.cos(be1) * math.sin(la1), math.sin(be1))
    v2 = (math.cos(be2) * math.cos(la2), math.cos(be2) * math.sin(la2), math.sin(be2))
    d = max(-1.0, min(1.0, sum(p * q for p, q in zip(v1, v2))))
    return math.degrees(math.acos(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="output directory or .out files")
    ap.add_argument("--ref", required=True, help="reference directory")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".out")]
        else:
            files.append(p)
    if not files:
        sys.exit("no .out files found")

    print(f"{'task':16s} {'rows':>7s} {'exact':>7s} {'<=5deg':>7s} {'<=1deg':>7s}"
          f" {'period':>10s} {'rms':>10s} {'chi':>10s}  status")
    n_all = exact_all = a1_all = a5_all = 0
    worst = [0.0, 0.0, 0.0]
    failures = 0

    for path in files:
        name = os.path.basename(path)
        ref = os.path.join(args.ref, name)
        if not os.path.exists(ref):
            print(f"{name:16s}  (no reference)")
            failures += 1
            continue
        X, Y = load(path), load(ref)
        if len(X) != len(Y):
            print(f"{name:16s}  ROW COUNT {len(X)} vs {len(Y)}")
            failures += 1
            continue
        n = len(X)
        exact = sum(1 for x, y in zip(X, Y) if x == y)
        a1 = sum(1 for x, y in zip(X, Y) if angle_deg(x, y) <= 1.0)
        a5 = sum(1 for x, y in zip(X, Y) if angle_deg(x, y) <= 5.0)
        w = [0.0, 0.0, 0.0]
        bad = 0
        for x, y in zip(X, Y):
            for i in range(3):
                r = rel(x[i], y[i])
                w[i] = max(w[i], r)
                if r > TOL[i]:
                    bad += 1
        status = "PASS" if bad == 0 else f"FAIL ({bad} lines)"
        print(f"{name:16s} {n:7d} {100.0*exact/n:6.2f}% {100.0*a5/n:6.2f}% {100.0*a1/n:6.2f}%"
              f" {w[0]:10.3e} {w[1]:10.3e} {w[2]:10.3e}  {status}")
        n_all += n
        exact_all += exact
        a1_all += a1
        a5_all += a5
        for i in range(3):
            worst[i] = max(worst[i], w[i])
        failures += bad

    if n_all:
        print(f"{'POOLED':16s} {n_all:7d} {100.0*exact_all/n_all:6.2f}%"
              f" {100.0*a5_all/n_all:6.2f}% {100.0*a1_all/n_all:6.2f}%"
              f" {worst[0]:10.3e} {worst[1]:10.3e} {worst[2]:10.3e}")
        print("margins: " + ", ".join(
            f"{NAMES[i]} {TOL[i]/worst[i]:.1f}x" if worst[i] else f"{NAMES[i]} inf"
            for i in range(3)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
