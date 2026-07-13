"""Parse the label-efficiency logs into the with/without-teacher curve (best val macro AUPRC per run)."""
import re
caps = [10, 25, 50, 100, 0]
def best(tag):
    try:
        t = open(f"/tmp/{tag}.log").read()
        b = [float(x) for x in re.findall(r"best=([0-9.]+)", t)]
        return max(b) if b else float("nan")
    except Exception:
        return float("nan")
print("=== Label-efficiency: WITH vs WITHOUT teacher (8-verb, fold0 val, best macro AUPRC) ===")
print(f"{'labels':>7s} {'scratch':>8s} {'pretrained':>11s} {'gap (pre-scr)':>14s}")
for cap in caps:
    s = best(f"le8_scr_{cap}"); p = best(f"le8_pre_{cap}")
    lbl = "full(~190)" if cap == 0 else str(cap)
    print(f"{lbl:>7s} {s:8.3f} {p:11.3f} {p-s:+14.3f}")
print("\nExpected story: large gap at 10-25 labels, ~0 by full => distillation is a warm start whose benefit")
print("vanishes by ~full data; substantial below. Converts Stage 1 from a design choice into a finding.")
