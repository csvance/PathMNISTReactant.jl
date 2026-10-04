"""Structural comparison of two dump_hlo.jl output directories.

    python3 compare_hlo.py <dir_a> <dir_b>

For the StableHLO stages: op counts by name. For XLA's optimized HLO: the instructions in the
ENTRY computation (each one a kernel launch or a host-side op), fusions by kind, custom calls by
target, and how many fusion computations contain a reduce.
"""

import collections
import re
import sys
from pathlib import Path


def stablehlo_ops(text):
    return collections.Counter(re.findall(r'\b((?:stablehlo|enzyme|enzymexla|chlo)\.[a-z_]+)\b', text))


def xla_summary(text):
    # Split into computations: "name {" ... "}" at column 0; the ENTRY one is marked.
    comps, cur, name, entry = {}, [], None, None
    for line in text.splitlines():
        m = re.match(r'^(ENTRY\s+)?(%?[\w.\-]+)\s.*\{\s*$', line)
        if m and not line.startswith(" "):
            name = m.group(2)
            if m.group(1):
                entry = name
            cur = []
            continue
        if line.startswith("}") and name:
            comps[name] = cur
            name = None
            continue
        if name:
            cur.append(line)
    out = collections.Counter()
    entry_lines = comps.get(entry, [])
    for line in entry_lines:
        m = re.search(r'=\s*[^\s]+\s+([a-z\-]+)\(', line)
        if not m:
            continue
        op = m.group(1)
        if op in ("parameter", "get-tuple-element", "tuple", "constant", "bitcast"):
            continue
        if op == "fusion":
            k = re.search(r'kind=(k\w+)', line)
            out[f"fusion {k.group(1) if k else '?'}"] += 1
        elif op == "custom-call":
            t = re.search(r'custom_call_target="([^"]+)"', line)
            out[f"custom-call {t.group(1) if t else '?'}"] += 1
        else:
            out[op] += 1
    out["(kernels in ENTRY, approx)"] = sum(v for k, v in out.items() if not k.startswith("("))
    fused_reduce = sum(1 for n, ls in comps.items() if n != entry and any(" reduce(" in l for l in ls))
    out["(computations containing a reduce)"] = fused_reduce
    return out


def table(title, a, b, la, lb, keys=None, top=None):
    keys = keys or sorted(set(a) | set(b), key=lambda k: (-abs(a.get(k, 0) - b.get(k, 0)), -max(a.get(k, 0), b.get(k, 0))))
    if top:
        keys = keys[:top]
    print(f"\n### {title}\n")
    print(f"| op | {la} | {lb} | diff |")
    print("|---|---:|---:|---:|")
    for k in keys:
        x, y = a.get(k, 0), b.get(k, 0)
        print(f"| `{k}` | {x} | {y} | {y - x:+d} |")


def main():
    da, db = Path(sys.argv[1]), Path(sys.argv[2])
    la, lb = da.name, db.name
    for stage in ("stablehlo_raw.mlir", "stablehlo_opt.mlir"):
        a, b = stablehlo_ops((da / stage).read_text()), stablehlo_ops((db / stage).read_text())
        print(f"\n## {stage}: total ops {sum(a.values())} vs {sum(b.values())}")
        table(f"{stage}: largest differences", a, b, la, lb, top=15)
    a = xla_summary((da / "xla_opt.hlo.txt").read_text())
    b = xla_summary((db / "xla_opt.hlo.txt").read_text())
    print("\n## xla_opt.hlo.txt (ENTRY computation)")
    table("XLA optimized HLO", a, b, la, lb)


if __name__ == "__main__":
    main()
