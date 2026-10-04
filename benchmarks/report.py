"""Regenerates RESULTS.md from the JSON lines in results/.

    python3 report.py

Inputs, all under results/:
  pytorch.jsonl       ConvNeXt V2 Tiny, PyTorch Lightning (bench_lightning.py)
  pytorch_bit.jsonl   BiT ResNetV2-50x1, PyTorch Lightning (bench_lightning.py --model ...)
  julia.jsonl         ConvNeXt V2 Tiny, ReactantNitro (bench.jl)
  julia_bit.jsonl     BiT ResNetV2-50x1, ReactantNitro (bench.jl ... bit)
  pytorch_node.jsonl  neural ODE, plain PyTorch loop (bench_node.py)
  julia_node.jsonl    neural ODE, ReactantNitro (bench_node.jl)
  pytorch_cnn_node.jsonl  CNN -> neural ODE -> classifier (bench_lightning.py --node-steps)
  julia_cnn_node.jsonl    the same, ReactantNitro (bench_cnn_node.jl)
  julia_effort.jsonl      the Julia compiler-effort ladder across all four workloads

run_all.sh writes all of them; a {"failed": ...} line is listed under its table.
  status.md           hand-written status and notes
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
STATUS = HERE / "results" / "status.md"


def load(name):
    path = HERE / "results" / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def ratio(x, ref):
    return f"{x / ref:.2f}x" if ref else ""


def split(rows):
    ok = [r for r in rows if "failed" not in r]
    failed = [r["failed"] + (f" ({r['reason']})" if r.get("reason") else "") for r in rows if "failed" in r]
    return ok, failed


def failed_note(failed):
    return ["", "Failed runs: " + "; ".join(f"`{f}`" for f in failed)] if failed else []


def vision_section(title, blurb, pt_rows, jl_rows):
    pt, failed = split(pt_rows)
    jl_rows, jl_failed = split(jl_rows)
    eager = next((r["images_per_s"] for r in pt if r["compile"] == "none" and r["precision"] == "tf32"
                  and not r["channels_last"] and r["cudnn_benchmark"]), None)
    best = max((r["images_per_s"] for r in pt if r["precision"] == "tf32"), default=None)

    out = [f"## {title}", "", blurb, ""]
    out += ["### PyTorch (Lightning), random batches already on the GPU", "",
            "| compile | channels_last | precision | cudnn.benchmark | first step (s) | step (ms) | images/s | vs eager | peak mem (GB) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in pt:
        out.append(f"| {r['compile']} | {r['channels_last']} | {r['precision']} | {r['cudnn_benchmark']} | "
                   f"{fmt(r['first_step_s'])} | {fmt(r['step_ms'], 2)} | {fmt(r['images_per_s'])} | "
                   f"{ratio(r['images_per_s'], eager)} | {fmt(r['peak_mem_gb'], 2)} |")
    if not pt:
        out.append("| pending | | | | | | | | |")
    out += failed_note(failed)

    out += ["", "### Julia (ReactantNitro)", "",
            "| data | precision | run | threads | first step (s) | step (ms) | images/s | vs PyTorch eager | vs best PyTorch (tf32) | data wait |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in jl_rows:
        wait = "" if r.get("data_wait_frac") is None else f"{100 * r['data_wait_frac']:.1f}%"
        out.append(f"| {r['mode']} | {r['precision']} | {r['run']} | {r['threads']} | {fmt(r['first_step_s'])} | "
                   f"{fmt(r['step_ms'], 2)} | {fmt(r['images_per_s'])} | {ratio(r['images_per_s'], eager)} | "
                   f"{ratio(r['images_per_s'], best)} | {wait} |")
    if not jl_rows:
        out.append("| | | pending | | | | | | | |")
    return out + failed_note(jl_failed) + [""]


def node_section(pt_rows, jl_rows):
    pt, failed = split(pt_rows)
    jl_rows, jl_failed = split(jl_rows)
    eager = next((r["step_ms"] for r in pt if r["compile"] == "none" and r["precision"] == "tf32"), None)
    best = min((r["step_ms"] for r in pt if r["precision"] == "tf32"), default=None)
    ref = (pt or jl_rows or [{}])[0]

    out = ["## Neural ODE (RK4)", "",
           f"An MLP vector field `du/dt = dec(tanh(enc(u)))`, state {ref.get('dims', 32)}, hidden "
           f"{ref.get('hidden', 256)} ({ref.get('params', 16672)} parameters), integrated over t in [0, 1] "
           f"with classical fixed-step RK4, {ref.get('steps', 100)} steps; MSE on the final state; batch "
           f"{ref.get('batch_size', 128)}. One step is the forward rollout, the backward through every RK4 "
           "stage, and an RAdam update (lr 1e-3). Random inputs already on the device. PyTorch runs a plain "
           "training loop (no Lightning); Julia unrolls the loop at trace time (SpiNODENitro's RK4 rule), so "
           "the rollout and its Enzyme adjoint are one XLA program. Speedup is PyTorch step time over this "
           "row's step time.", ""]
    out += ["### PyTorch", "",
            "| compile | precision | first step (s) | step (ms) | steps/s | speedup vs eager | peak mem (GB) |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for r in pt:
        sp = f"{eager / r['step_ms']:.2f}x" if eager else ""
        out.append(f"| {r['compile']} | {r['precision']} | {fmt(r['first_step_s'])} | {fmt(r['step_ms'], 3)} | "
                   f"{fmt(r['steps_per_s'])} | {sp} | {fmt(r['peak_mem_gb'], 3)} |")
    if not pt:
        out.append("| pending | | | | | | |")
    out += failed_note(failed)

    out += ["", "### Julia (ReactantNitro)", "",
            "| precision | run | first step (s) | step (ms) | steps/s | speedup vs PyTorch eager | vs best PyTorch (tf32) |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for r in jl_rows:
        sp = f"{eager / r['step_ms']:.2f}x" if eager else ""
        sb = f"{best / r['step_ms']:.2f}x" if best else ""
        out.append(f"| {r['precision']} | {r['run']} | {fmt(r['first_step_s'])} | {fmt(r['step_ms'], 3)} | "
                   f"{fmt(r['steps_per_s'])} | {sp} | {sb} |")
    if not jl_rows:
        out.append("| | pending | | | | | |")
    return out + failed_note(jl_failed) + [""]


EFFORT_WORKLOADS = [  # (title, match, PyTorch results file)
    ("Neural ODE (RK4)", lambda r: r.get("workload") == "node-rk4", "pytorch_node.jsonl"),
    ("CNN -> neural ODE -> classifier", lambda r: r.get("workload") == "cnn-node-rk4", "pytorch_cnn_node.jsonl"),
    ("ConvNeXt V2 Tiny", lambda r: r.get("workload") == "vision" and "convnext" in r.get("backbone", ""), "pytorch.jsonl"),
    ("BiT ResNetV2-50x1", lambda r: r.get("workload") == "vision" and "resnetv2" in r.get("backbone", ""), "pytorch_bit.jsonl"),
]


def effort_section(rows):
    rows, failed = split(rows)
    out = ["## Julia compiler effort", "",
           "The analog of PyTorch's compile modes: each workload at the default precision (TF32) under "
           "rising compiler effort, every run a cold compile. XLA flags are set through `XLA_FLAGS`, "
           "the Reactant preset through `Nitro(...; compile_options)`; the exact flags are in "
           "`run_all.sh` and `julia/common.jl`. Each configuration trains twice in one process, and "
           "both runs time 100 steady-state steps after warmup: the step is their mean and the spread "
           "their difference, a floor on what counts as a real difference between rows. The first "
           "step is the cold compile (the first run). \"vs best PyTorch\" divides PyTorch's fastest "
           "tf32 step for the same workload by this row's step.", ""]
    if not rows and not failed:
        return out + ["Pending.", ""]
    for title, match, pt_file in EFFORT_WORKLOADS:
        mine = [r for r in rows if match(r)]
        if not mine:
            continue
        pt = [r for r in load(pt_file) if "failed" not in r and r.get("precision") == "tf32"]
        best_pt = min((r["step_ms"] for r in pt), default=None)
        configs = list(dict.fromkeys(r["config"] for r in mine))
        cached = {r["config"]: r for r in mine if r["run"] == "cached"}
        first = {r["config"]: r for r in mine if r["run"] == "first"}
        steps = {c: [r["step_ms"] for r in (first.get(c), cached.get(c)) if r] for c in configs}
        mean = {c: sum(v) / len(v) for c, v in steps.items()}
        base = mean.get("default")
        out += [f"### {title}", "",
                "| config | Reactant preset | first step (s) | step (ms) | spread (ms) | vs Julia default | vs best PyTorch (tf32) |",
                "|---|---|---:|---:|---:|---:|---:|"]
        for c in configs:
            r = cached.get(c) or first.get(c)
            fs = first.get(c, {}).get("first_step_s")
            step = mean[c]
            spread = max(steps[c]) - min(steps[c])
            vs_base = f"{base / step:.2f}x" if base else ""
            vs_pt = f"{best_pt / step:.2f}x" if best_pt else ""
            out.append(f"| {c} | {r.get('reactant_preset', '')} | {fmt(fs)} | {fmt(step, 3)} | "
                       f"{fmt(spread, 3)} | {vs_base} | {vs_pt} |")
        out.append("")
    return out + failed_note(failed) + [""]


def bit_norm_section(rows, pt_rows):
    rows, failed = split(rows)
    out = ["## BiT ResNetV2-50x1: GroupNorm statistics", "",
           "Lux's `GroupNorm` (statistics from `mean`, then `var` around it: two dependent reductions, so "
           "XLA reads each activation three times) against Luximm's `ShiftedGroupNorm` (`norm_impl = "
           ":shifted`: shifted sums, two independent reductions XLA can merge into one pass). Same "
           "parameters and weights; at strict FP32 the logits agree to 3e-6 relative and the gradients "
           "to 0.12% relative L2, against 2.9% for TF32 noise alone. Processes interleave the two "
           "implementations; each gives two steady-state measurements, all of which are averaged. "
           "The shift is detached from the gradient (its gradient is exactly zero; the gradient through "
           "the mean and variance is intact). A first version with the shift attached measured 121.4 ms "
           "against 73.0 ms (0.60x; results in `results/archive/`): the reverse pass of the slice "
           "scattered into a full-size tensor at every GroupNorm. In a conv -> GroupNorm -> ReLU -> conv "
           "microbenchmark at 56x56x256x16 the detached version is level with Lux (4.908 against 4.893 "
           "ms forward and backward), and 11% faster on the GroupNorm alone, but in the full model it is "
           "still slower: the expected saving does not materialize.", ""]
    if not rows and not failed:
        return out + ["Pending.", ""]
    pt_best = min((r["step_ms"] for r in pt_rows if "failed" not in r and r.get("precision") == "tf32"),
                  default=None)
    groups = {}
    for r in rows:
        groups.setdefault((r.get("precision"), r.get("norm_impl", "lux")), []).append(r)
    out += ["| precision | GroupNorm | measurements | first step (s) | step (ms) | spread (ms) | images/s | vs Lux GroupNorm | vs best PyTorch (tf32) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for prec in ("default", "highest"):
        base = None
        for impl in ("lux", "shifted"):
            g = groups.get((prec, impl))
            if not g:
                continue
            steps = [r["step_ms"] for r in g]
            mean = sum(steps) / len(steps)
            firsts = [r["first_step_s"] for r in g if r["run"] == "first"]
            if impl == "lux":
                base = mean
            ips = sum(r.get("images_per_s", r["batch_size"] * 1000 / r["step_ms"]) for r in g) / len(g)
            vs_base = f"{base / mean:.3f}x" if base else ""
            vs_pt = f"{pt_best / mean:.2f}x" if pt_best and prec == "default" else ""
            out.append(f"| {prec} | {impl} | {len(steps)} | {fmt(sum(firsts) / len(firsts) if firsts else None)} | "
                       f"{fmt(mean, 2)} | {fmt(max(steps) - min(steps), 2)} | {fmt(ips)} | {vs_base} | {vs_pt} |")
    return out + failed_note(failed) + [""]


def meanvar_section(rows):
    rows, failed = split(rows)
    out = ["## Normalization backward: μ detached inside the variance", "",
           "LuxLib's `mean_var` computes `var(x; mean = μ)`; Enzyme's reverse pass then spends a full-size "
           "reduction per normalization on ∂σ²/∂μ, which is identically zero. `BENCH_MEANVAR=stopgrad` "
           "detaches μ inside the variance only (an opt-in override for traced arrays in "
           "`julia/common.jl`): the forward is bit-identical and the gradient to x is unchanged (BiT at "
           "strict FP32: logits identical, gradients 4.9e-7 relative L2). It reaches every LuxLib "
           "normalization, GroupNorm in BiT and LayerNorm in ConvNeXt. Default precision (TF32); "
           "processes interleave the two modes and each gives two steady-state measurements.", ""]
    if not rows and not failed:
        return out + ["Pending.", ""]
    groups = {}
    for r in rows:
        groups.setdefault((r.get("backbone"), r.get("meanvar", "default")), []).append(r)
    out += ["| backbone | mean_var | measurements | step (ms) | spread (ms) | images/s | vs default |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for bb in dict.fromkeys(r.get("backbone") for r in rows):
        base = None
        for mv in ("default", "stopgrad"):
            g = groups.get((bb, mv))
            if not g:
                continue
            steps = [r["step_ms"] for r in g]
            mean = sum(steps) / len(steps)
            if mv == "default":
                base = mean
            ips = sum(r.get("images_per_s", r["batch_size"] * 1000 / r["step_ms"]) for r in g) / len(g)
            out.append(f"| {bb} | {mv} | {len(steps)} | {fmt(mean, 2)} | {fmt(max(steps) - min(steps), 2)} | "
                       f"{fmt(ips)} | {f'{base / mean:.3f}x' if base else ''} |")
    return out + failed_note(failed) + [""]


def optimized_section(rows):
    rows, failed = split(rows)
    out = ["## Julia with the normalization and layout changes", "",
           "Every run here includes LuxLib's `mean_var` Reactant specialization (Lux.jl branch "
           "`luxlib/reactant-mean-var`) and Luximm's StdConv through `mean_var` (branch "
           "`perf/mean-var-norms`). `conv_mlp = false` is Luximm's channels-last ConvNeXtV2 block "
           "(branch `perf/convnextv2-channels-last`): LayerNorm over a contiguous channel axis and GEMM "
           "pointwise layers, same weights; config `conv-mlp-false-grn-rows` adds its GRN with the "
           "spatial sum of squares made a contiguous reduction. Default precision (TF32); step is the mean of the two "
           "steady-state measurements of one process. \"vs best PyTorch\" divides PyTorch's fastest "
           "tf32 step for the same workload by this row's step.", ""]
    if not rows and not failed:
        return out + ["Pending.", ""]
    pt_best = {
        ("vision", "convnext"): min((r["step_ms"] for r in load("pytorch.jsonl") if "failed" not in r and r["precision"] == "tf32"), default=None),
        ("vision", "resnetv2"): min((r["step_ms"] for r in load("pytorch_bit.jsonl") if "failed" not in r and r["precision"] == "tf32"), default=None),
        ("node-rk4", "node"): min((r["step_ms"] for r in load("pytorch_node.jsonl") if "failed" not in r and r["precision"] == "tf32"), default=None),
        ("cnn-node-rk4", "convnext"): min((r["step_ms"] for r in load("pytorch_cnn_node.jsonl") if "failed" not in r and r["precision"] == "tf32"), default=None),
    }
    groups = {}
    for r in rows:
        groups.setdefault((r["workload"], r.get("backbone", "node"), r.get("conv_mlp", True), r.get("config", "")), []).append(r)
    out += ["| workload | backbone | conv_mlp | config | first step (s) | step (ms) | images/s | vs best PyTorch (tf32) |",
            "|---|---|---|---|---:|---:|---:|---:|"]
    for (wl, bb, cm, cfg), g in groups.items():
        steps = [r["step_ms"] for r in g]
        mean = sum(steps) / len(steps)
        fs = [r["first_step_s"] for r in g if r["run"] == "first"]
        ref = pt_best.get((wl, "node" if bb == "node" else "convnext" if "convnext" in bb else "resnetv2"))
        ips = sum(r.get("images_per_s", r["batch_size"] * 1000 / r["step_ms"]) for r in g) / len(g)
        out.append(f"| {wl} | {bb} | {'' if 'convnext' not in bb else cm} | {cfg} | {fmt(fs[0] if fs else None)} | "
                   f"{fmt(mean, 2)} | {fmt(ips)} | {f'{ref / mean:.2f}x' if ref else ''} |")
    return out + failed_note(failed) + [""]


def main():
    pt_cnx, pt_bit = load("pytorch.jsonl"), load("pytorch_bit.jsonl")
    jl_cnx, jl_bit = load("julia.jsonl"), load("julia_bit.jsonl")
    jl = jl_cnx + jl_bit

    out = ["# Training throughput: PyTorch vs ReactantNitro", ""]
    r = next((r for r in pt_cnx + pt_bit if "failed" not in r and "lightning" in r), None)
    if r:
        out += [f"GPU: {r['gpu']} (GPU 2). Throughput is steady state over 100 optimizer steps after 30 "
                "warmup steps; `first step` includes compilation.", "",
                f"PyTorch {r['torch']}, Lightning {r['lightning']}, timm {r['timm']}."]
    j = next((r for r in jl if "failed" not in r), None)
    if j and "reactantnitro" in j:
        out += [f"Julia {j['julia']}, ReactantNitro {j['reactantnitro']}, Reactant {j['reactant']}."]
    out += [""]
    if STATUS.exists():
        out += [STATUS.read_text().rstrip(), ""]

    common = ("9 classes, batch {b} x {a} accumulation steps, UInt8 input normalized on device, RAdam lr "
              "1e-3, no decay, no clipping.")
    out += vision_section(
        "ConvNeXt V2 Tiny", "27,873,417 parameters; " + common.format(b=16, a=2), pt_cnx, jl_cnx)
    out += vision_section(
        "BiT ResNetV2-50x1", "23,518,793 parameters (timm's bare linear head, no extra LayerNorm); "
        + common.format(b=16, a=2) + " Normalization mean and std 0.5.", pt_bit, jl_bit)
    out += node_section(load("pytorch_node.jsonl"), load("julia_node.jsonl"))
    out += vision_section(
        "CNN -> neural ODE -> classifier",
        "ConvNeXt V2 Tiny backbone, pooled features -> linear (768 => 64) -> neural ODE (the MLP field "
        "above, hidden 256, RK4 with 100 steps) -> linear (64 => 9) logits, cross-entropy; 27,949,385 "
        "parameters. A mixed program: a few large convolution kernels, then 400 small vector-field "
        "evaluations, forward and backward. " + common.format(b=16, a=2),
        load("pytorch_cnn_node.jsonl"), load("julia_cnn_node.jsonl"))

    out += effort_section(load("julia_effort.jsonl"))
    out += bit_norm_section(load("julia_bit_norm.jsonl"), load("pytorch_bit.jsonl"))
    out += meanvar_section(load("julia_meanvar.jsonl"))
    out += optimized_section(load("julia_luxlib.jsonl"))

    (HERE / "RESULTS.md").write_text("\n".join(out).rstrip() + "\n")


if __name__ == "__main__":
    main()
