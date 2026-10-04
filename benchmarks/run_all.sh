#!/usr/bin/env bash
# Runs every benchmark end to end, one process per configuration, on one GPU, then
# regenerates RESULTS.md. Safe to leave running overnight.
#
#   ./run_all.sh              # everything
#   ./run_all.sh node bit     # only the groups named (vision, bit, node, cnn-node,
#                             # julia-effort)
#
# Environment:
#   BENCH_GPU      GPU index (PCI bus order) to run on; default 2
#   BENCH_TIMEOUT  per-run limit in seconds; default 5400
#   JULIA          Julia executable; default julia
#
# The previous results of the groups being run move to results/archive/<timestamp>/
# first, so each section of RESULTS.md comes from one consistent run. Before each run the script waits for
# the GPU to have no compute process (up to 10 minutes) and records a skip if it never
# frees; a run that fails or times out is recorded as {"failed": ...} and the rest go on.
# Progress goes to results/run_all.log.

set -u
cd "$(dirname "$0")"
GPU=${BENCH_GPU:-2}
TIMEOUT=${BENCH_TIMEOUT:-5400}
JULIA=${JULIA:-julia}
GROUPS_WANTED=${*:-vision bit node cnn-node julia-effort}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU

wants() { [[ " $GROUPS_WANTED " == *" $1 "* ]]; }

R=results
stamp=$(date +%Y%m%d-%H%M%S)
LOG=$R/run_all.log
# Checked before anything is archived, so a broken environment moves no results.
for exe in uv "$JULIA" nvidia-smi timeout python3; do
    command -v "$exe" > /dev/null || { echo "missing executable: $exe" | tee -a "$LOG"; exit 1; }
done
files=()
wants vision && files+=(pytorch julia)
wants bit && files+=(pytorch_bit julia_bit)
wants node && files+=(pytorch_node julia_node)
wants cnn-node && files+=(pytorch_cnn_node julia_cnn_node)
wants julia-effort && files+=(julia_effort)
mkdir -p "$R/archive/$stamp"
for f in "${files[@]}"; do
    mv "$R/$f.jsonl" "$R/$f.err" "$R/archive/$stamp/" 2>/dev/null
done
echo "== run_all $stamp on GPU $GPU: $GROUPS_WANTED" >> "$LOG"

UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1 == g {print $2}')
[ -n "$UUID" ] || { echo "no GPU with index $GPU" | tee -a "$LOG"; exit 1; }

gpu_free() {
    for _ in $(seq 1 120); do
        nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID" || return 0
        sleep 5
    done
    return 1
}

# run <results file> <label> <command...>
run() {
    local out=$1 label=$2; shift 2
    local t0=$(date +%s)
    if ! gpu_free; then
        echo "{\"failed\": \"$label\", \"reason\": \"GPU $GPU busy\"}" >> "$R/$out"
        echo "$(date +%T) SKIP  $label (GPU busy)" | tee -a "$LOG"
        return
    fi
    # Cold compiles: fresh Inductor and Triton caches for every run, deleted afterwards.
    # Reactant's persistent cache is off for this project (julia/LocalPreferences.toml).
    local cache
    cache=$(mktemp -d)
    if BENCH_OUT="$R/$out" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton" \
        timeout "$TIMEOUT" "$@" > /dev/null 2>> "$R/${out%.jsonl}.err"; then
        echo "$(date +%T) ok    $label ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG"
    else
        echo "{\"failed\": \"$label\", \"reason\": \"exit $?\"}" >> "$R/$out"
        echo "$(date +%T) FAIL  $label ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG"
    fi
    rm -rf "$cache"
}

# Arrays, not functions: `timeout` executes a program and cannot call a shell function.
PY=(uv run --project pytorch python)
JL=("$JULIA" --startup-file=no -t 16,1 --project=julia)

CONVNEXT=convnextv2_tiny.fcmae_ft_in22k_in1k
BIT=resnetv2_50x1_bit.goog_in21k
COMPILE_MODES="default reduce-overhead max-autotune max-autotune-no-cudagraphs"

# The PyTorch vision matrix for one model: eager at both precisions, with and without
# channels_last; eager without cudnn.benchmark; every compile mode at tf32, with and
# without channels_last.
vision_matrix() {
    local out=$1 model=$2
    for cl in "" --channels-last; do
        for prec in tf32 fp32; do
            run "$out" "pytorch $model eager $prec $cl" "${PY[@]}" pytorch/bench_lightning.py --model "$model" --precision $prec $cl
        done
    done
    run "$out" "pytorch $model eager tf32 no-cudnn-benchmark" "${PY[@]}" pytorch/bench_lightning.py --model "$model" --precision tf32 --no-cudnn-benchmark
    for mode in $COMPILE_MODES; do
        for cl in "" --channels-last; do
            run "$out" "pytorch $model $mode tf32 $cl" "${PY[@]}" pytorch/bench_lightning.py --model "$model" --compile $mode --precision tf32 $cl
        done
    done
}

if wants vision; then
    vision_matrix pytorch.jsonl "$CONVNEXT"
    for args in "prebuilt default" "prebuilt highest" "pipeline default"; do
        run julia.jsonl "julia convnextv2 $args" "${JL[@]}" julia/bench.jl $args convnextv2
    done
fi

if wants bit; then
    vision_matrix pytorch_bit.jsonl "$BIT"
    for args in "prebuilt default" "prebuilt highest"; do
        run julia_bit.jsonl "julia bit $args" "${JL[@]}" julia/bench.jl $args bit
    done
fi

if wants node; then
    run pytorch_node.jsonl "pytorch node eager fp32" "${PY[@]}" pytorch/bench_node.py --precision fp32
    for mode in none $COMPILE_MODES; do
        run pytorch_node.jsonl "pytorch node $mode tf32" "${PY[@]}" pytorch/bench_node.py --compile $mode --precision tf32
    done
    for prec in default highest; do
        run julia_node.jsonl "julia node $prec" "${JL[@]}" julia/bench_node.jl $prec
    done
fi

if wants cnn-node; then
    for mode in none $COMPILE_MODES; do
        run pytorch_cnn_node.jsonl "pytorch cnn-node $mode tf32" "${PY[@]}" pytorch/bench_lightning.py --model "$CONVNEXT" --node-steps 100 --compile $mode --precision tf32
    done
    for prec in default highest; do
        run julia_cnn_node.jsonl "julia cnn-node $prec" "${JL[@]}" julia/bench_cnn_node.jl $prec convnextv2
    done
fi

# The Julia compiler-effort ladder, the analog of PyTorch's compile modes: each workload at the
# default precision (TF32, the match for PyTorch tf32) under rising effort. XLA flags go through
# XLA_FLAGS (read once, when Reactant loads); the Reactant preset through Nitro's compile_options
# (julia/common.jl REACTANT_PRESETS).
XLA_GRAPHS="--xla_gpu_enable_command_buffer=+CONVOLUTION,+WHILE --xla_gpu_graph_min_graph_size=1"
XLA_MAX="$XLA_GRAPHS --xla_gpu_exhaustive_tiling_search=true --xla_gpu_cudnn_gemm_fusion_level=2 --xla_gpu_command_buffer_unroll_loops=true"
# name|XLA_FLAGS|Reactant preset
EFFORT=("default||default" "xla-graphs|$XLA_GRAPHS|default" "xla-max|$XLA_MAX|default" "max|$XLA_MAX|max")
EFFORT_CNN=("max-nhwc|$XLA_MAX --xla_gpu_force_conv_nhwc=true|max")

effort() {   # effort <label> <conv?> <script args...>
    local label=$1 conv=$2; shift 2
    local configs=("${EFFORT[@]}")
    [ "$conv" = conv ] && configs+=("${EFFORT_CNN[@]}")
    local c name flags preset
    for c in "${configs[@]}"; do
        IFS='|' read -r name flags preset <<< "$c"
        run julia_effort.jsonl "julia effort $label $name" \
            env XLA_FLAGS="$flags" BENCH_REACTANT="$preset" BENCH_CONFIG="$name" "${JL[@]}" "$@"
    done
}

if wants julia-effort; then
    effort node noconv julia/bench_node.jl default
    effort cnn-node conv julia/bench_cnn_node.jl default convnextv2
    effort convnextv2 conv julia/bench.jl prebuilt default convnextv2
    effort bit conv julia/bench.jl prebuilt default bit
fi

python3 report.py && echo "$(date +%T) RESULTS.md regenerated" | tee -a "$LOG"
echo "== run_all $stamp done" >> "$LOG"
