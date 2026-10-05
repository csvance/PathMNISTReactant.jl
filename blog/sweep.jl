# Sweep runner for the Kaimon session: trains each config in turn on the session's GPU, logs
# best validation and test metrics to sweep.jsonl, and frees the handle between runs.
using PathMNISTReactant, ReactantNitro, Reactant, Lux, Random, Dates
const DATA = "/cache/cvance@medicalmetrics.local/pathmnist-blog/data"
const SWEEP_LOG = "/cache/cvance@medicalmetrics.local/pathmnist-blog/sweep.jsonl"
const NOHSV = (; aug_hue = 0.0, aug_sat = 0.0, aug_val = 0.0)
const BASE = (; NOHSV..., grad_clip = 0.0)        # the post's model
best_val(n) = maximum(r.macro_acc for r in ReactantNitro.history(n).rows)
function log_line(d)
    open(io -> println(io, "{", join(["\"$k\":" * (v isa AbstractString ? "\"$(replace(v, "\"" => "'"))\"" : string(v)) for (k, v) in pairs(d)], ","), "}"), SWEEP_LOG, "a")
end
function run_config(name, kw)
    e2 = PathMNIST(; data_dir = DATA, kw...)
    global nitro = Nitro(e2; run_dir = "runs/sweep_" * name, restore_best = true)
    t0 = time(); train!(nitro); secs = time() - t0
    test = evaluate(nitro; split = :test)
    log_line((; name, kw = string(kw), secs = round(secs), val_macro = best_val(nitro), test_macro = test.macro_acc,
        test_acc = test.acc, test_ece = test.ece, test_loss = test.val_loss, when = string(now())))
    global nitro = nothing; GC.gc(); GC.gc()
end
function run_queue!(queue, done_file)
    rm(done_file; force = true)
    while !isempty(queue)
        name, kw = popfirst!(queue)
        try
            run_config(name, kw)
        catch err
            log_line((; name, error = sprint(showerror, err)))
        end
    end
    write(done_file, "ok")
end
