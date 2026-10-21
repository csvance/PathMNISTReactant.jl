# CPU stage of the blog assets: every figure in the post, drawn twice (light and dark theme), and
# the data files the post's interactive figures read. Reads the GPU stage's dump (`data.jl`) and
# the memory-mapped test split; no GPU, no Reactant.
#
#   julia --project=<env with CairoMakie> blog/figures.jl <assets_dir> <data_dir> <post_figures_dir>
#
#   assets_dir        what `blog_data` wrote: test_probs.f32, traj_P.f32, traj_picks.i64
#   data_dir          the converted dataset (test_images.u8, test_labels.u8)
#   post_figures_dir  the post's figures/ directory; everything below is written there
#
# Writes: correct, incorrect (each error beside its two classes), reliability, confusion, nonagon, flows (.png and -dark.png),
# trajectories.json, confusion.json, hero/tiles.jpg and hero/tiles.json; and, into assets_dir,
# numbers.md, a report of every number the post quotes from this model.
using CairoMakie, Mmap, Random, Base64, Printf

const ASSETS, DATA, OUT = ARGS[1], ARGS[2], ARGS[3]
const CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
    "normal mucosa", "stroma", "tumor epithelium",
]
const K = length(CLASSES)

# The validated reference sequential blue ramp (steps 100 to 700) and categorical palette, each
# with its dark-surface steps; "background" takes a neutral gray as the ninth class.
const BLUES = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
const THEMES = Dict(
    "light" => (;
        bg = "#ffffff", ink = "#1f2328", muted = "#57606a", grid = "#d0d7de",
        err = "#cf222e", bar = "#2a78d6", ramp = BLUES,
        cls = ["#2a78d6", "#8c959f", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    ),
    "dark" => (;
        bg = "#0f1115", ink = "#d7dae0", muted = "#9aa3ae", grid = "#2a2f38",
        err = "#ff7b72", bar = "#3987e5", ramp = reverse(BLUES),
        cls = ["#3987e5", "#6e7681", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    ),
)

function makie_theme(T)
    spines = (; leftspinecolor = T.muted, rightspinecolor = T.muted, topspinecolor = T.muted, bottomspinecolor = T.muted)
    return Theme(;
        backgroundcolor = T.bg, textcolor = T.ink,
        Axis = (;
            backgroundcolor = T.bg, xgridcolor = T.grid, ygridcolor = T.grid, titlecolor = T.ink,
            xtickcolor = T.muted, ytickcolor = T.muted, xticklabelcolor = T.ink, yticklabelcolor = T.ink,
            xlabelcolor = T.ink, ylabelcolor = T.ink, spines...,
        ),
        Legend = (; backgroundcolor = T.bg, framevisible = false, labelcolor = T.ink),
        Colorbar = (;
            tickcolor = T.muted, ticklabelcolor = T.ink, labelcolor = T.ink,
            leftspinecolor = T.muted, rightspinecolor = T.muted, topspinecolor = T.muted, bottomspinecolor = T.muted,
        ),
        Label = (; color = T.ink)
    )
end

# ── Data ─────────────────────────────────────────────────────────────────────────
const TILE = (3, 224, 224)
images = open(joinpath(DATA, "test_images.u8")) do io
    Mmap.mmap(io, Array{UInt8, 4}, (TILE..., filesize(io) ÷ prod(TILE)))
end
labels = Int.(read(joinpath(DATA, "test_labels.u8"))) .+ 1
probs = Float64.(reshape(reinterpret(Float32, read(joinpath(ASSETS, "test_probs.f32"))), K, :))
conf = vec(maximum(probs; dims = 1))
pred = [argmax(c) for c in eachcol(probs)]
C = zeros(Int, K, K); for (l, p) in zip(labels, pred)
    C[l, p] += 1
end
share = C ./ sum(C; dims = 2)
picks = reinterpret(Int64, read(joinpath(ASSETS, "traj_picks.i64")))
P = Float64.(reshape(reinterpret(Float32, read(joinpath(ASSETS, "traj_P.f32"))), K, length(picks), :))
plabels = labels[picks]

tile(i) = permutedims(view(images, :, :, :, i), (2, 3, 1))
rgb(t) = [RGBf(t[x, y, 1] / 255, t[x, y, 2] / 255, t[x, y, 3] / 255) for x in axes(t, 1), y in axes(t, 2)]
fname(base, mode) = joinpath(OUT, mode == "light" ? "$base.png" : "$base-dark.png")

# ── Figures ──────────────────────────────────────────────────────────────────────
function prediction_grid(T, mode, idxs, title, base; cols)
    rows = cld(length(idxs), cols)
    fig = Figure(size = (230 * cols, 260 * rows + 40))
    Label(fig[0, 1:cols], title; fontsize = 18, font = :bold)
    for (j, i) in enumerate(idxs)
        r, c = fldmod1(j, cols)
        ok = pred[i] == labels[i]
        ax = Axis(
            fig[r, c]; aspect = DataAspect(), yreversed = true, titlesize = 12,
            title = "true: $(CLASSES[labels[i]])\npred: $(CLASSES[pred[i]]) ($(round(Int, 100conf[i]))%)",
            titlecolor = ok ? T.ink : T.err
        )
        hidedecorations!(ax); hidespines!(ax)
        image!(ax, rgb(tile(i)))
    end
    return save(fname(base, mode), fig; px_per_unit = 1)
end

# Correctly classified tiles of class k at evenly spaced confidence ranks, so the examples are
# typical of the class rather than its easiest cases.
function examples(k, n)
    idx = findall((labels .== k) .& (pred .== k))
    idx = idx[sortperm(conf[idx])]
    return [idx[round(Int, j * length(idx) / (n + 1))] for j in 1:n]
end

# The most confident errors, one per row, each beside correctly classified tiles of the class it
# was predicted as and of its true class, so every error sits next to both of its references.
# Rows that share a class show different examples of it.
function error_comparison(T, mode, errs; n = 6, per = 2, px = 150)
    errs = errs[1:min(n, end)]
    seen = Dict{Int, Int}()                       # examples of each class used so far
    pool = Dict(k => examples(k, per * n) for k in unique(vcat(labels[errs], pred[errs])))
    take(k) = (u = get(seen, k, 0); seen[k] = u + per; pool[k][(u + 1):(u + per)])
    cols = 1 + 2 * per
    fig = Figure()
    Label(fig[0, 1:cols], "The most confident errors, beside the class predicted and the true class"; fontsize = 18, font = :bold)
    Label(fig[1, 1], "Error"; fontsize = 14, font = :bold, color = T.err)
    Label(fig[1, 2:(1 + per)], "Predicted class"; fontsize = 14, font = :bold)
    Label(fig[1, (2 + per):cols], "True class"; fontsize = 14, font = :bold)
    for (r, i) in enumerate(errs)
        t, p = labels[i], pred[i]
        tiles = vcat(
            [(i, "$(CLASSES[t]) → $(CLASSES[p])\n$(round(100conf[i]; digits = 1))% confident", T.err)],
            [(j, CLASSES[p], T.muted) for j in take(p)], [(j, CLASSES[t], T.muted) for j in take(t)]
        )
        for (c, (j, title, color)) in enumerate(tiles)
            ax = Axis(
                fig[r + 1, c]; width = px, height = px, yreversed = true, titlesize = 11, title,
                titlecolor = color
            )
            hidedecorations!(ax); hidespines!(ax)
            image!(ax, rgb(tile(j)))
        end
    end
    colgap!(fig.layout, 1, 24)                   # after the error column
    colgap!(fig.layout, 1 + per, 24)              # between predicted and true examples
    rowgap!(fig.layout, 10)
    resize_to_layout!(fig)
    return save(fname("incorrect", mode), fig; px_per_unit = 1)
end

function reliability(T, mode; bins = 15)
    correct = pred .== labels
    edges = range(0, 1; length = bins + 1)
    bin = clamp.(ceil.(Int, conf .* bins), 1, bins)
    rows = map(1:bins) do b
        in_b = bin .== b
        n = count(in_b)
        (;
            x = (edges[b] + edges[b + 1]) / 2, n, acc = count(in_b .& correct) / max(n, 1),
            conf = sum(conf[in_b]) / max(n, 1),
        )
    end
    used = filter(r -> r.n > 0, rows)
    ece = sum(r.n * abs(r.conf - r.acc) for r in used) / length(conf)
    fig = Figure(size = (460, 420))
    ax = Axis(
        fig[1, 1]; xlabel = "confidence", ylabel = "accuracy", limits = (0, 1, 0, 1),
        title = "Reliability (test), ECE = $(round(100ece; digits = 2))%"
    )
    lines!(ax, [0, 1], [0, 1]; color = T.muted, linestyle = :dash)
    # A bar below the diagonal is right less often than its confidence claims (overconfident).
    xs = [r.x for r in used]
    under, over = T.bar, T.cls[3]
    barplot!(
        ax, xs, [r.acc for r in used]; width = 1 / bins, gap = 0.05,
        color = [r.acc < r.conf ? over : under for r in used]
    )
    axislegend(
        ax, [PolyElement(color = under), PolyElement(color = over)],
        ["underconfident: accuracy above confidence", "overconfident: accuracy below confidence"];
        position = :lt, labelsize = 10
    )
    text!(
        ax, xs, [min(r.acc, 0.93) + 0.02 for r in used]; text = [string(r.n) for r in used],
        align = (:center, :bottom), fontsize = 9, rotation = pi / 2, color = T.ink
    )
    return save(fname("reliability", mode), fig; px_per_unit = 1)
end

# Correct and wrong cells do different jobs, so they get different scales, as in the post's
# interactive version (figures/confusion-matrix.js): blue for correct over the recall range, red
# for errors on a log scale. Saturation carries the amount in both themes: each ramp runs from a
# near-gray tint to the most vivid colour of its hue.
const CM_RAMPS = Dict(
    "light" => (;
        good = ["#e2ecfc", "#d0e0fa", "#bfd4f8", "#aec9f5", "#9dbdf2", "#8cb1ee", "#7ba5eb", "#6a99e7", "#598de3", "#4881df", "#3674db", "#2267d7", "#0559d2"],
        bad = ["#fce5e2", "#fbd7d2", "#fac9c2", "#f9bab2", "#f7aca3", "#f49d93", "#f28e83", "#ee7f74", "#eb6f64", "#e65e54", "#e24c44", "#dd3633", "#d8151e"],
    ),
    "dark" => (;
        good = ["#272e3b", "#2a3549", "#2c3d57", "#2f4466", "#324b75", "#355384", "#375a94", "#3a62a4", "#3c69b4", "#3e71c4", "#4178d5", "#4380e6", "#4588f7"],
        bad = ["#3a2927", "#482d29", "#57302c", "#65332e", "#743530", "#843831", "#933a33", "#a33b35", "#b33d36", "#c33d37", "#d43e38", "#e43e39", "#f53e39"],
    ),
)
function ramp_at(ramp, s)                  # s in [0, 1], interpolated between steps
    f = 1 + clamp(s, 0, 1) * (length(ramp) - 1); a = floor(Int, f); b = min(a + 1, length(ramp))
    x, y, w = parse(Makie.RGBf, ramp[a]), parse(Makie.RGBf, ramp[b]), Float32(f - a)
    return Makie.RGBf(x.r + w * (y.r - x.r), x.g + w * (y.g - x.g), x.b + w * (y.b - x.b))
end
lin(c) = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4
relum(c) = 0.2126lin(c.r) + 0.7152lin(c.g) + 0.0722lin(c.b)
contrast(a, b::String) = contrast(a, parse(Makie.RGBf, b))
contrast(a, b) = ((x, y) = extrema((relum(a), relum(b))); (y + 0.05) / (x + 0.05))
function confusion(T, mode)
    fig = Figure(size = (760, 720), fontsize = 13)
    ax = Axis(
        fig[1, 1:2]; title = "Test confusion matrix, share of the true class (%)", titlesize = 16,
        xlabel = "predicted class", ylabel = "true class (tiles)",
        xticks = (1:K, CLASSES), xticklabelrotation = pi / 4,
        yticks = (1:K, ["$(CLASSES[k]) ($(sum(C[k, :])))" for k in 1:K]),
        yreversed = true, aspect = DataAspect(), xgridvisible = false, ygridvisible = false,
        leftspinevisible = false, bottomspinevisible = false, topspinevisible = false,
        rightspinevisible = false, xticksvisible = false, yticksvisible = false
    )
    d = [share[k, k] for k in 1:K]
    o = [share[t, p] for t in 1:K, p in 1:K if t != p && C[t, p] > 0]
    dlo, dhi = extrema(d); olo, ohi = extrema(o)
    empty = mode == "light" ? "#f3f5f7" : "#161a20"
    for t in 1:K, p in 1:K
        v = share[t, p]
        col = C[t, p] == 0 ? parse(Makie.RGBf, empty) :
            t == p ? ramp_at(CM_RAMPS[mode].good, (v - dlo) / (dhi - dlo)) :
            ramp_at(CM_RAMPS[mode].bad, log(v / olo) / log(ohi / olo))
        poly!(ax, Rect2f(p - 0.47, t - 0.47, 0.94, 0.94); color = col, strokewidth = 0)
        C[t, p] == 0 && continue
        alt = mode == "light" ? "#ffffff" : "#0f1115"
        ink = contrast(col, T.ink) >= contrast(col, alt) ? T.ink : alt
        text!(
            ax, p, t; text = v > 0.9995 ? "100" : string(round(100v; digits = 1)),
            align = (:center, :center), fontsize = 12, color = ink
        )
    end
    pcts(x) = x > 0.9995 ? "100%" : string(round(100x; digits = 1), "%")
    Colorbar(
        fig[2, 1]; colormap = [ramp_at(CM_RAMPS[mode].good, s) for s in 0:0.05:1], limits = (dlo, dhi),
        vertical = false, flipaxis = false, label = "correct: recall (linear)",
        ticks = ([dlo, dhi], pcts.([dlo, dhi])), width = 220
    )
    Colorbar(
        fig[2, 2]; colormap = [ramp_at(CM_RAMPS[mode].bad, s) for s in 0:0.05:1], limits = (log(olo), log(ohi)),
        vertical = false, flipaxis = false, label = "errors (log scale)",
        ticks = ([log(olo), log(ohi)], pcts.([olo, ohi])), width = 220
    )
    colgap!(fig.layout, 70)
    return save(fname("confusion", mode), fig; px_per_unit = 1.5)
end

corner(k) = Point2f(sinpi(2(k - 1) / K), cospi(2(k - 1) / K))

function nonagon(T, mode)
    pos(p) = sum(p[k] * corner(k) for k in 1:K)
    fig = Figure(size = (720, 720))
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "Class probability along the ODE trajectory")
    hidedecorations!(ax); hidespines!(ax)
    lines!(ax, [corner.(1:K); [corner(1)]]; color = T.grid)
    for n in axes(P, 2)
        pts = [pos(view(P, :, n, t)) for t in axes(P, 3)]
        lines!(ax, pts; color = (T.cls[plabels[n]], 0.4), linewidth = 1)
        scatter!(ax, [last(pts)]; color = T.cls[plabels[n]], markersize = 4)
    end
    for k in 1:K
        text!(ax, 1.12f0 * corner(k); text = CLASSES[k], align = (:center, :center), fontsize = 14, color = T.ink)
    end
    limits!(ax, -1.35, 1.35, -1.35, 1.35)
    return save(fname("nonagon", mode), fig; px_per_unit = 1)
end

function flows(T, mode, sel)
    ts = range(0, 1; length = size(P, 3))
    fig = Figure(size = (760, 200 * length(sel)))
    for (row, n) in enumerate(sel)
        pk = argmax(view(P, :, n, size(P, 3)))
        ax_i = Axis(
            fig[row, 1]; aspect = DataAspect(), yreversed = true, titlesize = 12,
            title = "true: $(CLASSES[plabels[n]])\npred: $(CLASSES[pk])",
            titlecolor = pk == plabels[n] ? T.ink : T.err
        )
        hidedecorations!(ax_i); hidespines!(ax_i)
        image!(ax_i, rgb(tile(picks[n])))
        ax = Axis(fig[row, 2]; xlabel = row == length(sel) ? "t" : "", ylabel = "p(class)", limits = (0, 1, 0, 1))
        lower = zeros(length(ts))
        for k in 1:K
            upper = lower .+ P[k, n, :]
            band!(ax, ts, lower, upper; color = T.cls[k], label = CLASSES[k])
            lower = upper
        end
        row == 1 && Legend(fig[1:length(sel), 3], ax)
    end
    colsize!(fig.layout, 1, Fixed(170))
    return save(fname("flows", mode), fig; px_per_unit = 1)
end

# The same tiles as before: a typical correct prediction per class, the 12 most confident
# errors, and for the flows one correct tile from four classes plus the first two errors.
wrong = findall(pred .!= labels)
worst = wrong[sortperm(conf[wrong]; rev = true)][1:12]
typical = map(1:K) do k
    idx = findall((labels .== k) .& (pred .== k)); idx[sortperm(conf[idx])][cld(length(idx), 2)]
end
pend = [argmax(view(P, :, n, size(P, 3))) for n in axes(P, 2)]
ok = findall(pend .== plabels); bad = findall(pend .!= plabels)
sel = vcat([first(filter(n -> plabels[n] == k, ok)) for k in (1, 4, 8, 6)], first(bad, 2))

mkpath(OUT)
for (mode, T) in THEMES
    with_theme(makie_theme(T)) do
        prediction_grid(T, mode, typical, "Correct: a typical test prediction per class", "correct"; cols = 3)
        error_comparison(T, mode, worst)
        reliability(T, mode); confusion(T, mode); nonagon(T, mode); flows(T, mode, sel)
    end
end

# ── Data for the interactive figures ─────────────────────────────────────────────
const CODES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
jstr(x::AbstractString) = "\"" * escape_string(x) * "\""
jarr(xs) = "[" * join(xs, ",") * "]"

# trajectories.json (trajectory-player.js): p as bytes, p[(t * N + n) * K + k] = round(255 p).
let q = round.(UInt8, clamp.(P, 0, 1) .* 255)            # (K, N, T) column-major is that order
    write(
        joinpath(OUT, "trajectories.json"), string(
            "{\"classes\":", jarr(jstr.(CLASSES)),
            ",\"labels\":", jarr(plabels .- 1), ",\"K\":", K, ",\"N\":", size(P, 2), ",\"T\":", size(P, 3),
            ",\"p\":", jstr(base64encode(vec(q))), "}\n"
        )
    )
end

# confusion.json (confusion-matrix.js): counts[true][predicted].
write(
    joinpath(OUT, "confusion.json"), string(
        "{\"classes\":", jarr(jstr.(CLASSES)), ",\"codes\":",
        jarr(jstr.(CODES)), ",\"counts\":", jarr([jarr(C[t, :]) for t in 1:K]), "}\n"
    )
)

# hero/tiles.jpg and hero/tiles.json (hero.js): 48 test tiles, five confident correct ones per
# class and three confident errors, shuffled, as an 8 x 6 sprite of 112 px tiles.
let rng = Xoshiro(7), S = 112, cols = 8
    good = [shuffle(rng, findall((labels .== k) .& (pred .== k) .& (conf .> 0.9)))[1:5] for k in 1:K]
    bad = shuffle(rng, findall((pred .!= labels) .& (conf .> 0.8)))[1:3]
    sel = shuffle(rng, vcat(good..., bad))
    rows = cld(length(sel), cols)
    sheet = zeros(UInt8, 3, cols * S, rows * S)
    for (j, i) in enumerate(sel)
        ox, oy = ((j - 1) % cols) * S, ((j - 1) ÷ cols) * S
        sheet[:, ox .+ (1:S), oy .+ (1:S)] = images[:, 1:2:end, 1:2:end, i]   # 2x nearest downsample
    end
    mkpath(joinpath(OUT, "hero"))
    ppm = tempname() * ".ppm"
    open(ppm, "w") do io
        write(io, "P6 $(cols * S) $(rows * S) 255\n", vec(sheet))
    end
    run(`ffmpeg -loglevel error -y -i $ppm -q:v 7 $(joinpath(OUT, "hero", "tiles.jpg"))`)
    rm(ppm)
    write(
        joinpath(OUT, "hero", "tiles.json"), jarr(
            [
                string(
                    "{\"label\":", labels[i] - 1, ",\"pred\":",
                    pred[i] - 1, ",\"conf\":", round(conf[i]; digits = 3), "}"
                ) for i in sel
            ]
        ) * "\n"
    )
end

# ── numbers.md: every number the post quotes from this model ─────────────────────
let io = IOBuffer(), N = length(labels)
    recall = [C[k, k] / sum(C[k, :]) for k in 1:K]
    correct = pred .== labels
    bins = 15; bin = clamp.(ceil.(Int, conf .* bins), 1, bins)
    ece = sum(abs(sum(conf[bin .== b]) - count(correct .& (bin .== b))) for b in 1:bins) / N
    nwrong = count(!, correct)
    println(io, "# Test-set numbers\n")
    @printf(io, "- macro accuracy %.2f%%, accuracy %.2f%%, ECE %.2f%%\n", 100sum(recall) / K, 100count(correct) / N, 100ece)
    println(io, "- errors: $nwrong of $N")
    counts = [count(==(b), bin) for b in 1:bins]
    println(io, "- reliability: $(counts[end]) tiles in the top bin; largest lower bin holds $(maximum(counts[1:(end - 1)]))")
    @printf(io, "- test loss (cross-entropy): %.3f\n", -sum(log(probs[labels[i], i]) for i in 1:N) / N)
    println(io, "- errors above 99% confidence: ", count(conf[.!correct] .> 0.99))
    pair(i) = Set((labels[i], pred[i])) == Set((6, 8))
    println(io, "- of the 12 most confident errors, stroma/smooth muscle either way: ", count(pair, worst))
    println(io, "\n## Per-class recall\n\n| Class | Recall | Tiles |\n|---|---:|---:|")
    for k in 1:K
        @printf(io, "| %s | %.2f%% | %d |\n", CLASSES[k], 100recall[k], sum(C[k, :]))
    end
    println(io, "\n## Largest confusions\n\n| True | Predicted | Count |\n|---|---|---:|")
    off = sort([(C[t, p], t, p) for t in 1:K, p in 1:K if t != p && C[t, p] > 0]; rev = true)
    for (n, t, p) in off[1:min(8, end)]
        println(io, "| ", CLASSES[t], " | ", CLASSES[p], " | ", n, " |")
    end
    # When a trajectory decides: the first t after which its final class stays above 0.5.
    ts = range(0, 1; length = size(P, 3))
    decide = map(axes(P, 2)) do n
        k = argmax(view(P, :, n, size(P, 3)))
        above = view(P, k, n, :) .> 0.5
        last_below = findlast(!, above)
        last_below === nothing ? 0.0 : last_below == length(ts) ? NaN : ts[last_below + 1]
    end
    d = sort(filter(!isnan, decide))
    q(x) = d[clamp(round(Int, x * length(d)), 1, length(d))]
    @printf(
        io, "\n## Trajectories\n\n- decision time t (final class stays above 0.5): 10%% %.2f, 25%% %.2f, median %.2f, 75%% %.2f, 90%% %.2f\n",
        q(0.1), q(0.25), q(0.5), q(0.75), q(0.9)
    )
    write(joinpath(ASSETS, "numbers.md"), take!(io))
end
println("wrote figures and JSON to ", OUT, ", numbers.md to ", ASSETS)
