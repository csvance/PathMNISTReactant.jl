# Visualize how the NeuralODE head moves a prediction between the 9 classes.
#
# The logits are dimensions of the ODE state, so a prediction can be read off at any
# time: p(t) = softmax(z(t)), starting uniform at t = 0. This script loads a trained
# checkpoint, compiles one extra program that returns p(t) along the RK4 trajectory
# (separate from `forward`, which returns only the end point), and draws two figures:
#
#   nonagon.png   the 9 classes on the corners of a regular 9-gon; each sample's path runs
#                 from sum_k p_k(0) corner_k to sum_k p_k(1) corner_k, coloured by true class
#   flows.png     for a few chosen tiles, the tile beside a stacked plot of p_k(t)
#
#   julia -t 16,1 --project=<env with PathMNISTReactant, CairoMakie> scripts/trajectories.jl \
#       <run_dir> <data_dir> <out_dir>
#
# The checkpoint is the best one in <run_dir> by the experiment's checkpointer metric.
using PathMNISTReactant, ReactantNitro, Reactant, Lux, CairoMakie, Random, Statistics
using PathMNISTReactant: to_float32, CLASS_NAMES
using PathMNISTReactant.MLUtils: getobs

const SAMPLE_EVERY = 2          # record p(t) every this many RK4 steps

# The model's forward pass up to the ODE, then RK4 recording the state. Mirrors
# `build_model` and `node_head`; reads the sub-layers out of the @compact containers.
function class_trajectory(e, model, ps, st, img)
    L, H = model.layers, model.layers.head.layers
    sH = st.head
    μ = reshape(collect(e.pixel_mean), 1, 1, 3, 1)
    σ = reshape(collect(e.pixel_std), 1, 1, 3, 1)
    x01 = to_float32(img)
    feats, _ = L.backbone((x01 .- μ) ./ σ, ps.backbone, st.backbone)
    pooled = dropdims(mean(feats; dims = (1, 2)); dims = (1, 2))
    pooled, _ = L.norm(pooled, ps.norm, st.norm)
    if e.wavelet_levels > 0
        xw = e.wavelet_space === :hed ? PathMNISTReactant.hed_density(x01) : x01
        w, _ = L.wnorm(PathMNISTReactant.wavelet_features(xw, e.wavelet_levels), ps.wnorm, st.wnorm)
        pooled = vcat(pooled, w)
    end
    apply(name, v) = first(H[name](v, ps.head[name], sH[name]))
    f(u) = apply(:dec, apply(:enc, u))
    A = e.aug_dims
    probs(u) = softmax(u[(A + 1):end, :]; dims = 1)       # the logits live in the state
    a0 = apply(:proj, pooled)
    u = vcat(a0, zeros(eltype(a0), e.num_classes) .* a0[1:1, :])
    T, K = eltype(u), e.node_steps
    dt = one(T) / T(K)
    path = [probs(u)]
    for k in 1:K
        k1 = f(u)
        k2 = f(u .+ (dt / 2) .* k1)
        k3 = f(u .+ (dt / 2) .* k2)
        k4 = f(u .+ dt .* k3)
        u = u .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        k % SAMPLE_EVERY == 0 && push!(path, probs(u))
    end
    return cat((reshape(p, size(p)..., 1) for p in path)...; dims = 3)   # (classes, N, times)
end

corner(k, K = 9) = Point2f(sinpi(2(k - 1) / K), cospi(2(k - 1) / K))

function nonagon(P, labels; path)
    K = size(P, 1)
    pos(p) = sum(p[k] * corner(k, K) for k in 1:K)
    colors = Makie.wong_colors()
    colors = [colors; RGBf(0.4, 0.4, 0.4); RGBf(0.6, 0.3, 0.6)][1:K]
    fig = Figure(size = (720, 720))
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "Class probability along the ODE trajectory")
    hidedecorations!(ax); hidespines!(ax)
    lines!(ax, [corner.(1:K); [corner(1)]]; color = :gray80)
    for n in axes(P, 2)
        pts = [pos(view(P, :, n, t)) for t in axes(P, 3)]
        lines!(ax, pts; color = (colors[labels[n]], 0.35), linewidth = 1)
        scatter!(ax, [last(pts)]; color = colors[labels[n]], markersize = 4)
    end
    for k in 1:K
        text!(ax, 1.12f0 * corner(k, K); text = CLASS_NAMES[k], align = (:center, :center),
            color = colors[k], fontsize = 14)
    end
    limits!(ax, -1.35, 1.35, -1.35, 1.35)
    save(path, fig; px_per_unit = 1)
end

function flows(P, tiles, labels; path)
    K, N = size(P, 1), size(P, 2)
    ts = range(0, 1; length = size(P, 3))
    colors = Makie.wong_colors()
    colors = [colors; RGBf(0.4, 0.4, 0.4); RGBf(0.6, 0.3, 0.6)][1:K]
    fig = Figure(size = (760, 200 * N))
    for n in 1:N
        img = tiles[:, :, :, n]
        pred = argmax(view(P, :, n, size(P, 3)))
        ax_i = Axis(fig[n, 1]; aspect = DataAspect(), yreversed = true,
            title = "true: $(CLASS_NAMES[labels[n]])\npred: $(CLASS_NAMES[pred])", titlesize = 12,
            titlecolor = pred == labels[n] ? :black : :firebrick)
        hidedecorations!(ax_i); hidespines!(ax_i)
        image!(ax_i, [RGBf(img[x, y, 1] / 255, img[x, y, 2] / 255, img[x, y, 3] / 255)
                      for x in axes(img, 1), y in axes(img, 2)])
        ax = Axis(fig[n, 2]; xlabel = n == N ? "t" : "", ylabel = "p(class)", limits = (0, 1, 0, 1))
        lower = zeros(length(ts))
        for k in 1:K
            upper = lower .+ P[k, n, :]
            band!(ax, ts, lower, upper; color = colors[k], label = CLASS_NAMES[k])
            lower = upper
        end
        n == 1 && Legend(fig[1:N, 3], ax; framevisible = false)
    end
    colsize!(fig.layout, 1, Fixed(170))
    save(path, fig; px_per_unit = 1)
end

function main(run_dir, data_dir, out_dir; n_overview = 256, picks = nothing)
    e = PathMNIST(; data_dir)
    nitro = Nitro(e; run_dir, weights = :latest => :best)
    test = PathMNISTSplit(e, "test")
    rng = Xoshiro(1)
    idx = picks === nothing ? randperm(rng, length(test.labels))[1:n_overview] : picks
    batch = getobs(test, idx)
    img = Reactant.to_rarray(batch.img)
    traj = @compile class_trajectory(nitro.e, nitro.model, nitro.ps, nitro.st, img)
    P = Array(traj(nitro.e, nitro.model, nitro.ps, nitro.st, img))
    mkpath(out_dir)
    nonagon(P, batch.label; path = joinpath(out_dir, "nonagon.png"))
    return (; P, batch, idx)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
