# Fixed-step RK4 for du/dt = f(u) over t in [0, 1]. Returns the final state and, per
# sample, the kinetic energy of the trajectory, the integral of |f(u)|² dt, by the same
# RK4 weights over the four stage evaluations it already computes. The field is
# autonomous, so `t` never enters. Under Reactant the loop unrolls at trace time into one
# static program, which XLA then fuses.
function rk4(f, u, steps::Integer)
    T = eltype(u)
    dt = one(T) / T(steps)
    half = dt / T(2)
    sixth = dt / T(6)
    sq(k) = sum(abs2, k; dims = 1)
    energy = zero(sq(u))
    for _ in 1:steps
        k1 = f(u)
        k2 = f(u .+ half .* k1)
        k3 = f(u .+ half .* k2)
        k4 = f(u .+ dt .* k3)
        u = u .+ sixth .* (k1 .+ T(2) .* k2 .+ T(2) .* k3 .+ k4)
        energy = energy .+ sixth .* (sq(k1) .+ T(2) .* sq(k2) .+ T(2) .* sq(k3) .+ sq(k4))
    end
    return (; u, energy)
end

# A small random init for the classifier: N(0, 0.01²).
small_init(rng, dims...) = 0.01f0 .* randn(rng, Float32, dims...)

"""
    node_head(e::PathMNIST, width)

The classifier head: project the pooled backbone features into the ODE state, integrate
an MLP vector field (encoder, tanh, decoder) with RK4, and classify the final state.
Returns the logits with two per-sample diagnostics of the dynamics, each `(1, N)`: the
trajectory's kinetic energy, which the loss penalizes, and how far the state moved,
`|u(1) - u(0)| / |u(0)|`.

The decoder starts at zero, so du/dt = 0 at initialization and the ODE begins as the
identity: training starts from the backbone's features, and the dynamics grow from
there. The classifier starts small, so the first logits are near uniform.
"""
function node_head(e::PathMNIST, width::Integer)
    # Captured by the closure rather than passed to @compact, which would put it in the
    # layer's state.
    steps = e.node_steps
    return @compact(
        proj = Dense(width => e.node_dims),
        enc = Dense(e.node_dims => e.node_hidden, tanh),
        dec = Dense(e.node_hidden => e.node_dims; init_weight = zeros32, init_bias = zeros32),
        cls = Dense(e.node_dims => e.num_classes; init_weight = small_init),
    ) do pooled                            # (width, N)
        u0 = proj(pooled)
        (; u, energy) = rk4(u -> dec(enc(u)), u0, steps)
        motion = sqrt.(sum(abs2, u .- u0; dims = 1) ./ sum(abs2, u0; dims = 1))
        @return (; logits = cls(u), energy, motion)
    end
end
