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

"""
    node_head(e::PathMNIST, width)

The classifier head, an augmented NeuralODE whose state carries its own logits. The state
is `[a; z]`: `a` has `e.aug_dims` dimensions predicted from the pooled backbone features,
and `z` has one dimension per class and always starts at zero. An MLP vector field
(encoder, tanh, decoder) moves both with RK4, and the logits are `z(1)`, read directly
with no classifier layer. So every prediction starts uniform and only the flow can carry
it to a class.

The decoder starts at zero, so at initialization du/dt = 0, the logits stay at zero, and
the first loss is exactly log(num_classes); the dynamics grow from there.

Returns the logits with two per-sample diagnostics of the dynamics, each `(1, N)`: the
trajectory's kinetic energy, which the loss can penalize, and how far the state moved,
`|u(1) - u(0)| / |u(0)|`.
"""
function node_head(e::PathMNIST, width::Integer)
    # Captured by the closure rather than passed to @compact, which would put them in the
    # layer's state.
    steps, A, K = e.node_steps, e.aug_dims, e.num_classes
    D = A + K
    return @compact(
        proj = Dense(width => A),
        enc = Dense(D => e.node_hidden, tanh),
        dec = Dense(e.node_hidden => D; init_weight = zeros32, init_bias = zeros32),
    ) do pooled                            # (width, N)
        a0 = proj(pooled)
        # z(0) = 0, built by broadcasting so it is a constant under trace as well as eagerly.
        u0 = vcat(a0, zeros(eltype(a0), K) .* a0[1:1, :])
        (; u, energy) = rk4(u -> dec(enc(u)), u0, steps)
        motion = sqrt.(sum(abs2, u .- u0; dims = 1) ./ sum(abs2, u0; dims = 1))
        @return (; logits = u[(A + 1):D, :], energy, motion)
    end
end
