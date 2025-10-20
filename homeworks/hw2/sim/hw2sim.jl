import Pkg; Pkg.add("Plots"); Pkg.add("StatsBase"); Pkg.add("Distributions")
using Random, Statistics, Plots

default(size=(800,450), legend=:best, lw=2)

# ---------- (Reaction) Gillespie simulator for 2M <-> D ----------
"""
    gillespie_dimer(; N, kb=1.0, ku=2.0, tf=5.0, rng)

Discrete propensities:
  a_bind   = kb * M*(M-1)   (2M -> D)
  a_unbind = ku * D         (D -> 2M)
Initial: M(0)=N, D(0)=0.  Returns (t, M, D) vectors.
"""
function gillespie_dimer(; N::Int, kb::Float64=1.0, ku::Float64=2.0,
                         tf::Float64=5.0, rng=Random.default_rng())
    t = Float64[0.0]; M = Int[N]; D = Int[0]
    while t[end] < tf
        m, d = M[end], D[end]
        a1 = (m > 1) ? kb * m * (m - 1) : 0.0  # 2M -> D
        a2 = ku * d                            # D -> 2M
        a0 = a1 + a2
        if a0 <= 0.0
            push!(t, tf); push!(M, m); push!(D, d); break
        end
        τ = -log(rand(rng)) / a0
        tnext = t[end] + τ
        if tnext > tf
            push!(t, tf); push!(M, m); push!(D, d); break
        end
        r = rand(rng) * a0
        if r < a1      # bind
            m -= 2; d += 1
        else           # unbind
            m += 2; d -= 1
        end
        push!(t, tnext); push!(M, m); push!(D, d)
    end
    return t, M, D
end

# ---------- Deterministic (continuum) closed-form solution ----------
# dM/dt = -2 kb M^2 - ku M + ku N,  M(0)=N
function M_det(t; N::Float64, kb::Float64=1.0, ku::Float64=2.0)
    disc = sqrt(ku^2 + 8kb*ku*N)
    Mplus  = (-ku + disc) / (4kb)   # stable equilibrium
    Mminus = (-ku - disc) / (4kb)   # < 0
    Q0 = (N - Mplus) / (N - Mminus)
    Q = Q0 * exp(-2*(Mplus - Mminus)*t)
    (Mplus - Q*Mminus) / (1 - Q)
end
D_det(t; N::Float64, kb::Float64=1.0, ku::Float64=2.0) = (N - M_det(t; N=N, kb=kb, ku=ku)) / 2

# ---------- helpers for ensembles ----------
"""
    resample_last_hold(t_src, y_src, t_grid)

Piecewise-constant resampling of jump process onto a regular grid.
"""
function resample_last_hold(t_src::Vector{<:Real}, y_src::Vector{T}, t_grid::Vector{<:Real}) where {T}
    out = similar(t_grid, T); i = 1; cur = y_src[1]
    for (k, tg) in enumerate(t_grid)
        while i <= length(t_src) && t_src[i] <= tg
            cur = y_src[i]; i += 1
        end
        out[k] = cur
    end
    out
end

"""
    ensemble_stats(; N, kb=1.0, ku=2.0, tf=5.0, nruns=1000, dt=0.01, rng)

Return (tgrid, Mmean, Dmean, Mstd, Dstd).
"""
function ensemble_stats(; N::Int, kb::Float64=1.0, ku::Float64=2.0,
                        tf::Float64=5.0, nruns::Int=500, dt::Float64=0.01,
                        rng=Random.default_rng())
    tgrid = collect(0.0:dt:tf)
    Ms = zeros(length(tgrid)); Ds = zeros(length(tgrid))
    Ms2 = zeros(length(tgrid)); Ds2 = zeros(length(tgrid))
    for _ in 1:nruns
        t, M, D = gillespie_dimer(N=N, kb=kb, ku=ku, tf=tf, rng=rng)
        Mr = resample_last_hold(t, M, tgrid)
        Dr = resample_last_hold(t, D, tgrid)
        @inbounds for k in eachindex(tgrid)
            Ms[k]  += Mr[k];  Ds[k]  += Dr[k]
            Ms2[k] += Mr[k]^2; Ds2[k] += Dr[k]^2
        end
    end
    Mmean = Ms ./ nruns; Dmean = Ds ./ nruns
    Mstd  = sqrt.((Ms2 ./ nruns) .- Mmean.^2)
    Dstd  = sqrt.((Ds2 ./ nruns) .- Dmean.^2)
    tgrid, Mmean, Dmean, Mstd, Dstd
end

# Choose N values and final time
rng = MersenneTwister(42)
for N in (2, 90, 10000)
    tf = 5.0
    # stochastic trajectory
    t, M, D = gillespie_dimer(N=N, tf=tf, rng=rng)
    # deterministic on a dense grid
    t_dense = range(0, tf, length=400)
    M_det_vals = [M_det(tt; N=Float64(N)) for tt in t_dense]
    D_det_vals = [D_det(tt; N=Float64(N)) for tt in t_dense]

    p1 = plot(t, M, seriestype=:steppost, label="stochastic M(t)", xlabel="time",
              ylabel="count", title="N = $N : monomers")
    plot!(p1, t_dense, M_det_vals, label="continuum M(t)", ls=:dash)
    p2 = plot(t, D, seriestype=:steppost, label="stochastic D(t)", xlabel="time",
              ylabel="count", title="N = $N : dimers")
    plot!(p2, t_dense, D_det_vals, label="continuum D(t)", ls=:dash)
    display(plot(p1, p2, layout=(1,2)))
end

# Compare ensemble average vs continuum for N = 2 and N = 90
# for N in (2, 90)
#     tf = 5.0
#     t, Mmean, Dmean, Mstd, Dstd = ensemble_stats(N=N, tf=tf, nruns=1000, dt=0.01,
#                                                  rng=MersenneTwister(123))
#     Mdet = [M_det(tt; N=Float64(N)) for tt in t]
#     Ddet = [D_det(tt; N=Float64(N)) for tt in t]

#     # ribbon = ±std
#     pM = plot(t, Mmean, ribbon=Mstd, label="ensemble ⟨M⟩ ± σ", xlabel="time",
#               ylabel="count", title="Ensemble vs ODE (N = $N)")
#     plot!(pM, t, Mdet, label="continuum M(t)", ls=:dash, c=:black)

#     pD = plot(t, Dmean, ribbon=Dstd, label="ensemble ⟨D⟩ ± σ", xlabel="time",
#               ylabel="count", title="Ensemble vs ODE (N = $N)")
#     plot!(pD, t, Ddet, label="continuum D(t)", ls=:dash, c=:black)

#     display(plot(pM, pD, layout=(1,2)))
# end

# Quick sweep: smallest N with late-time fluctuations < ±20%
# function late_time_rel_fluct(N; tf=5.0, nruns=600)
#     t, Mmean, Dmean, Mstd, _ = ensemble_stats(N=N, tf=tf, nruns=nruns, dt=0.01,
#                                               rng=MersenneTwister(7))
#     L = round(Int, 0.9*length(t)):length(t)          # last 10% window
#     rel = mean(Mstd[L] ./ max.(Mmean[L], 1e-9))
#     rel
# end

# Ns = [2,4,6,8,10,15,20,30,40,60,90,120,200]
# rels = [late_time_rel_fluct(N; tf=5.0, nruns=400) for N in Ns]

# p_rel = plot(Ns, 100 .* rels, m=:o, xlabel="N", ylabel="late-time RMS fluctuation of M (%)",
#              title="Fluctuations vs system size", ylim=(0,100))
# hline!(p_rel, [20.0], ls=:dash, c=:red, label="20%")
# display(p_rel)

# minN = findfirst(>(0.20), 0.20 .< rels) === nothing ? nothing :
#        Ns[findfirst(x->x<0.20, rels)]
# println("Approx. smallest N with late-time fluctuations < 20%: ",
#         something(minN, "not reached within tested Ns"))

# -------- CTRW samplers --------
struct ParetoShifted{T}
    τ0::T
    α::T
end
# Inverse-CDF for shifted Pareto: F(τ)=1-(τ0/(τ0+τ))^(α-1)
(P::ParetoShifted)(rng) = P.τ0 * ((1 - rand(rng))^(-1/(P.α - 1)) - 1)

step_±1(rng) = (rand(rng) < 0.5 ? -1.0 : 1.0)

function ctrw(; tf::Float64, wait_sampler, step_sampler, x0::Float64=0.0,
              rng=Random.default_rng())
    t = Float64[0.0]; x = Float64[x0]; waits = Float64[]
    while t[end] < tf
        τ = wait_sampler(rng)
        tnext = t[end] + τ
        if tnext > tf
            push!(t, tf); push!(x, x[end]); break
        else
            dx = step_sampler(rng)
            push!(waits, τ)
            push!(t, tnext); push!(x, x[end] + dx)
        end
    end
    return t, x, waits
end

# -------- Example CTRW run and plots --------
rng = MersenneTwister(2025)
wait = ParetoShifted(0.1, 1.6)             # heavy-tailed (α=1.6)
tf = 50.0
t_ctrw, x_ctrw, waits = ctrw(tf=tf, wait_sampler=wait, step_sampler=step_±1, rng=rng)

p_traj = plot(t_ctrw, x_ctrw, seriestype=:steppost, xlabel="time", ylabel="position",
              title="CTRW trajectory (heavy-tailed waits)")
p_hist = histogram(waits, bins=50, xlabel="waiting time τ", ylabel="count",
                   title="Sampled waiting times", normalize=false)
display(plot(p_traj, p_hist, layout=(1,2)))

println("CTRW: $(length(waits)) jumps by t = $tf.")
