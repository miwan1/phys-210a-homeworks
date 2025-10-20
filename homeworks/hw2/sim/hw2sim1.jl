
import Pkg; Pkg.add("Plots"); Pkg.add("StatsBase"); Pkg.add("Distributions")
using Random, Statistics, Plots

# ------------------------
# 1) CTRW core simulator
# ------------------------
"""
    ctrw(tf; wait_sampler, jump_sampler, x0=0.0, rng=Random.default_rng())

Simulate a 1D continuous-time random walk up to final time `tf` WITHOUT using tiny time steps.
- `wait_sampler(rng)`  -> returns a single waiting time τ ≥ 0 (drawn from ψ(τ))
- `jump_sampler(rng)`  -> returns a single spatial jump Δx
- Starts at x(0)=x0.
Returns (t, x) where:
  * t = [0, t1, t2, ..., tf] are event times plus the final time tf
  * x = [x0, x1, x2, ..., x(tf)] are positions (piecewise constant; last value held to tf)
"""
function ctrw(tf; wait_sampler, jump_sampler, x0=0.0, rng=Random.default_rng())
    t = Float64[0.0]
    x = Float64[x0]
    while true
        τ = wait_sampler(rng)
        tnext = t[end] + τ
        if tnext > tf
            # we overshot: append final time and HOLD position (no jump)
            push!(t, tf)
            push!(x, x[end])
            break
        else
            # we jump at tnext
            dx = jump_sampler(rng)
            push!(t, tnext)
            push!(x, x[end] + dx)
        end
    end
    return t, x
end

# -----------------------------------------------------------
# 2) Example samplers: heavy-tailed waits + ±1 jump sizes
# -----------------------------------------------------------
# Pareto (shifted) waiting time:  ψ(τ) ∝ (τ0 + τ)^(-α),  α>1; long tail for α close to 1.
struct ParetoShifted{T}
    τ0::T
    α::T
end
# Inverse-CDF sampler: F(τ)=1-(τ0/(τ0+τ))^(α-1)  ⇒  τ = τ0 * ((1-u)^(-1/(α-1)) - 1)
(P::ParetoShifted)(rng) = P.τ0 * ((1 - rand(rng))^(-1/(P.α - 1)) - 1)

# Symmetric ±1 jumps (you can replace with any distribution)
step_pm1(rng) = (rand(rng) < 0.5 ? -1.0 : 1.0)

# -----------------------------------------------------------
# 3) Minimal, reproducible example
# -----------------------------------------------------------
rng = MersenneTwister(2025)

# Heavy-tailed waits: α = 1.6 (very heavy; infinite mean for α≤2 in this parameterization)
wait = ParetoShifted(0.1, 1.6)

# Simulate up to tf without tiny steps:
tf = 50.0
t, x = ctrw(tf; wait_sampler=wait, jump_sampler=step_pm1, rng=rng)

println("CTRW finished with $(length(t)-1) jumps by t = $tf.")
println("First few event times: ", t[1:min(end,6)])
println("First few positions:    ", x[1:min(end,6)])

# -----------------------------------------------------------
# 4) (Optional) Resample on a regular grid for analysis/plotting
# -----------------------------------------------------------
"""
    hold_resample(t_events, y_events, t_grid)

Piecewise-constant resample: for each t in t_grid, return y at the last event time ≤ t.
"""
function hold_resample(t_events, y_events, t_grid)
    y = similar(t_grid, eltype(y_events))
    j = 1
    cur = y_events[1]
    for i in eachindex(t_grid)
        tg = t_grid[i]
        while j <= length(t_events) && t_events[j] <= tg
            cur = y_events[j]
            j += 1
        end
        y[i] = cur
    end
    y
end

# Example resample onto uniform grid (for plotting or statistics)
tgrid = 0.0:0.1:tf
xgrid = hold_resample(t, x, collect(tgrid))

# If you want a quick plot, uncomment the following:
default(size=(800,400), lw=2)
plot(tgrid, xgrid, seriestype=:steppost, xlabel="time", ylabel="position",
     title="CTRW with heavy-tailed waits")
