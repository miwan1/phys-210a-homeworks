############################################################
# Double-well Langevin simulation (overdamped, discrete)
#
# y_{n+1} = y_n + α E y_n (1 - y_n^2) + sqrt(α/2) * η_n
# with η_n ~ N(0,1)
############################################################

import Pkg; Pkg.activate(temp=true); Pkg.add("Plots"); Pkg.add("StatsBase"); Pkg.add("Distributions"); Pkg.add("Statistics"); Pkg.add("Random") 

using Random
using Statistics
using Distributions
using Plots
using StatsBase

# ---------------------------
# (1) Basic parameters
# ---------------------------
# physical-ish parameters (dimensionless form)
α  = 5e-3            # small number: α = 4 T Δt / (γ x0^2)
E  = 5.0             # E = U0 / T   (barrier height in kT units)
Nsteps = 2_000_000   # total steps for parts (c),(d) (you can lower while testing)
y0 = 1.0             # start in right well

# potential in dimensionless y: U(y)/T = E (1 - y^2)^2
U_over_T(y, E) = E * (1 - y^2)^2

# ---------------------------
# (2) One Euler–Maruyama step
# ---------------------------
function step_y(y, α, E)
    drift = α * E * y * (1 - y^2)
    noise = sqrt(α/2) * randn()
    return y + drift + noise
end

# ---------------------------
# (b) Simulate trajectory
# ---------------------------
function simulate_trajectory(y0, α, E, Nsteps)
    y = y0
    ys = Vector{Float64}(undef, Nsteps)
    for n in 1:Nsteps
        y = step_y(y, α, E)
        ys[n] = y
    end
    return ys
end

println("Running main trajectory...")
ys = simulate_trajectory(y0, α, E, Nsteps)

############################################################
# (c) Check numerically that P(x) ∝ exp[-U(x)/T]
# Here x = x0 * y, but in our dimensionless version we just check y.
# We compare the histogram of y to the theoretical Boltzmann weight
#   P(y) ∝ exp( - E (1 - y^2)^2 )
############################################################

burnin = 50_000
ys_eq = ys[burnin+1:end]

nbins = 100
# this is the key change:
hist_emp = StatsBase.fit(StatsBase.Histogram, ys_eq; nbins=nbins)
# or: hist_emp = StatsBase.Histogram(ys_eq; nbins=nbins)

# bin centers
edges = hist_emp.edges[1]
bincenters = 0.5 .* (edges[1:end-1] .+ edges[2:end])

# theoretical Boltzmann in y
U_over_T(y, E) = E * (1 - y^2)^2
p_theo_unnorm = @. exp( - U_over_T(bincenters, E) )
p_theo = p_theo_unnorm .* (sum(hist_emp.weights) / sum(p_theo_unnorm))

plt_c = plot(
    bincenters,
    hist_emp.weights,
    seriestype = :steppre,
    label = "empirical",
    xlabel = "y",
    ylabel = "counts",
    title = "Empirical vs Boltzmann, E = $(E)",
)
plot!(plt_c, bincenters, p_theo, label = "Boltzmann")
display(plt_c)

############################################################
# (d) Jump rate vs E
#
# Idea:
#  - For each E, run a trajectory
#  - Count how many times we jump from y < 0 to y > 0 or vice versa
#  - Divide by total time to get a rate
#  - Plot rate vs E and see if it looks Arrhenius-like
############################################################

function count_jumps(ys; thresh = 0.0, dwell = 50)
    # crude but robust-ish: we only count a crossing if we've been on one side
    # for at least `dwell` steps and then later on the other side for `dwell` steps
    side(z) = z > thresh ? 1 : (z < -thresh ? -1 : 0)
    s_prev = side(ys[1])
    count = 0
    same_side_run = 1

    for k in 2:length(ys)
        s = side(ys[k])
        if s == s_prev && s != 0
            same_side_run += 1
        elseif s != s_prev && s != 0 && same_side_run >= dwell
            # we changed stable side
            count += 1
            s_prev = s
            same_side_run = 1
        else
            # reset the run if we are in the barrier region (|y| small)
            same_side_run = 1
            if s != 0
                s_prev = s
            end
        end
    end
    return count
end

# scan over E
Es = 2.0:1.0:10.0    # choose a range of barrier heights
rates = Float64[]

println("Scanning over E for jump rate...")

for Ecur in Es
    ysE = simulate_trajectory(1.0, α, Ecur, Nsteps)
    ysE = ysE[burnin+1:end]
    njumps = count_jumps(ysE; thresh = 0.1, dwell = 80)
    # effective time = number of steps * Δt; in dimensionless α we have
    # α = 4 T Δt / (γ x0^2). If you want the rate in 1/Δt units, just divide by steps.
    # We'll report jumps per step here:
    rate = njumps / length(ysE)
    push!(rates, rate)
    println("E = $(Ecur): jumps = $(njumps), rate ≈ $(rate)")
end

# plot rate vs E
plt_d = plot(
    Es,
    rates,
    seriestype = :scatter,
    xlabel = "E = U₀/T",
    ylabel = "jump rate (per step)",
    title = "Part (d): jump rate vs E",
    label = "simulated"
)

# If Arrhenius: rate ~ A * exp(-E)
# We can try to fit by eye: plot rates * exp(E) to see if ~ const
display(plt_d)

# optional: transform to semi-log to see approximate Arrhenius behavior
plt_d_log = plot(
    Es,
    rates,
    seriestype = :scatter,
    yscale = :log10,
    xlabel = "E",
    ylabel = "jump rate (log)",
    title = "Part (d): semi-log plot (Arrhenius check)",
    label = "simulated"
)
display(plt_d_log)

############################################################
# (d) jump rate vs E -- easier-to-trigger version
############################################################

# classify a point as being in left / right / middle
# well_cut ~ how far from 0 we call "definitely in a well"
function well_side(y; well_cut = 0.7)
    if y > well_cut
        return 1   # right well
    elseif y < -well_cut
        return -1  # left well
    else
        return 0   # in between
    end
end

# detect jumps: we only count L -> R or R -> L when we have
# minstay consecutive samples in the new well
function count_jumps(ys; well_cut = 0.7, minstay = 10)
    current = well_side(ys[1]; well_cut=well_cut)
    jumps = 0
    stay = 1
    for k in 2:length(ys)
        s = well_side(ys[k]; well_cut=well_cut)
        if s == current && s != 0
            stay += 1
        elseif s != 0 && s != current
            # we entered the other well: check we can stay there
            # look ahead up to minstay steps (bounded by array length)
            ok = true
            last = min(k + minstay - 1, length(ys))
            for j in k:last
                if well_side(ys[j]; well_cut=well_cut) != s
                    ok = false
                    break
                end
            end
            if ok
                jumps += 1
                current = s
                stay = minstay
            else
                # treat as noise, do nothing
            end
        else
            # in the middle
            stay = 1
        end
    end
    return jumps
end

# scan over E: start from smaller barriers
Es = 0.5:0.5:5.0
rates = Float64[]
burnin = 20_000

println("Scanning over E for jump rate...")
for Ecur in Es
    ysE = simulate_trajectory(1.0, α, Ecur, Nsteps)
    ysE = ysE[burnin+1:end]
    njumps = count_jumps(ysE; well_cut=0.7, minstay=10)
    rate = njumps / length(ysE)     # jumps per step
    push!(rates, rate)
    println("E = $(Ecur): jumps = $(njumps), rate ≈ $(rate)")
end

# plot (linear)
using Plots
plt_d = plot(
    Es, rates,
    seriestype = :scatter,
    xlabel = "E = U₀/T",
    ylabel = "jump rate (per step)",
    title = "Jump rate vs E",
    label = "simulate"
)
display(plt_d)

# plot (semilog) but only for positive rates
pos_idx = findall(>(0), rates)
if !isempty(pos_idx)
    plt_d_log = plot(
        Es[pos_idx], rates[pos_idx],
        seriestype = :scatter,
        # yscale = :log10,
        xlabel = "E",
        ylabel = "jump rate",
        # title = "Arrhenius",
        label = "simulate"
    )
    display(plt_d_log)
end

