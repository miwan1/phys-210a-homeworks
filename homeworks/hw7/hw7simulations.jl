import Pkg
Pkg.add ("StatsBase", "Plots", "Statistics", "Random")
using Random
using Statistics
using StatsBase
using Plots

# Simulate one run and return α_N = A/N,
# where A is the size of the component containing paper 1.
function simulate_alpha(N::Int; rng=Random.GLOBAL_RNG)
    A = 1
    B = 1
    total = 2

    while total < N
        # New paper cites a uniformly random existing paper:
        # probability to land in component A is A/total
        if rand(rng) < A / total
            A += 1
        else
            B += 1
        end
        total += 1
    end

    return A / total
end

# Parameters
N = 1000
M = 20000  # number of Monte Carlo runs

rng = MersenneTwister(1234)

alphas = [simulate_alpha(N; rng=rng) for _ in 1:M]

println("Mean α_N ≈ ", mean(alphas))
println("Var  α_N ≈ ", var(alphas))

# Histogram
histogram(
    alphas;
    bins=60,
    normalize=true,
    xlabel="α_N",
    ylabel="Density",
    title="Histogram of α_N for N=$N over M=$M runs",
    legend=false
)

# Optional: overlay a Uniform(0,1) density for visual comparison
plot!(0:0.01:1, ones(101); lw=2, label="Uniform(0,1)")

savefig("alpha_hist_N1000.png")
println("Saved plot to alpha_hist_N1000.png")
